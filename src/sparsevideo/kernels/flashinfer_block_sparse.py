"""Variable-block-size and BSR sparse attention via flashinfer.

Provides two public functions:
  - variable_block_sparse_attn  — for SVG2, SVOO, Draft, AdaCluster (cluster-based)
  - bsr_sparse_attn             — for Radial, STA (fixed-block BSR)

Ported from:
  training_free/SVOO/svoo/co_clustering.py          (dynamic_block_sparse_fwd_flashinfer)
  training_free/SVOO/svoo/utils/flashinfer_sparse.py (_block_mask_map_to_expanded_indices,
                                                       _memory_efficient_plan, wrapper factory)
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _cuda_root_has_runtime_library(root: Path) -> bool:
    for lib_dir in (
        root / "lib64",
        root / "lib",
        root / "targets" / "x86_64-linux" / "lib",
    ):
        if any(lib_dir.glob("libcudart.so*")):
            return True
    return False


def _cuda_root_has_toolkit(root: Path) -> bool:
    return (
        (root / "bin" / "nvcc").exists()
        and (
            (root / "include" / "cuda_runtime.h").exists()
            or (root / "targets" / "x86_64-linux" / "include" / "cuda_runtime.h").exists()
        )
        and _cuda_root_has_runtime_library(root)
    )


def _candidate_cuda_roots():
    seen = set()

    def emit(root):
        root = Path(root).expanduser().resolve()
        if root in seen:
            return
        seen.add(root)
        yield root

    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            yield from emit(value)

    pytorch_nvcc = os.environ.get("PYTORCH_NVCC")
    if pytorch_nvcc:
        yield from emit(Path(pytorch_nvcc).expanduser().resolve().parents[1])

    prefixes = [Path(sys.prefix).resolve(), *Path(sys.executable).resolve().parents]
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if base_prefix not in prefixes:
        prefixes.append(base_prefix)
    for root in prefixes:
        yield from emit(root)

    nvcc = shutil.which("nvcc")
    if nvcc:
        yield from emit(Path(nvcc).resolve().parents[1])

    yield from emit(Path("/usr/local/cuda"))


def _configure_cuda_env(root: Path) -> None:
    os.environ["CUDA_HOME"] = str(root)
    os.environ["CUDA_PATH"] = str(root)
    os.environ["PYTORCH_NVCC"] = str(root / "bin" / "nvcc")
    bin_dir = str(root / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


def _ensure_cuda_home_for_flashinfer_jit() -> None:
    for root in _candidate_cuda_roots():
        if _cuda_root_has_toolkit(root):
            _configure_cuda_env(root)
            return


def _load_flashinfer():
    """Load the vendored sparsevideo_flashinfer bundled with sparsevideo."""
    _ensure_cuda_home_for_flashinfer_jit()
    _vendor = Path(__file__).resolve().parent / "_flashinfer"
    if _vendor.exists() and str(_vendor) not in sys.path:
        sys.path.insert(0, str(_vendor))
    import sparsevideo_flashinfer.sparse as _sparse
    import sparsevideo_flashinfer as _fi
    return _fi, _sparse


_flashinfer, _fi_sparse = _load_flashinfer()
HAS_FLASHINFER = True


def get_flashinfer():
    return _flashinfer


def _env_int_first(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _env_str_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _env_flag_first(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        return value.lower() not in ("0", "", "false", "no", "off")
    return default


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


def _flashinfer_kernel_head_dim(dtype: torch.dtype, head_dim: int) -> int:
    if dtype in (torch.float16, torch.bfloat16) and head_dim & (head_dim - 1):
        padded = _next_power_of_two(head_dim)
        if padded <= 256:
            return padded
    return head_dim


def _trim_head_dim(result, head_dim: int):
    if isinstance(result, tuple):
        out, lse = result
        return out[..., :head_dim].contiguous(), lse
    return result[..., :head_dim].contiguous()


# ---------------------------------------------------------------------------
# Triton helper — fill token-level KV indices for variable block sparse
# Ported verbatim from SVOO/svoo/utils/flashinfer_sparse.py
# ---------------------------------------------------------------------------

@triton.jit
def _fill_variable_block_kv_indices_kernel(
    base_ptr,
    lengths_ptr,
    starts_ptr,
    out_ptr,
    num_segments,
    BLOCK_M: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)

    seg_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    valid_seg = seg_offsets < num_segments

    lengths = tl.load(lengths_ptr + seg_offsets, mask=valid_seg, other=0)
    starts = tl.load(starts_ptr + seg_offsets, mask=valid_seg, other=0)
    bases = tl.load(base_ptr + seg_offsets, mask=valid_seg, other=0)

    mask = valid_seg[:, None] & (token_offsets[None, :] < lengths[:, None])
    out_offsets = starts[:, None] + token_offsets[None, :]
    values = bases[:, None] + token_offsets[None, :]
    tl.store(out_ptr + out_offsets, values, mask=mask)


def _block_mask_map_to_expanded_indices(
    block_mask_map: torch.Tensor,  # [BH, nqc, nkc] bool, on device
    block_col_sz: torch.Tensor,    # [BH, nkc] int32, on device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert block mask + cluster sizes to CSR-style (kv_indptr, kv_indices).

    Ported from SVOO/svoo/utils/flashinfer_sparse.py.
    """
    device = block_mask_map.device
    dtype_i = torch.int32

    row_lengths = (
        block_mask_map.to(dtype_i) * block_col_sz[:, None, :].to(dtype_i)
    ).sum(-1, dtype=dtype_i)
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=device),
            torch.cumsum(row_lengths.flatten(), 0, dtype=dtype_i),
        ],
        dim=0,
    )

    col_offset = (
        torch.cumsum(block_col_sz.to(dtype_i), 1, dtype=dtype_i)
        - block_col_sz.to(dtype_i)
    )
    head_len = block_col_sz.sum(1, dtype=dtype_i)
    head_offset = torch.cumsum(head_len, 0, dtype=dtype_i) - head_len

    h_idx, _, c_idx = block_mask_map.nonzero(as_tuple=True)
    lengths = block_col_sz[h_idx, c_idx].to(dtype_i)
    base = head_offset[h_idx] + col_offset[h_idx, c_idx]

    if lengths.numel() == 0:
        kv_indices = torch.empty((0,), dtype=dtype_i, device=device)
        return kv_indptr, kv_indices

    starts = torch.cumsum(lengths, 0, dtype=dtype_i) - lengths
    total = int(kv_indptr[-1].item())
    kv_indices = torch.empty((total,), dtype=dtype_i, device=device)
    if total > 0:
        block_m = 16
        block_l = 128
        grid = (
            triton.cdiv(lengths.numel(), block_m),
            triton.cdiv(int(lengths.max().item()), block_l),
        )
        _fill_variable_block_kv_indices_kernel[grid](
            base,
            lengths,
            starts,
            kv_indices,
            lengths.numel(),
            BLOCK_M=block_m,
            BLOCK_L=block_l,
            num_warps=8,
        )

    return kv_indptr, kv_indices


# ---------------------------------------------------------------------------
# Memory-efficient plan — patches flashinfer's VariableBlockSparseAttentionWrapper
# Ported verbatim from SVOO/svoo/utils/flashinfer_sparse.py
# ---------------------------------------------------------------------------

def _memory_efficient_plan(
    self,
    flashinfer_sparse,
    block_mask_map: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = False,
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
) -> None:
    q_data_type = flashinfer_sparse.canonicalize_torch_dtype(q_data_type)
    if kv_data_type is None:
        kv_data_type = q_data_type
    kv_data_type = flashinfer_sparse.canonicalize_torch_dtype(kv_data_type)
    self._o_dtype = q_data_type

    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    num_blocks_row = block_row_sz.shape[-1]
    num_blocks_col = block_col_sz.shape[-1]

    qo_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
            torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
    last_block_len = torch.full(
        (num_blocks_row * num_kv_heads,),
        1,
        dtype=torch.int32,
        device=block_mask_map.device,
    )

    kv_indptr, kv_indices = _block_mask_map_to_expanded_indices(block_mask_map, block_col_sz)
    kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
    kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

    self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
    self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
    self._paged_kv_indices_buf = kv_indices.to(self.device, non_blocking=non_blocking)
    self._paged_kv_last_page_len = last_block_len.to(self.device, non_blocking=non_blocking)
    torch.cuda.synchronize()

    self._mask_mode = (
        flashinfer_sparse.MaskMode.CAUSAL.value
        if causal
        else flashinfer_sparse.MaskMode.NON_CAUSAL.value
    )

    assert num_qo_heads % num_kv_heads == 0
    assert num_blocks_row * num_kv_heads + 1 == kv_indptr_host.shape[0]
    assert kv_indptr_host[-1].item() == kv_indices_host.shape[0]
    assert num_kv_heads == block_mask_map.shape[0]
    assert num_kv_heads == block_row_sz.shape[0]
    assert num_kv_heads == block_col_sz.shape[0]
    assert num_blocks_row == block_mask_map.shape[1]
    assert num_blocks_col == block_mask_map.shape[2]

    if self._backend == "auto":
        self._backend = flashinfer_sparse.determine_attention_backend(
            self.device,
            flashinfer_sparse.PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            self._mask_mode == flashinfer_sparse.MaskMode.CUSTOM.value,
            q_data_type,
            kv_data_type,
        )

    get_module_args = (
        q_data_type,
        kv_data_type,
        self._o_dtype,
        kv_indptr_host.dtype,
        head_dim,
        head_dim,
        flashinfer_sparse.PosEncodingMode[pos_encoding_mode].value,
        False,
        logits_soft_cap > 0,
        use_fp16_qk_reduction,
    )
    self._cached_module = flashinfer_sparse.get_batch_prefill_module(
        self._backend, *get_module_args
    )

    kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]
    required_size = len(kv_lens_arr_host)
    if required_size > self._kv_lens_buffer.shape[0]:
        self._kv_lens_buffer = torch.empty(
            (required_size,), dtype=torch.int32, device=self.device
        )
    self._kv_lens_buffer[:required_size].copy_(kv_lens_arr_host)

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_arr_host,
        qo_indptr_host[-1].item(),
        num_blocks_row * num_kv_heads,
        num_qo_heads // num_kv_heads,
        1,
        1,
        False,
        head_dim,
        head_dim,
        causal,
        -1,  # window_left (disabled)
    ]
    if self._backend == "fa2":
        args.append(-1)    # fixed_split_size
        args.append(False) # disable_split_kv
        args.append(0)     # num_colocated_ctas
    try:
        self._plan_info = self._cached_module.plan(*args)
    except TypeError as exc:
        if "expected at most 15" not in str(exc) and "received 19" not in str(exc):
            raise
        self._plan_info = self._cached_module.plan(*args[:15])

    self._pos_encoding_mode = pos_encoding_mode
    self._use_fp16_qk_reduction = use_fp16_qk_reduction
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
    self._num_kv_heads = num_kv_heads
    self._gqa_group_size = num_qo_heads // num_kv_heads


def _patch_variable_block_sparse_wrapper(flashinfer_sparse) -> None:
    cls = flashinfer_sparse.VariableBlockSparseAttentionWrapper
    if getattr(cls, "_sv_mem_efficient_patched", False):
        return

    original_plan = cls.plan

    def plan(self, *args, **kwargs):
        if _env_flag_first(
            ("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN", "SV_FLASHINFER_MEM_EFFICIENT_PLAN"),
            True,
        ):
            return _memory_efficient_plan(self, flashinfer_sparse, *args, **kwargs)
        return original_plan(self, *args, **kwargs)

    cls._sv_original_plan = original_plan
    cls.plan = plan
    cls._sv_mem_efficient_patched = True


def _make_variable_block_sparse_wrapper(f_buffer, backend="auto"):
    _patch_variable_block_sparse_wrapper(_fi_sparse)
    return _fi_sparse.VariableBlockSparseAttentionWrapper(f_buffer, backend=backend)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _variable_block_sparse_plan_chunks(
    block_mask_map: torch.Tensor,
    block_col_sz: torch.Tensor,
    max_kv_indices: int,
) -> list[tuple[int, int]]:
    if max_kv_indices <= 0 or block_mask_map.shape[0] <= 1:
        return [(0, int(block_mask_map.shape[0]))]

    work = (
        block_mask_map.to(torch.int64)
        * block_col_sz.to(torch.int64)[:, None, :]
    ).sum(dim=(1, 2))
    work_cpu = [int(item) for item in work.detach().cpu().tolist()]

    chunks: list[tuple[int, int]] = []
    start = 0
    current = 0
    for idx, amount in enumerate(work_cpu):
        if idx > start and current + amount > max_kv_indices:
            chunks.append((start, idx))
            start = idx
            current = 0
        current += amount
    chunks.append((start, len(work_cpu)))
    return chunks


def _variable_block_sparse_attn_impl(
    q: torch.Tensor,             # [BH, S, D]  (batch × heads already folded)
    k: torch.Tensor,             # [BH, S, D]
    v: torch.Tensor,             # [BH, S, D]
    dynamic_map: torch.Tensor,   # [BH, nqc, nkc] bool (CPU or GPU)
    q_sizes: torch.Tensor,       # [BH, nqc] int
    k_sizes: torch.Tensor,       # [BH, nkc] int
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # [BH, S, D], optional [BH, S] LSE
    """Variable-block-size sparse attention via flashinfer.

    Used by SVG2, SVOO, Draft, Radial, AdaCluster.
    dynamic_map[bh, i, j] = True means Q-cluster i in head bh attends to K-cluster j.
    Tokens must already be sorted by cluster within each BH slot.
    """

    BH, S, D = q.shape
    nqc = q_sizes.shape[-1]
    nkc = k_sizes.shape[-1]

    assert dynamic_map.shape == (BH, nqc, nkc)

    dev = q.device
    workspace_bytes = _env_int_first(
        ("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", "SV_FLASHINFER_WORKSPACE_BYTES"),
        128 * 1024 * 1024,
    )
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=dev)
    backend = _env_str_first(
        ("SVOO_FLASHINFER_SPARSE_BACKEND", "SV_FLASHINFER_BACKEND"),
        "auto",
    ).lower()

    _ensure_cuda_home_for_flashinfer_jit()
    wrapper = _make_variable_block_sparse_wrapper(f_buffer, backend=backend)
    int_workspace_bytes = _env_int_first(
        ("SVOO_FLASHINFER_SPARSE_INT_WORKSPACE_BYTES", "SV_FLASHINFER_INT_WORKSPACE_BYTES"),
        0,
    )
    i_buffer = None
    if int_workspace_bytes > 0:
        i_buffer = torch.empty((int_workspace_bytes,), dtype=torch.uint8, device=dev)
        wrapper.reset_workspace_buffer(
            float_workspace_buffer=f_buffer,
            int_workspace_buffer=i_buffer,
        )

    wrapper.plan(
        block_mask_map=dynamic_map,
        block_row_sz=q_sizes,
        block_col_sz=k_sizes,
        num_qo_heads=BH,
        num_kv_heads=BH,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        sm_scale=sm_scale,
    )

    o = wrapper.run(q, k, v, return_lse=return_lse)
    del wrapper, f_buffer, i_buffer
    return o


def variable_block_sparse_attn(
    q: torch.Tensor,             # [BH, S, D]  (batch × heads already folded)
    k: torch.Tensor,             # [BH, S, D]
    v: torch.Tensor,             # [BH, S, D]
    dynamic_map: torch.Tensor,   # [BH, nqc, nkc] bool (CPU or GPU)
    q_sizes: torch.Tensor,       # [BH, nqc] int
    k_sizes: torch.Tensor,       # [BH, nkc] int
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # [BH, S, D], optional [BH, S] LSE
    """Variable-block-size sparse attention via flashinfer.

    Used by SVG2, SVOO, Draft, Radial, AdaCluster.
    dynamic_map[bh, i, j] = True means Q-cluster i in head bh attends to K-cluster j.
    Tokens must already be sorted by cluster within each BH slot.
    """

    BH, _S, original_head_dim = q.shape
    nqc = q_sizes.shape[-1]
    nkc = k_sizes.shape[-1]

    assert dynamic_map.shape == (BH, nqc, nkc)

    kernel_head_dim = _flashinfer_kernel_head_dim(q.dtype, original_head_dim)
    if kernel_head_dim != original_head_dim:
        if sm_scale is None:
            sm_scale = original_head_dim ** -0.5
        pad = kernel_head_dim - original_head_dim
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))

    dev = q.device
    dynamic_map_dev = dynamic_map.bool().to(dev)
    q_sizes_dev = q_sizes.to(torch.int32).to(dev)
    k_sizes_dev = k_sizes.to(torch.int32).to(dev)

    max_kv_indices = _env_int_first(
        ("SV_FLASHINFER_MAX_KV_INDICES_PER_PLAN", "SVOO_FLASHINFER_MAX_KV_INDICES_PER_PLAN"),
        256_000_000,
    )
    chunks = _variable_block_sparse_plan_chunks(dynamic_map_dev, k_sizes_dev, max_kv_indices)
    if len(chunks) == 1:
        result = _variable_block_sparse_attn_impl(
            q, k, v,
            dynamic_map_dev, q_sizes_dev, k_sizes_dev,
            sm_scale=sm_scale,
            return_lse=return_lse,
        )
        if kernel_head_dim != original_head_dim:
            return _trim_head_dim(result, original_head_dim)
        return result

    outputs = []
    lses = []
    for start, end in chunks:
        result = _variable_block_sparse_attn_impl(
            q[start:end],
            k[start:end],
            v[start:end],
            dynamic_map_dev[start:end],
            q_sizes_dev[start:end],
            k_sizes_dev[start:end],
            sm_scale=sm_scale,
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = result
            outputs.append(out)
            lses.append(lse)
        else:
            outputs.append(result)

    out = torch.cat(outputs, dim=0)
    if kernel_head_dim != original_head_dim:
        out = out[..., :original_head_dim].contiguous()
    if return_lse:
        return out, torch.cat(lses, dim=0)
    return out


def hunyuan_flashinfer_varlen_attn(
    q: torch.Tensor,             # [BH, S, D]
    k: torch.Tensor,             # [BH, S, D]
    v: torch.Tensor,             # [BH, S, D]
    valid_len: int,
) -> torch.Tensor:               # [BH, S, D]
    """HunyuanVideo two-segment varlen attention via FlashInfer.

    Ported from SVOO/svoo/models/hunyuan10/attention.py::flashinfer_varlen_func.
    The first segment contains valid video+prompt tokens; the second segment
    contains padded prompt tokens that only attend to themselves.
    """

    BH, S, D = q.shape
    valid = max(0, min(int(valid_len), int(S)))
    padded = int(S) - valid
    dev = q.device

    block_mask_map = torch.tensor(
        [[True, False], [False, True]],
        device=dev,
        dtype=torch.bool,
    ).expand(BH, 2, 2).contiguous()
    block_sizes = torch.tensor([valid, padded], device=dev, dtype=torch.int32)
    block_sizes = block_sizes.expand(BH, 2).contiguous()

    workspace_bytes = _env_int_first(
        ("SVOO_FLASHINFER_VARLEN_WORKSPACE_BYTES", "SV_FLASHINFER_VARLEN_WORKSPACE_BYTES"),
        128 * 1024 * 1024,
    )
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=dev)

    _ensure_cuda_home_for_flashinfer_jit()
    wrapper = _make_variable_block_sparse_wrapper(f_buffer, backend="auto")
    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_sizes,
        block_col_sz=block_sizes,
        num_qo_heads=BH,
        num_kv_heads=BH,
        head_dim=D,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)
    del wrapper, f_buffer
    return o


def bsr_sparse_attn(
    q: torch.Tensor,       # [S, num_heads, head_dim]  (SHD layout)
    k: torch.Tensor,       # [S, num_heads, head_dim]
    v: torch.Tensor,       # [S, num_heads, head_dim]
    indptr: torch.Tensor,  # [num_row_blocks + 1] int32
    indices: torch.Tensor, # [nnz] int32
    M: int,                # query sequence length
    N: int,                # key sequence length
    num_heads: int,
    head_dim: int,
    block_size: int = 128,
) -> torch.Tensor:         # [S, num_heads, head_dim]
    """Fixed-block BSR sparse attention via flashinfer.BlockSparseAttentionWrapper.

    Used by Radial, STA.
    """

    workspace_bytes = int(os.environ.get("SV_FLASHINFER_WORKSPACE_BYTES", str(128 * 1024 * 1024)))
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=q.device)

    wrapper = _fi_sparse.BlockSparseAttentionWrapper(f_buffer)
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        M=M,
        N=N,
        R=block_size,
        C=block_size,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    o = wrapper.run(q, k, v)
    del wrapper, f_buffer
    return o


def build_bsr_from_mask(
    mask: torch.Tensor,  # [num_row_blocks, num_col_blocks] bool
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D dense block mask to BSR (indptr, indices)."""
    dtype_i = torch.int32
    num_row = mask.shape[0]
    row_nnz = mask.to(dtype_i).sum(dim=1)
    indptr = torch.cat(
        [
            torch.zeros(1, dtype=dtype_i, device=row_nnz.device),
            torch.cumsum(row_nnz, 0, dtype=dtype_i),
        ],
        dim=0,
    ).to(device)
    indices = mask.nonzero(as_tuple=False)[:, 1].to(dtype_i).to(device)
    return indptr, indices


# ---------------------------------------------------------------------------
# Error-Aware Reduction (EAR) pruned block-sparse attention
#
# Ported from Sparse-VideoGen (SVG-EAR, branch ear-wan22-support). Selected
# blocks (mask == 1) run exact token attention via the package's FlashInfer
# variable-block-sparse path; pruned blocks (mask == 0) are approximated by
# their key/value centroids and merged back with an online-softmax update so
# their contribution is recovered rather than dropped.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_qc_kernel_opt(
    Q,
    K_centroids,
    V_centroids,
    block_mask_map,
    block_col_sz,
    O_flash,
    LSE_flash,
    QC_INDPTR,
    LSE_final,
    Out,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_ch,
    stride_cn,
    stride_cd,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mk,
    stride_szb,
    stride_szh,
    stride_szn,
    stride_ipb,
    stride_iph,
    stride_ipn,
    sm_scale,
    B,
    H,
    S,
    D: tl.constexpr,
    KC_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one query cluster for one (batch, head) pair.
    qc_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    batch_idx = bh_idx // H
    head_idx = bh_idx % H

    # Query cluster boundaries are stored as an indptr array.
    qc_indptr_ptr = QC_INDPTR + batch_idx * stride_ipb + head_idx * stride_iph + qc_idx * stride_ipn
    start_s = tl.load(qc_indptr_ptr)
    end_s = tl.load(qc_indptr_ptr + 1)

    LN2 = 0.69314718056
    # Pad the head-dim block to a power of 2 (tl.arange requirement) and mask
    # offs_d < D so non-power-of-2 head dims (e.g. Allegro's 96) compile correctly.
    offs_d = tl.arange(0, BLOCK_D)

    # These bases point to the metadata and inputs for the current query cluster.
    mask_base = block_mask_map + batch_idx * stride_mb + head_idx * stride_mh + qc_idx * stride_mq
    size_base = block_col_sz + batch_idx * stride_szb + head_idx * stride_szh

    q_base = Q + bh_idx * stride_qh
    o_flash_base = O_flash + bh_idx * stride_qh
    lse_flash_base = LSE_flash + bh_idx * S
    k_centroids_base = K_centroids + bh_idx * stride_ch
    v_centroids_base = V_centroids + bh_idx * stride_ch

    for m_start in range(start_s, end_s, BLOCK_M):
        # Process the current query cluster in BLOCK_M-sized chunks.
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < end_s

        # FlashInfer returns logsumexp in log2 space, so convert it back to natural log space.
        lse_f = tl.load(lse_flash_base + m_offsets, mask=m_mask, other=-float("inf"))
        m_i = lse_f * LN2
        l_i = tl.where(m_mask, 1.0, 0.0)

        q_ptrs = q_base + m_offsets[:, None] * stride_qs + offs_d[None, :]
        q = tl.load(q_ptrs, mask=m_mask[:, None] & (offs_d[None, :] < D), other=0.0)

        # Initialize the accumulator from the token-level FlashInfer output.
        acc_ptrs = o_flash_base + m_offsets[:, None] * stride_qs + offs_d[None, :]
        acc = tl.load(acc_ptrs, mask=m_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        for n_start in range(0, KC_NUM, BLOCK_N):
            # Iterate over centroid blocks and only keep entries with mask == 0.
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < KC_NUM

            k_ptrs = k_centroids_base + n_offsets[:, None] * stride_cn + offs_d[None, :]
            k = tl.load(k_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < D), other=0.0)

            scores = tl.dot(q, tl.trans(k)) * sm_scale

            # Cluster sizes act as log-priors for centroid attention.
            weights = tl.load(size_base + n_offsets * stride_szn, mask=n_mask, other=0.0)
            scores += tl.math.log(weights[None, :] + 1e-6)

            masks = tl.load(mask_base + n_offsets * stride_mk, mask=n_mask, other=1)
            scores = tl.where((masks == 0)[None, :] & n_mask[None, :], scores, -float("inf"))

            # Merge centroid attention with the existing FlashInfer statistics using stable softmax updates.
            m_ij = tl.max(scores, axis=1)
            m_next = tl.maximum(m_i, m_ij)

            alpha = tl.math.exp(m_i - m_next)
            p = tl.math.exp(scores - m_next[:, None])

            l_tile = tl.sum(p, axis=1)
            l_i_next = l_i * alpha + l_tile

            v_ptrs = v_centroids_base + n_offsets[:, None] * stride_cn + offs_d[None, :]
            v = tl.load(v_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < D), other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

            m_i = m_next
            l_i = l_i_next

        # Normalize the merged accumulator back to the final attention output.
        out_final = acc / l_i[:, None]

        tl.store(LSE_final + bh_idx * S + m_offsets, m_i + tl.math.log(l_i), mask=m_mask)
        tl.store(
            Out + bh_idx * stride_qh + m_offsets[:, None] * stride_qs + offs_d[None, :],
            out_final.to(Out.dtype.element_ty),
            mask=m_mask[:, None] & (offs_d[None, :] < D),
        )


def dynamic_block_sparse_prune_fwd_flashinfer(
    q: torch.Tensor,             # [B, H, S, D]
    k: torch.Tensor,             # [B, H, S, D]
    v: torch.Tensor,             # [B, H, S, D]
    k_centroids: torch.Tensor,   # [B, H, kc, D]
    v_centroids: torch.Tensor,   # [B, H, kc, D]
    block_mask_map: torch.Tensor,  # [B, H, qc, kc] bool — selected blocks (FlashInfer)
    block_row_sz: torch.Tensor,    # [B, H, qc] int
    block_col_sz: torch.Tensor,    # [B, H, kc] int
    sm_scale: Optional[float] = None,
    prune_mask: Optional[torch.Tensor] = None,  # [B, H, qc, kc] bool — blocks NOT approximated by centroids
) -> torch.Tensor:               # [B, H, S, D]
    """Run selected-block token attention via FlashInfer, then fuse the pruned
    blocks' centroid attention with Triton (EAR path).

    ``block_mask_map`` (mask == 1) selects blocks that get exact token attention.
    The centroid-approximation step approximates blocks where ``prune_mask`` == 0.
    When ``prune_mask`` is None it defaults to ``block_mask_map`` (every non-selected
    block is approximated). Passing a distinct ``prune_mask`` lets callers fully
    exclude certain blocks (e.g. video→padding-text) from BOTH paths.
    """
    B, H, S, D = q.shape
    qc_num, kc_num = block_mask_map.shape[-2:]
    scale = sm_scale if sm_scale is not None else D ** -0.5
    num_heads = B * H

    assert block_mask_map.shape == (B, H, qc_num, kc_num)
    if prune_mask is None:
        prune_mask = block_mask_map
    assert prune_mask.shape == (B, H, qc_num, kc_num)

    # --- Part 1: FlashInfer for selected blocks (mask == 1) ---
    # Reuse the package's variable-block-sparse path (head-dim padding, chunking,
    # memory-efficient plan). LSE comes back in the same log2 space the fused
    # kernel expects.
    o_flash, lse_flash = variable_block_sparse_attn(
        q.reshape(num_heads, S, D),
        k.reshape(num_heads, S, D),
        v.reshape(num_heads, S, D),
        block_mask_map.reshape(num_heads, qc_num, kc_num),
        block_row_sz.reshape(num_heads, qc_num),
        block_col_sz.reshape(num_heads, kc_num),
        sm_scale=scale,
        return_lse=True,
    )
    o_flash = o_flash.contiguous()
    lse_flash = lse_flash.contiguous()

    # --- Part 2: Triton for pruned blocks via centroids (mask == 0) ---
    qc_indptr = torch.zeros((B, H, qc_num + 1), device=q.device, dtype=torch.int32)
    qc_indptr[..., 1:] = torch.cumsum(block_row_sz.to(q.device), dim=-1)

    q_flat = q.reshape(num_heads, S, D).contiguous()
    k_centroids_flat = k_centroids.reshape(num_heads, kc_num, D).contiguous()
    v_centroids_flat = v_centroids.reshape(num_heads, kc_num, D).contiguous()
    block_col_sz = block_col_sz.contiguous()
    prune_mask = prune_mask.contiguous()

    o_final = torch.empty((num_heads, S, D), device=q.device, dtype=q.dtype)
    lse_final = torch.empty((num_heads, S), device=q.device, dtype=torch.float32)
    grid = (qc_num, num_heads)

    _fused_qc_kernel_opt[grid](
        Q=q_flat,
        K_centroids=k_centroids_flat,
        V_centroids=v_centroids_flat,
        block_mask_map=prune_mask,
        block_col_sz=block_col_sz,
        O_flash=o_flash,
        LSE_flash=lse_flash,
        QC_INDPTR=qc_indptr,
        LSE_final=lse_final,
        Out=o_final,
        stride_qh=q_flat.stride(0),
        stride_qs=q_flat.stride(1),
        stride_qd=q_flat.stride(2),
        stride_ch=k_centroids_flat.stride(0),
        stride_cn=k_centroids_flat.stride(1),
        stride_cd=k_centroids_flat.stride(2),
        stride_mb=prune_mask.stride(0),
        stride_mh=prune_mask.stride(1),
        stride_mq=prune_mask.stride(2),
        stride_mk=prune_mask.stride(3),
        stride_szb=block_col_sz.stride(0),
        stride_szh=block_col_sz.stride(1),
        stride_szn=block_col_sz.stride(2),
        stride_ipb=qc_indptr.stride(0),
        stride_iph=qc_indptr.stride(1),
        stride_ipn=qc_indptr.stride(2),
        sm_scale=scale,
        B=B,
        H=H,
        S=S,
        D=D,
        KC_NUM=kc_num,
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
        num_stages=2,
    )

    return o_final.reshape(B, H, S, D)
