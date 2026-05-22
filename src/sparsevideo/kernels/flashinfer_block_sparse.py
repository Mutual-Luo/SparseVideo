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

try:
    import flashinfer.sparse as _fi_sparse
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


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
    if dtype == torch.bfloat16 and head_dim & (head_dim - 1):
        padded = _next_power_of_two(head_dim)
        if padded <= 256:
            return padded
    return head_dim


def _trim_head_dim(result, head_dim: int):
    if isinstance(result, tuple):
        out, lse = result
        return out[..., :head_dim].contiguous(), lse
    return result[..., :head_dim].contiguous()


def _cuda_root_has_toolkit(root: Path) -> bool:
    return (
        (root / "bin" / "nvcc").exists()
        and (
            (root / "include" / "cuda_runtime.h").exists()
            or (root / "targets" / "x86_64-linux" / "include" / "cuda_runtime.h").exists()
        )
    )


def _candidate_cuda_roots():
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            yield Path(value).expanduser()

    nvcc = shutil.which("nvcc")
    if nvcc:
        yield Path(nvcc).resolve().parents[1]

    prefixes = [Path(sys.prefix).resolve()]
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if base_prefix not in prefixes:
        prefixes.append(base_prefix)
    prefixes.extend(Path(sys.executable).resolve().parents)
    prefixes.append(Path("/usr/local/cuda"))

    seen = set()
    for root in prefixes:
        if root in seen:
            continue
        seen.add(root)
        yield root


def _ensure_cuda_home_for_flashinfer_jit() -> None:
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return

    for root in _candidate_cuda_roots():
        if _cuda_root_has_toolkit(root):
            os.environ["CUDA_HOME"] = str(root)
            os.environ.setdefault("CUDA_PATH", str(root))
            bin_dir = str(root / "bin")
            path = os.environ.get("PATH", "")
            if bin_dir not in path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + path
            return


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
        -1,
    ]
    if self._backend == "fa2":
        args.append(-1)
        args.append(False)
        args.append(0)
    self._plan_info = self._cached_module.plan(*args)

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
    if not HAS_FLASHINFER:
        raise ImportError("flashinfer is required for variable_block_sparse_attn")

    BH, S, D = q.shape
    nqc = q_sizes.shape[-1]
    nkc = k_sizes.shape[-1]

    assert dynamic_map.shape == (BH, nqc, nkc)

    workspace_bytes = _env_int_first(
        ("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", "SV_FLASHINFER_WORKSPACE_BYTES"),
        128 * 1024 * 1024,
    )
    f_buffer = torch.empty((workspace_bytes,), dtype=torch.uint8, device=q.device)
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
    if not HAS_FLASHINFER:
        raise ImportError("flashinfer is required for variable_block_sparse_attn")

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
    if not HAS_FLASHINFER:
        raise ImportError("flashinfer is required for hunyuan_flashinfer_varlen_attn")

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
    if not HAS_FLASHINFER:
        raise ImportError("flashinfer is required for bsr_sparse_attn")

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
