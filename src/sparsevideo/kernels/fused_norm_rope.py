"""Fused inplace kernels for RMSNorm and RoPE.

These replace the allocating PyTorch paths in the WAN and HunyuanVideo
processors. A SparseVideo-owned C++ extension (`_kernels`) is used when
available and requested; otherwise the lightweight Triton kernels below are
used. Set SPARSEVIDEO_FUSED_KERNEL_BACKEND to one of:

  auto    - use SparseVideo `_kernels` if a built extension is found, else Triton
  native  - require SparseVideo `_kernels`
  triton  - force SparseVideo Triton kernels
  pytorch - force PyTorch reference paths

Kernel variants:
  triton_rmsnorm_inplace      — inplace RMSNorm over last dim; used by HunyuanVideo
  triton_rope_wan_inplace     — inplace RoPE in WAN's complex-pair format [B,S,H,D]
  triton_rope_hyvideo_inplace — inplace RoPE for HunyuanVideo (txt-last, skips text tokens)
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl


_NATIVE_KERNELS_CHECKED = False
_NATIVE_KERNELS = None
_NATIVE_KERNELS_ERROR = None
_NATIVE_NARROW_NORM_DIMS = {32, 64, 128, 256}
_EXPECTED_NATIVE_KERNEL_OPS = {
    "apply_qk_rope_inplace_cossin",
    "apply_qk_rope_inplace_cossin_complex",
    "apply_qk_rope_inplace_cossin_txtlast",
    "layer_norm_forward",
    "rms_norm_forward",
}


def _kernel_backend() -> str:
    backend = os.environ.get("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "auto").lower()
    if backend not in {"auto", "native", "triton", "pytorch"}:
        raise ValueError(
            "SPARSEVIDEO_FUSED_KERNEL_BACKEND must be one of: auto, native, triton, pytorch"
        )
    return backend


def _candidate_native_kernel_dirs():
    env_root = os.environ.get("SPARSEVIDEO_NATIVE_KERNEL_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if "training_free" in env_path.parts:
            raise RuntimeError(
                "Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free; "
                "SparseVideo native kernels must be built under src/sparsevideo."
            )
        yield env_path

    repo_root = Path(__file__).resolve().parents[3]
    yield repo_root / "src" / "sparsevideo" / "kernels" / "native" / "build"


def _load_native_kernels(required: bool = False):
    """Load SparseVideo-owned `_kernels` lazily.

    We only touch it at runtime so basic `import sparsevideo` stays independent
    from optional compiled extensions.
    """
    global _NATIVE_KERNELS_CHECKED, _NATIVE_KERNELS, _NATIVE_KERNELS_ERROR
    if _NATIVE_KERNELS_CHECKED:
        if _NATIVE_KERNELS is None and required:
            raise ImportError(_NATIVE_KERNELS_ERROR)
        return _NATIVE_KERNELS

    _NATIVE_KERNELS_CHECKED = True
    try:
        candidate_dirs = list(_candidate_native_kernel_dirs())
        search_dirs = [str(path) for path in candidate_dirs]
        for path in reversed(candidate_dirs):
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        native = importlib.import_module("_kernels")
        module_file = getattr(native, "__file__", None)
        if module_file is None:
            raise ImportError("Imported _kernels module has no __file__; cannot verify package ownership")
        module_path = Path(module_file).resolve()
        if not any(module_path.is_relative_to(path.resolve()) for path in candidate_dirs if path.exists()):
            raise ImportError(
                "Imported _kernels from outside SparseVideo native dirs: "
                f"{module_path}. Candidate dirs: {search_dirs}"
            )
        missing_ops = sorted(_EXPECTED_NATIVE_KERNEL_OPS - set(dir(native)))
        if missing_ops:
            raise ImportError(
                "SparseVideo `_kernels` is missing expected fused ops: "
                f"{missing_ops}"
            )
        _NATIVE_KERNELS = native
    except Exception as exc:
        _NATIVE_KERNELS = None
        search_dirs = locals().get("search_dirs", [])
        _NATIVE_KERNELS_ERROR = (
            "Could not import SparseVideo-owned `_kernels` extension. Build it "
            "under src/sparsevideo/kernels/native/build, or set "
            "SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton. Searched: "
            f"{search_dirs}. Original error: {type(exc).__name__}: {exc}"
        )
        if required:
            raise ImportError(_NATIVE_KERNELS_ERROR) from exc
    return _NATIVE_KERNELS


def _should_try_native() -> bool:
    return _kernel_backend() in {"auto", "native"}


def _native_required() -> bool:
    return _kernel_backend() == "native"


def _native_rmsnorm_supported(hidden_dim: int) -> bool:
    return int(hidden_dim) in _NATIVE_NARROW_NORM_DIMS


def _requires_cuda_backend(backend: str) -> bool:
    return backend in {"native", "triton"}


def _raise_cuda_required(backend: str, op_name: str) -> None:
    raise RuntimeError(
        f"SPARSEVIDEO_FUSED_KERNEL_BACKEND={backend} requires CUDA tensors for {op_name}; "
        "set SPARSEVIDEO_FUSED_KERNEL_BACKEND=auto or pytorch only for CPU/debug fallback"
    )


def _to_device(tensor: torch.Tensor, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if tensor.device == device and (dtype is None or tensor.dtype == dtype):
        return tensor
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


# ---------------------------------------------------------------------------
# RMSNorm inplace
# Ported from: training_free/SVOO/svoo/kernels/triton/rmsnorm.py
# Adapted for inplace operation (output written back to input buffer).
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_inplace_kernel(
    X,           # input/output pointer [M, N]
    W,           # weight [N]
    x_stride,    # row stride of X
    M,           # number of rows
    N: tl.constexpr,
    N2: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    rstd = tl.reshape(rstd, (BLOCK_M, 1))

    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = (x * rstd * w).to(X.type.element_ty)
    tl.store(x_ptr, y, mask=mask)


def triton_rmsnorm_inplace(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm inplace over the last dimension.

    x: [..., D] — any leading dims, modified in place and returned.
    weight: [D]
    """
    backend = _kernel_backend()
    if backend == "pytorch":
        return _rmsnorm_pytorch(x, weight, eps)
    if not x.is_cuda:
        if _requires_cuda_backend(backend):
            _raise_cuda_required(backend, "RMSNorm")
        return _rmsnorm_pytorch(x, weight, eps)

    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D).contiguous()
    weight = _to_device(weight, x_2d.device)

    if _should_try_native():
        native = _load_native_kernels(required=_native_required())
        if native is not None and _native_rmsnorm_supported(D):
            try:
                native.rms_norm_forward(x_2d, weight, float(eps))
                return x_2d.reshape(orig_shape)
            except ValueError:
                if _native_required():
                    raise
        elif native is not None and _native_required():
            raise RuntimeError(
                "SparseVideo `_kernels` RMSNorm does not support hidden_dim="
                f"{D}; set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton only for an explicit Triton run"
            )

    M = x_2d.shape[0]

    N2 = triton.next_power_of_2(D)
    BLOCK_M = 32 if D <= 512 else 1
    grid = (triton.cdiv(M, BLOCK_M),)

    try:
        _rmsnorm_inplace_kernel[grid](
            x_2d, weight, x_2d.stride(0), M, D, N2, eps,
            BLOCK_M=BLOCK_M, num_warps=8,
        )
    except Exception:
        if _native_required():
            raise
        return _rmsnorm_pytorch(x, weight, eps)
    return x_2d.reshape(orig_shape)


def _rmsnorm_pytorch(x, weight, eps):
    orig_dtype = x.dtype
    x_f = x.float()
    weight_f = _to_device(weight, x.device, dtype=torch.float32)
    var = (x_f * x_f).mean(-1, keepdim=True)
    x_norm = x_f / (var + eps).sqrt()
    return (x_norm * weight_f).to(orig_dtype)


# ---------------------------------------------------------------------------
# RoPE inplace — WAN format
# WAN uses a "complex-pair" interleaved RoPE:
#   x_pairs = q.view(..., D//2, 2)  →  x1 = x_pairs[..., 0], x2 = x_pairs[..., 1]
#   q[..., 0::2] = x1*cos - x2*sin
#   q[..., 1::2] = x1*sin + x2*cos
# where cos[s] = freqs_cos[s, 0::2],  sin[s] = freqs_sin[s, 1::2]
# Input shapes: q/k [B, S, H, D],  freqs_cos/freqs_sin [S, D]
# ---------------------------------------------------------------------------

@triton.jit
def _rope_wan_inplace_kernel(
    Q, K,              # [B, S, H, D]
    Cos, Sin,          # [S, D]  (full D; even cols → cos, odd cols → sin)
    B, S, H,
    D: tl.constexpr,
    D2: tl.constexpr,  # D // 2
    stride_qb, stride_qs, stride_qh,
    stride_cs,
    BLOCK_S: tl.constexpr,
):
    # grid: (B * H, ceil(S / BLOCK_S))
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    d2_offs = tl.arange(0, D2)  # index into pairs

    # Load cos / sin  — shape [BLOCK_S, D2]
    c_base = Cos + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2      # even cols
    s_base = Sin + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2 + 1  # odd cols
    cs_mask = s_mask[:, None]  # broadcast over d2

    cos = tl.load(c_base, mask=cs_mask, other=0.0).to(tl.float32)
    sin = tl.load(s_base, mask=cs_mask, other=0.0).to(tl.float32)

    for buf in range(2):  # 0 = Q, 1 = K
        base = Q if buf == 0 else K
        base_ptr = base + b * stride_qb + s_offs[:, None] * stride_qs + h * stride_qh

        # Load interleaved pairs: even → x1, odd → x2
        x1_ptr = base_ptr + d2_offs[None, :] * 2
        x2_ptr = base_ptr + d2_offs[None, :] * 2 + 1
        x1 = tl.load(x1_ptr, mask=cs_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr, mask=cs_mask, other=0.0).to(tl.float32)

        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        y1 = y1.to(base.type.element_ty)
        y2 = y2.to(base.type.element_ty)
        tl.store(x1_ptr, y1, mask=cs_mask)
        tl.store(x2_ptr, y2, mask=cs_mask)


def triton_rope_wan_inplace(
    q: torch.Tensor,        # [B, S, H, D]
    k: torch.Tensor,        # [B, S, H, D]
    freqs_cos: torch.Tensor, # [1, S, 1, D] or [S, D] — any shape that reshapes to [S, D]
    freqs_sin: torch.Tensor, # same
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply WAN RoPE inplace to q and k.  Returns (q, k) (same tensors, modified)."""
    backend = _kernel_backend()
    if backend == "pytorch":
        return _rope_wan_pytorch(q, k, freqs_cos, freqs_sin)
    if not q.is_cuda:
        if _requires_cuda_backend(backend):
            _raise_cuda_required(backend, "WAN RoPE")
        return _rope_wan_pytorch(q, k, freqs_cos, freqs_sin)

    q = q.contiguous()
    k = k.contiguous()
    B, S, H, D = q.shape
    D2 = D // 2

    # Normalise cos/sin to [S, D] regardless of input shape ([1,S,1,D] or [S,D])
    cos = _to_device(freqs_cos.reshape(S, -1).contiguous(), q.device, dtype=torch.float32)
    sin = _to_device(freqs_sin.reshape(S, -1).contiguous(), q.device, dtype=torch.float32)

    if _should_try_native():
        native = _load_native_kernels(required=_native_required())
        if native is not None:
            if cos.shape[1] == D:
                cos_native = cos[:, 0::2].contiguous()
            else:
                cos_native = cos.contiguous()
            if sin.shape[1] == D:
                sin_native = sin[:, 1::2].contiguous()
            else:
                sin_native = sin.contiguous()
            if cos_native.shape[1] != D2 or sin_native.shape[1] != D2:
                if _native_required():
                    raise RuntimeError(
                        "SparseVideo `_kernels` WAN RoPE expects cos/sin width head_dim/2 "
                        f"after normalization; got {cos_native.shape[1]} and {sin_native.shape[1]}"
                    )
            else:
                q_bhsd = q.permute(0, 2, 1, 3).contiguous()
                k_bhsd = k.permute(0, 2, 1, 3).contiguous()
                native.apply_qk_rope_inplace_cossin_complex(q_bhsd, k_bhsd, cos_native, sin_native, 0)
                return (
                    q_bhsd.permute(0, 2, 1, 3).contiguous(),
                    k_bhsd.permute(0, 2, 1, 3).contiguous(),
                )

    BLOCK_S = min(64, triton.next_power_of_2(S))
    grid = (B * H, triton.cdiv(S, BLOCK_S))

    _rope_wan_inplace_kernel[grid](
        q, k, cos, sin,
        B, S, H, D, D2,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0),
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return q, k


def _rope_wan_pytorch(q, k, freqs_cos, freqs_sin):
    # freqs_cos/sin: any shape broadcastable to [B, S, H, D//2] after extracting pairs
    # Normalise to [S, D] then unsqueeze to [1, S, 1, D//2] for broadcasting
    S = q.shape[1]
    cos = _to_device(
        freqs_cos.reshape(S, -1)[..., 0::2].unsqueeze(0).unsqueeze(2),
        q.device,
        dtype=torch.float32,
    )  # [1, S, 1, D//2]
    sin = _to_device(
        freqs_sin.reshape(S, -1)[..., 1::2].unsqueeze(0).unsqueeze(2),
        q.device,
        dtype=torch.float32,
    )
    def _apply(hidden):
        x1, x2 = hidden.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D//2]
        out = torch.empty_like(hidden)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden)
    return _apply(q), _apply(k)


# ---------------------------------------------------------------------------
# RoPE inplace — HunyuanVideo format (txt-last)
# HunyuanVideo uses diffusers apply_rotary_emb format:
#   image_rotary_emb = (cos, sin)  where cos/sin are [1, S_vid, H, D/2] or [S, D/2]
# Only applies RoPE to the first (S - txt_len) tokens (video tokens).
# Text tokens at the END are NOT rotated (hence "txt-last").
# ---------------------------------------------------------------------------

@triton.jit
def _rope_hyvideo_inplace_kernel(
    Q, K,              # [B, S, H, D]
    Cos, Sin,          # [S, D] full-dimension cos/sin cache
    B, S, H,
    D: tl.constexpr,
    D2: tl.constexpr,
    stride_qb, stride_qs, stride_qh,
    stride_cs,
    BLOCK_S: tl.constexpr,
):
    # grid: (B * H, ceil(S / BLOCK_S))
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    d2_offs = tl.arange(0, D2)
    cs_mask = s_mask[:, None]

    cos_even = tl.load(Cos + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2, mask=cs_mask, other=0.0).to(tl.float32)
    sin_even = tl.load(Sin + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2, mask=cs_mask, other=0.0).to(tl.float32)
    cos_odd = tl.load(Cos + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2 + 1, mask=cs_mask, other=0.0).to(tl.float32)
    sin_odd = tl.load(Sin + s_offs[:, None] * stride_cs + d2_offs[None, :] * 2 + 1, mask=cs_mask, other=0.0).to(tl.float32)

    for buf in range(2):
        base = Q if buf == 0 else K
        base_ptr = base + b * stride_qb + s_offs[:, None] * stride_qs + h * stride_qh

        x1_ptr = base_ptr + d2_offs[None, :] * 2
        x2_ptr = base_ptr + d2_offs[None, :] * 2 + 1
        x1 = tl.load(x1_ptr, mask=cs_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr, mask=cs_mask, other=0.0).to(tl.float32)

        y1 = x1 * cos_even - x2 * sin_even
        y2 = x2 * cos_odd + x1 * sin_odd

        tl.store(x1_ptr, y1.to(base.type.element_ty), mask=cs_mask)
        tl.store(x2_ptr, y2.to(base.type.element_ty), mask=cs_mask)


def _triton_rope_hyvideo_full_inplace(q, k, cos, sin):
    q = q.contiguous()
    k = k.contiguous()
    B, S, H, D = q.shape
    D2 = D // 2
    cos_2d = _to_device(cos.reshape(S, -1).contiguous(), q.device, dtype=torch.float32)
    sin_2d = _to_device(sin.reshape(S, -1).contiguous(), q.device, dtype=torch.float32)
    if cos_2d.shape[1] != D or sin_2d.shape[1] != D:
        raise RuntimeError(
            "SparseVideo HunyuanVideo Triton RoPE expects full-dimension cos/sin "
            f"with width {D}; got {cos_2d.shape[1]} and {sin_2d.shape[1]}"
        )

    BLOCK_S = min(64, triton.next_power_of_2(S))
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    _rope_hyvideo_inplace_kernel[grid](
        q, k, cos_2d, sin_2d,
        B, S, H, D, D2,
        q.stride(0), q.stride(1), q.stride(2),
        cos_2d.stride(0),
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return q, k


def triton_rope_hyvideo_inplace(q, k, cos, sin, txt_len=0):
    backend = _kernel_backend()
    if backend == "pytorch":
        return _rope_hyvideo_pytorch(q, k, cos, sin, txt_len)
    if not q.is_cuda:
        if _requires_cuda_backend(backend):
            _raise_cuda_required(backend, "HunyuanVideo RoPE")
        return _rope_hyvideo_pytorch(q, k, cos, sin, txt_len)
    S = q.shape[1]
    S_vid = S - txt_len

    # The native SVOO-style Hunyuan RoPE op expects [B,H,S,D], while Diffusers
    # and SparseVideo processors keep Hunyuan Q/K in [B,S,H,D]. In auto mode the
    # layout conversion costs two full Q/K copies and can OOM 720p/129f Hunyuan
    # on an 80GB A100. Use the BSHD Triton kernel unless native was explicitly
    # requested for debugging/parity checks.
    if _native_required():
        native = _load_native_kernels(required=_native_required())
        if native is not None:
            D = q.shape[-1]
            cos_2d = _to_device(cos.reshape(S_vid, -1).contiguous(), q.device, dtype=torch.float32)
            sin_2d = _to_device(sin.reshape(S_vid, -1).contiguous(), q.device, dtype=torch.float32)
            if cos_2d.shape[1] != D or sin_2d.shape[1] != D:
                if _native_required():
                    raise RuntimeError(
                        "SparseVideo `_kernels` Hunyuan RoPE expects cos/sin width head_dim; "
                        f"got {cos_2d.shape[1]} and {sin_2d.shape[1]} for head_dim {D}"
                    )
            else:
                q_bhsd = q.permute(0, 2, 1, 3).contiguous()
                k_bhsd = k.permute(0, 2, 1, 3).contiguous()
                native.apply_qk_rope_inplace_cossin_txtlast(q_bhsd, k_bhsd, cos_2d, sin_2d, int(txt_len))
                return (
                    q_bhsd.permute(0, 2, 1, 3).contiguous(),
                    k_bhsd.permute(0, 2, 1, 3).contiguous(),
                )

    if txt_len > 0:
        q_vid, k_vid = _triton_rope_hyvideo_full_inplace(
            q[:, :S_vid].contiguous(), k[:, :S_vid].contiguous(), cos, sin,
        )
        return torch.cat([q_vid, q[:, S_vid:]], dim=1), torch.cat([k_vid, k[:, S_vid:]], dim=1)
    return _triton_rope_hyvideo_full_inplace(q, k, cos, sin)


def _rope_hyvideo_pytorch(q, k, cos, sin, txt_len):
    S = q.shape[1]
    S_vid = S - txt_len
    cos_2d = _to_device(cos.reshape(S_vid, -1), q.device, dtype=torch.float32)
    sin_2d = _to_device(sin.reshape(S_vid, -1), q.device, dtype=torch.float32)

    def _apply(hidden):
        x_real, x_imag = hidden.reshape(*hidden.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        return (
            hidden.float() * cos_2d[None, :, None, :]
            + x_rotated.float() * sin_2d[None, :, None, :]
        ).to(hidden.dtype)

    if txt_len > 0:
        qr = _apply(q[:, :S_vid])
        kr = _apply(k[:, :S_vid])
        q = torch.cat([qr, q[:, S_vid:]], dim=1)
        k = torch.cat([kr, k[:, S_vid:]], dim=1)
    else:
        q = _apply(q)
        k = _apply(k)
    return q, k
