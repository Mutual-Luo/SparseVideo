"""Fused inplace Triton kernels for RMSNorm and RoPE.

These replace the allocating PyTorch paths in the WAN and HunyuanVideo processors,
matching the optimization strategy of the original SVG/SVOO _kernels C++ implementation.

Kernel variants:
  triton_rmsnorm_inplace      — inplace RMSNorm over last dim; used by HunyuanVideo
  triton_rope_wan_inplace     — inplace RoPE in WAN's complex-pair format [B,S,H,D]
  triton_rope_hyvideo_inplace — inplace RoPE for HunyuanVideo (txt-last, skips text tokens)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    if not x.is_cuda:
        return _rmsnorm_pytorch(x, weight, eps)

    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D).contiguous()
    M = x_2d.shape[0]

    N2 = triton.next_power_of_2(D)
    BLOCK_M = 32 if D <= 512 else 1
    grid = (triton.cdiv(M, BLOCK_M),)

    _rmsnorm_inplace_kernel[grid](
        x_2d, weight, x_2d.stride(0), M, D, N2, eps,
        BLOCK_M=BLOCK_M, num_warps=8,
    )
    return x_2d.reshape(orig_shape)


def _rmsnorm_pytorch(x, weight, eps):
    orig_dtype = x.dtype
    x_f = x.float()
    var = (x_f * x_f).mean(-1, keepdim=True)
    x_norm = x_f / (var + eps).sqrt()
    return (x_norm * weight.float()).to(orig_dtype)


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
    if not q.is_cuda:
        return _rope_wan_pytorch(q, k, freqs_cos, freqs_sin)

    q = q.contiguous()
    k = k.contiguous()
    B, S, H, D = q.shape
    D2 = D // 2

    # Normalise cos/sin to [S, D] regardless of input shape ([1,S,1,D] or [S,D])
    cos = freqs_cos.reshape(S, -1).contiguous().float()
    sin = freqs_sin.reshape(S, -1).contiguous().float()

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
    cos = freqs_cos.reshape(S, -1)[..., 0::2].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]
    sin = freqs_sin.reshape(S, -1)[..., 1::2].unsqueeze(0).unsqueeze(2)
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



def triton_rope_hyvideo_inplace(q, k, cos, sin, txt_len=0):
    if not q.is_cuda:
        return _rope_hyvideo_pytorch(q, k, cos, sin, txt_len)
    S = q.shape[1]
    S_vid = S - txt_len
    if txt_len > 0:
        q_vid, k_vid = triton_rope_wan_inplace(
            q[:, :S_vid].contiguous(), k[:, :S_vid].contiguous(), cos, sin,
        )
        q = torch.cat([q_vid, q[:, S_vid:]], dim=1)
        k = torch.cat([k_vid, k[:, S_vid:]], dim=1)
    else:
        q, k = triton_rope_wan_inplace(q, k, cos, sin)
    return q, k


def _rope_hyvideo_pytorch(q, k, cos, sin, txt_len):
    # Reuse WAN's pytorch path — same formula, just limit to video tokens
    S = q.shape[1]
    S_vid = S - txt_len
    # Reshape cos/sin to [S_vid, D] for _rope_wan_pytorch
    cos_2d = cos.reshape(S_vid, -1)
    sin_2d = sin.reshape(S_vid, -1)
    if txt_len > 0:
        qr, kr = _rope_wan_pytorch(q[:, :S_vid].clone(), k[:, :S_vid].clone(), cos_2d, sin_2d)
        q = torch.cat([qr, q[:, S_vid:]], dim=1)
        k = torch.cat([kr, k[:, S_vid:]], dim=1)
    else:
        q, k = _rope_wan_pytorch(q, k, cos_2d, sin_2d)
    return q, k
