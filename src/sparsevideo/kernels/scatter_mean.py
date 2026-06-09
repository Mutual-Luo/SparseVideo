from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_mean_contiguous_kernel(
    src_ptr, size_ptr, offsets_ptr, out_ptr,
    C, H, L, D, K,
    stride_src_c, stride_src_h, stride_src_l, stride_src_d,
    stride_size_c, stride_size_h, stride_size_k,
    stride_off_c, stride_off_h, stride_off_k,
    stride_out_c, stride_out_h, stride_out_k, stride_out_d,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    k_idx = pid % K
    bh_idx = pid // K
    h_idx = bh_idx % H
    c_idx = bh_idx // H

    count_ptr = size_ptr + c_idx * stride_size_c + h_idx * stride_size_h + k_idx * stride_size_k
    count = tl.load(count_ptr)

    if count <= 0:
        return

    off_ptr = offsets_ptr + c_idx * stride_off_c + h_idx * stride_off_h + k_idx * stride_off_k
    start_idx = tl.load(off_ptr)

    cols = tl.arange(0, BLOCK_D)
    d_mask = cols < D

    src_ptr_base = (
        src_ptr
        + c_idx * stride_src_c
        + h_idx * stride_src_h
        + start_idx * stride_src_l
        + cols * stride_src_d
    )

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(count):
        val = tl.load(src_ptr_base + i * stride_src_l, mask=d_mask, other=0.0)
        acc += val

    mean_val = acc / count

    out_ptr_final = (
        out_ptr
        + c_idx * stride_out_c
        + h_idx * stride_out_h
        + k_idx * stride_out_k
        + cols * stride_out_d
    )

    tl.store(out_ptr_final, mean_val.to(out_ptr.dtype.element_ty), mask=d_mask)


def scatter_mean_fused(src, size):
    """Per-cluster mean over a sequence already sorted by cluster id.

    Ported from Sparse-VideoGen (SVG-EAR) ``scatter_mean_fused``. Used to build
    value-cluster centroids for the error-aware (EAR) block selection.

    Args:
        src: (cfg, num_heads, seq_len, dim) contiguous data, sorted by cluster id.
        size: (cfg, num_heads, num_clusters) element count per cluster.

    Returns:
        (cfg, num_heads, num_clusters, dim) cluster means.
    """
    C, H, L, D = src.shape
    K = size.shape[-1]

    offsets = torch.zeros_like(size)
    offsets[..., 1:] = torch.cumsum(size[..., :-1], dim=-1)

    out = torch.zeros((C, H, K, D), device=src.device, dtype=src.dtype)
    BLOCK_D = triton.next_power_of_2(D)

    grid = (C * H * K,)
    _scatter_mean_contiguous_kernel[grid](
        src, size, offsets, out,
        C, H, L, D, K,
        src.stride(0), src.stride(1), src.stride(2), src.stride(3),
        size.stride(0), size.stride(1), size.stride(2),
        offsets.stride(0), offsets.stride(1), offsets.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=BLOCK_D,
    )
    return out
