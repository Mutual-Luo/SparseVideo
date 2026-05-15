"""Triton Sliding Tile Attention kernel with 3D neighborhood overlap.

Port of: training_free/FastVideo/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py
Generalized to work with any video shape (T, H, W) and configurable tile/kernel sizes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp(value, min_val, max_val):
    ret = tl.where(value > max_val, max_val, value)
    ret = tl.where(ret < min_val, min_val, ret)
    return ret


@triton.jit
def _sta_fwd_inner(
    q, k, v, kv_mask, m, l, acc, sm_scale,
    MASK_KV: tl.constexpr,
):
    """Online softmax accumulation for one KV block."""
    scores = tl.dot(q, tl.trans(k)) * sm_scale
    if MASK_KV:
        scores = tl.where(kv_mask[None, :], scores, -float('inf'))

    current_m = tl.max(scores, axis=1)
    new_m = tl.maximum(m, current_m)
    exp_scores = tl.math.exp2(scores - new_m[:, None])
    if MASK_KV:
        exp_scores = tl.where(kv_mask[None, :], exp_scores, 0.0)
    current_l = tl.sum(exp_scores, axis=1)

    alpha = tl.math.exp2(m - new_m)
    l = l * alpha + current_l
    m = new_m
    acc = acc * alpha[:, None] + tl.dot(exp_scores.to(v.type.element_ty), v)
    return m, l, acc


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': bq, 'BLOCK_KV': bkv}, num_stages=s, num_warps=w)
        for bq in [64, 128]
        for bkv in [64, 128]
        for s in [1, 2]
        for w in [4, 8]
    ],
    key=['head_dim'],
)
@triton.jit
def _triton_sta_kernel(
    Q, K, V, Output,
    seq_len: int, head_dim: int,
    img_seq_len: int,
    text_length: int,
    canvas_t: int, canvas_h: int, canvas_w: int,
    kernel_t: int, kernel_h: int, kernel_w: int,
    tile_t: int, tile_h: int, tile_w: int,
    scale: float,
    stride_qb: int, stride_qs: int, stride_qd: int,
    stride_ob: int, stride_os: int, stride_od: int,
    has_text: tl.constexpr,
    text_q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    total_tile_size = tile_t * tile_h * tile_w
    q_block_per_tile = (total_tile_size + BLOCK_Q - 1) // BLOCK_Q

    bh_idx = tl.program_id(0)  # batch * head index

    if text_q:
        q_block_idx = tl.program_id(1)
    else:
        q_tile_flat = tl.program_id(1) // q_block_per_tile
        q_block_idx = tl.program_id(1) % q_block_per_tile

    # Initialize accumulators
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DIM), dtype=tl.float32)

    q_offset = bh_idx * stride_qb
    if text_q:
        q_base_idx = img_seq_len + q_block_idx * BLOCK_Q
        q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < text_length
    else:
        q_base_idx = q_tile_flat * total_tile_size + q_block_idx * BLOCK_Q
        q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < total_tile_size

    q_idx = q_base_idx + tl.arange(0, BLOCK_Q)
    q = tl.load(
        Q + q_offset + q_idx[:, None] * stride_qs + tl.arange(0, BLOCK_DIM)[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0,
    )

    # log2(e) scaling for exp2-based softmax
    sm_scale = scale * 1.4426950408889634

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    tiles_per_hw = num_tiles_h * num_tiles_w

    if text_q:
        # Text queries attend to ALL image + text KV
        kv_tile_start_t = 0
        kv_tile_end_t = num_tiles_t
        kv_tile_start_h = 0
        kv_tile_end_h = num_tiles_h
        kv_tile_start_w = 0
        kv_tile_end_w = num_tiles_w
    else:
        # Image queries attend to neighborhood of tiles
        q_tile_t = q_tile_flat // tiles_per_hw
        remaining = q_tile_flat % tiles_per_hw
        q_tile_h = remaining // num_tiles_w
        q_tile_w = remaining % num_tiles_w

        kernel_center_t = _clamp(q_tile_t, kernel_t // 2, (num_tiles_t - 1) - kernel_t // 2)
        kernel_center_h = _clamp(q_tile_h, kernel_h // 2, (num_tiles_h - 1) - kernel_h // 2)
        kernel_center_w = _clamp(q_tile_w, kernel_w // 2, (num_tiles_w - 1) - kernel_w // 2)

        kv_tile_start_t = kernel_center_t - kernel_t // 2
        kv_tile_end_t = kernel_center_t + kernel_t // 2 + 1
        kv_tile_end_t = tl.where(kv_tile_end_t > num_tiles_t, num_tiles_t, kv_tile_end_t)

        kv_tile_start_h = kernel_center_h - kernel_h // 2
        kv_tile_end_h = kernel_center_h + kernel_h // 2 + 1
        kv_tile_end_h = tl.where(kv_tile_end_h > num_tiles_h, num_tiles_h, kv_tile_end_h)

        kv_tile_start_w = kernel_center_w - kernel_w // 2
        kv_tile_end_w = kernel_center_w + kernel_w // 2 + 1
        kv_tile_end_w = tl.where(kv_tile_end_w > num_tiles_w, num_tiles_w, kv_tile_end_w)

    # Iterate over KV tiles in neighborhood
    kv_offset = bh_idx * stride_qb  # same layout as Q
    for kv_t in tl.range(kv_tile_start_t, kv_tile_end_t):
        for kv_h in tl.range(kv_tile_start_h, kv_tile_end_h):
            for kv_w in tl.range(kv_tile_start_w, kv_tile_end_w):
                kv_base_idx = (kv_t * tiles_per_hw + kv_h * num_tiles_w + kv_w) * total_tile_size

                for kv_block_start in tl.range(0, total_tile_size, BLOCK_KV):
                    kv_rows = tl.arange(0, BLOCK_KV) + kv_block_start
                    kv_mask = kv_rows < total_tile_size
                    kv_global = kv_base_idx + kv_rows

                    k = tl.load(
                        K + kv_offset + kv_global[:, None] * stride_qs + tl.arange(0, BLOCK_DIM)[None, :] * stride_qd,
                        mask=kv_mask[:, None], other=0.0,
                    )
                    v = tl.load(
                        V + kv_offset + kv_global[:, None] * stride_qs + tl.arange(0, BLOCK_DIM)[None, :] * stride_qd,
                        mask=kv_mask[:, None], other=0.0,
                    )
                    m, l, acc = _sta_fwd_inner(q, k, v, kv_mask, m, l, acc, sm_scale, MASK_KV=True)

    # Text KV (all queries attend to text)
    if has_text:
        text_start = img_seq_len
        for text_block_start in tl.range(0, text_length, BLOCK_KV):
            kv_rows = tl.arange(0, BLOCK_KV) + text_block_start
            kv_mask = kv_rows < text_length
            kv_global = text_start + kv_rows

            k = tl.load(
                K + kv_offset + kv_global[:, None] * stride_qs + tl.arange(0, BLOCK_DIM)[None, :] * stride_qd,
                mask=kv_mask[:, None], other=0.0,
            )
            v = tl.load(
                V + kv_offset + kv_global[:, None] * stride_qs + tl.arange(0, BLOCK_DIM)[None, :] * stride_qd,
                mask=kv_mask[:, None], other=0.0,
            )
            m, l, acc = _sta_fwd_inner(q, k, v, kv_mask, m, l, acc, sm_scale, MASK_KV=True)

    # Normalize
    l_safe = tl.where(l == 0.0, 1.0, l)
    output_acc = acc / l_safe[:, None]

    tl.store(
        Output + bh_idx * stride_ob + q_idx[:, None] * stride_os + tl.arange(0, BLOCK_DIM)[None, :] * stride_od,
        output_acc.to(Output.dtype.element_ty),
        mask=q_mask[:, None],
    )


def triton_sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    canvas_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    kernel_size: tuple[int, int, int],
    text_length: int = 0,
) -> torch.Tensor:
    """Sliding Tile Attention via Triton with 3D neighborhood overlap.

    Args:
        q, k, v: [B*H, seq_len, D] (batch and heads already folded)
        canvas_shape: (T, H, W) video token grid shape
        tile_size: (tile_t, tile_h, tile_w) tile dimensions
        kernel_size: (kt, kh, kw) neighborhood kernel in tile units
        text_length: number of text/context tokens appended after video

    Returns:
        output: [B*H, seq_len, D]
    """
    BH, seq_len, D = q.shape
    canvas_t, canvas_h, canvas_w = canvas_shape
    tile_t, tile_h, tile_w = tile_size
    kernel_t, kernel_h, kernel_w = kernel_size

    img_seq_len = canvas_t * canvas_h * canvas_w

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    num_tiles = num_tiles_t * num_tiles_h * num_tiles_w
    total_tile_size = tile_t * tile_h * tile_w

    output = torch.empty_like(q)

    # Image queries
    grid_img = lambda META: (BH, num_tiles * ((total_tile_size + META['BLOCK_Q'] - 1) // META['BLOCK_Q']))
    _triton_sta_kernel[grid_img](
        q, k, v, output,
        seq_len, D,
        img_seq_len, text_length,
        canvas_t, canvas_h, canvas_w,
        kernel_t, kernel_h, kernel_w,
        tile_t, tile_h, tile_w,
        scale=D ** -0.5,
        stride_qb=q.stride(0), stride_qs=q.stride(1), stride_qd=q.stride(2),
        stride_ob=output.stride(0), stride_os=output.stride(1), stride_od=output.stride(2),
        has_text=(text_length > 0),
        text_q=False,
        BLOCK_DIM=D,
    )

    # Text queries (attend to everything)
    if text_length > 0:
        grid_text = lambda META: (BH, (text_length + META['BLOCK_Q'] - 1) // META['BLOCK_Q'])
        _triton_sta_kernel[grid_text](
            q, k, v, output,
            seq_len, D,
            img_seq_len, text_length,
            canvas_t, canvas_h, canvas_w,
            kernel_t, kernel_h, kernel_w,
            tile_t, tile_h, tile_w,
            scale=D ** -0.5,
            stride_qb=q.stride(0), stride_qs=q.stride(1), stride_qd=q.stride(2),
            stride_ob=output.stride(0), stride_os=output.stride(1), stride_od=output.stride(2),
            has_text=True,
            text_q=True,
            BLOCK_DIM=D,
        )

    return output
