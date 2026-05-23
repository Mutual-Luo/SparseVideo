from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_head_placement_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_out_ptr,
    key_out_ptr,
    value_out_ptr,
    best_mask_idx_ptr,
    query_stride_b,
    query_stride_h,
    query_stride_s,
    query_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim,
    head_dim_padded: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    offset_token = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim_padded)
    d_mask = offset_d < head_dim
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    if is_temporal:
        frame_id = offset_token // frame_size
        patch_id = offset_token - frame_id * frame_size
        offset_store_token = tl.where(
            offset_token >= seq_len - context_length,
            offset_token,
            patch_id * num_frame + frame_id,
        )
    else:
        offset_store_token = offset_token

    offset_load = (
        cfg * query_stride_b
        + head * query_stride_h
        + offset_token[:, None] * query_stride_s
        + offset_d[None, :] * query_stride_d
    )
    offset_store = (
        cfg * query_stride_b
        + head * query_stride_h
        + offset_store_token[:, None] * query_stride_s
        + offset_d[None, :] * query_stride_d
    )

    full_mask = offset_mask[:, None] & d_mask[None, :]
    query = tl.load(query_ptr + offset_load, mask=full_mask)
    key = tl.load(key_ptr + offset_load, mask=full_mask)
    value = tl.load(value_ptr + offset_load, mask=full_mask)
    tl.store(query_out_ptr + offset_store, query, mask=full_mask)
    tl.store(key_out_ptr + offset_store, key, mask=full_mask)
    tl.store(value_out_ptr + offset_store, value, mask=full_mask)


@triton.jit
def _hidden_states_placement_kernel(
    hidden_states_ptr,
    hidden_states_out_ptr,
    best_mask_idx_ptr,
    hidden_states_stride_b,
    hidden_states_stride_h,
    hidden_states_stride_s,
    hidden_states_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim,
    head_dim_padded: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    offset_token = tl.arange(0, BLOCK_SIZE) + block_id * BLOCK_SIZE
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim_padded)
    d_mask = offset_d < head_dim
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    if is_temporal:
        patch_id = offset_token // num_frame
        frame_id = offset_token - patch_id * num_frame
        offset_store_token = tl.where(
            offset_token >= seq_len - context_length,
            offset_token,
            frame_id * frame_size + patch_id,
        )
    else:
        offset_store_token = offset_token

    offset_load = (
        cfg * hidden_states_stride_b
        + head * hidden_states_stride_h
        + offset_token[:, None] * hidden_states_stride_s
        + offset_d[None, :] * hidden_states_stride_d
    )
    offset_store = (
        cfg * hidden_states_stride_b
        + head * hidden_states_stride_h
        + offset_store_token[:, None] * hidden_states_stride_s
        + offset_d[None, :] * hidden_states_stride_d
    )

    full_mask = offset_mask[:, None] & d_mask[None, :]
    hidden_states = tl.load(hidden_states_ptr + offset_load, mask=full_mask)
    tl.store(hidden_states_out_ptr + offset_store, hidden_states, mask=full_mask)


def sparse_head_placement(query, key, value, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = query.shape
    head_dim_padded = triton.next_power_of_2(head_dim)
    block_size = 128
    grid = (cfg, num_heads, triton.cdiv(seq_len, block_size))
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    value_out = torch.empty_like(value)

    _sparse_head_placement_kernel[grid](
        query,
        key,
        value,
        query_out,
        key_out,
        value_out,
        best_mask_idx,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        best_mask_idx.stride(0),
        best_mask_idx.stride(1),
        seq_len,
        head_dim,
        head_dim_padded,
        context_length,
        num_frame,
        frame_size,
        block_size,
    )
    return query_out, key_out, value_out


def hidden_states_placement(hidden_states, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = hidden_states.shape
    head_dim_padded = triton.next_power_of_2(head_dim)
    block_size = 128
    grid = (cfg, num_heads, triton.cdiv(seq_len, block_size))
    hidden_states_out = torch.empty_like(hidden_states)

    _hidden_states_placement_kernel[grid](
        hidden_states,
        hidden_states_out,
        best_mask_idx,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        hidden_states.stride(3),
        best_mask_idx.stride(0),
        best_mask_idx.stride(1),
        seq_len,
        head_dim,
        head_dim_padded,
        context_length,
        num_frame,
        frame_size,
        block_size,
    )
    return hidden_states_out
