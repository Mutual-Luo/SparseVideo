import torch

import triton
import triton.language as tl


@triton.jit
def _cluster_sparse_attn_topk(
    query,
    key,
    value,
    output,
    selected_kv_indices,
    q_counts,
    kv_counts,
    sm_scale,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    selected_stride_0,
    selected_stride_1,
    selected_stride_2,
    selected_stride_3,
    q_counts_stride_0,
    q_counts_stride_1,
    q_counts_stride_2,
    kv_counts_stride_0,
    kv_counts_stride_1,
    kv_counts_stride_2,
    HEAD_DIM: tl.constexpr,
    TOPK_NUM: tl.constexpr,
    BLOCK_N: tl.constexpr = 64,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_id = tl.program_id(2)

    qk_scale = sm_scale * 1.44269504

    q_end = tl.load(q_counts + b_id * q_counts_stride_0 + h_id * q_counts_stride_1 + q_id * q_counts_stride_2)
    q_start = tl.load(
        q_counts + b_id * q_counts_stride_0 + h_id * q_counts_stride_1 + (q_id - 1) * q_counts_stride_2,
        mask=q_id > 0,
        other=0,
    )
    q_len = q_end - q_start
    if q_len == 0:
        return

    q_iter_num = (q_len + BLOCK_N - 1) // BLOCK_N
    for i in range(q_iter_num):
        query_pos = q_start + i * BLOCK_N + tl.arange(0, BLOCK_N)
        query_offset = (
            b_id * query_stride_0
            + h_id * query_stride_1
            + query_pos * query_stride_2
        )
        query_offset = query_offset[:, None] + tl.arange(0, HEAD_DIM)
        query_load_mask = query_pos < q_end
        load_query = tl.load(query + query_offset, mask=query_load_mask[:, None])

        m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        for selected_pos in range(TOPK_NUM):
            selected_offset = (
                b_id * selected_stride_0
                + h_id * selected_stride_1
                + q_id * selected_stride_2
                + selected_pos * selected_stride_3
            )
            kv_id = tl.load(selected_kv_indices + selected_offset)
            kv_end = tl.load(
                kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + kv_id * kv_counts_stride_2
            )
            kv_start = tl.load(
                kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + (kv_id - 1) * kv_counts_stride_2,
                mask=kv_id > 0,
                other=0,
            )
            kv_len = kv_end - kv_start
            kv_iter_num = (kv_len + BLOCK_N - 1) // BLOCK_N
            for k in range(kv_iter_num):
                kv_pos = kv_start + k * BLOCK_N + tl.arange(0, BLOCK_N)

                key_offset = (
                    b_id * key_stride_0
                    + h_id * key_stride_1
                    + kv_pos * key_stride_2
                )
                key_offset = key_offset[:, None] + tl.arange(0, HEAD_DIM)
                key_load_mask = kv_pos < kv_end
                load_key = tl.load(key + key_offset, mask=key_load_mask[:, None])

                value_offset = (
                    b_id * value_stride_0
                    + h_id * value_stride_1
                    + kv_pos * value_stride_2
                )
                value_offset = value_offset[:, None] + tl.arange(0, HEAD_DIM)
                load_value = tl.load(value + value_offset, mask=key_load_mask[:, None])

                qk = tl.dot(load_query, load_key.T)
                qk = qk * qk_scale + tl.where(key_load_mask, 0, -1e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                p = p.to(load_value.dtype)
                acc = tl.dot(p, load_value, acc)
                m_i = m_ij

        acc = acc / l_i[:, None]
        output_offset = (
            b_id * output_stride_0
            + h_id * output_stride_1
            + query_pos * output_stride_2
        )
        output_offset = output_offset[:, None] + tl.arange(0, HEAD_DIM)
        tl.store(output + output_offset, acc.to(tl.float16), mask=query_load_mask[:, None])


def triton_cluster_sparse_attn_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_kv_indices: torch.Tensor,
    q_counts: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
):
    batch_size, num_heads, _q_len, head_dim = query.shape
    _batch, _heads, _q_kernel_num, topk_num = selected_kv_indices.shape
    grid = lambda args: (
        batch_size,
        num_heads,
        _q_kernel_num,
    )

    output = torch.zeros_like(query)
    selected_kv_indices = selected_kv_indices.to(torch.int32).contiguous()
    with torch.cuda.device(query.device):
        _cluster_sparse_attn_topk[grid](
            query,
            key,
            value,
            output,
            selected_kv_indices,
            q_counts,
            kv_counts,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            selected_kv_indices.stride(0),
            selected_kv_indices.stride(1),
            selected_kv_indices.stride(2),
            selected_kv_indices.stride(3),
            q_counts.stride(0),
            q_counts.stride(1),
            q_counts.stride(2),
            kv_counts.stride(0),
            kv_counts.stride(1),
            kv_counts.stride(2),
            head_dim,
            topk_num,
        )
    return output
