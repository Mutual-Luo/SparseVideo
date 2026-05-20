"""Triton block-sparse attention forward kernel with online softmax.

Port of: training_free/SVOO/svoo/co_clustering.py (_dynamic_block_sparse_fwd_kernel)
Adapted for [B, N, D] input where B = batch*heads (already folded by method code).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _block_d(head_dim: int) -> int:
    block = max(16, _next_pow2(head_dim))
    if block > 256:
        raise AssertionError(f"head_dim {head_dim} not supported, must be <= 256")
    return block


@triton.jit
def _block_sparse_fwd_kernel(
    Q, K, V, Out,
    DynMap,
    QCumSize, KCumSize,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_db, stride_dq, stride_dk,
    stride_qcb, stride_qcq,
    stride_kcb, stride_kck,
    S, D: tl.constexpr, BLOCK_D: tl.constexpr, scale,
    QC_NUM: tl.constexpr,
    KC_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    q_block_idx = pid % QC_NUM
    b = pid // QC_NUM

    qcs_base = QCumSize + b * stride_qcb
    q_start = tl.load(qcs_base + q_block_idx * stride_qcq)
    q_end = tl.load(qcs_base + (q_block_idx + 1) * stride_qcq)
    q_block_size = q_end - q_start

    if q_block_size == 0:
        return

    q_base = Q + b * stride_qb + q_start * stride_qs
    k_base = K + b * stride_kb
    v_base = V + b * stride_vb
    o_base = Out + b * stride_ob + q_start * stride_os
    dmap_base = DynMap + b * stride_db + q_block_idx * stride_dq
    kcs_base = KCumSize + b * stride_kcb

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    dim_mask = offs_d < D

    for q_chunk_start in range(0, q_block_size, BLOCK_M):
        q_rows = offs_m + q_chunk_start
        q_mask = q_rows < q_block_size

        q_ptrs = q_base + q_rows[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_chunk = tl.load(q_ptrs, mask=q_mask[:, None] & dim_mask[None, :], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for ki in range(KC_NUM):
            is_active = tl.load(dmap_base + ki * stride_dk)
            if is_active:
                k_start = tl.load(kcs_base + ki * stride_kck)
                k_end = tl.load(kcs_base + (ki + 1) * stride_kck)
                k_block_size = k_end - k_start

                if k_block_size > 0:
                    k_block_base = k_base + k_start * stride_ks
                    v_block_base = v_base + k_start * stride_vs
                    offs_n = tl.arange(0, BLOCK_N)

                    for k_chunk_start in range(0, k_block_size, BLOCK_N):
                        k_rows = offs_n + k_chunk_start
                        k_mask = k_rows < k_block_size

                        k_ptrs = k_block_base + k_rows[:, None] * stride_ks + offs_d[None, :] * stride_kd
                        v_ptrs = v_block_base + k_rows[:, None] * stride_vs + offs_d[None, :] * stride_vd
                        chunk_mask = k_mask[:, None] & dim_mask[None, :]
                        k_chunk = tl.load(k_ptrs, mask=chunk_mask, other=0.0)
                        v_chunk = tl.load(v_ptrs, mask=chunk_mask, other=0.0)

                        s_ij = tl.dot(q_chunk, k_chunk.T) * scale
                        s_ij = tl.where(k_mask[None, :], s_ij, float("-inf"))
                        s_ij = tl.where(q_mask[:, None], s_ij, float("-inf"))

                        m_ij = tl.max(s_ij, axis=1)
                        m_new = tl.maximum(m_i, m_ij)
                        p_ij = tl.exp(s_ij - m_new[:, None])
                        p_ij = tl.where(k_mask[None, :], p_ij, 0.0)

                        alpha = tl.exp(m_i - m_new)
                        l_ij = tl.sum(p_ij, axis=1)
                        l_i = l_i * alpha + l_ij

                        p_cast = p_ij.to(V.dtype.element_ty)
                        acc = acc * alpha[:, None] + tl.dot(p_cast, v_chunk)
                        m_i = m_new

        l_safe = tl.where(l_i == 0, 1.0, l_i)
        out_chunk = acc / l_safe[:, None]
        out_chunk = tl.where(l_i[:, None] == 0, 0.0, out_chunk)

        o_ptrs = o_base + q_rows[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out_chunk.to(Out.dtype.element_ty), mask=q_mask[:, None] & dim_mask[None, :])


def block_sparse_attention(
    q_sorted: torch.Tensor,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    dynamic_map: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dynamic block-sparse attention via Triton.

    Tokens must be pre-sorted by cluster. The kernel iterates only over
    active (Q-cluster, K-cluster) pairs specified by dynamic_map.

    Args:
        q_sorted: [B, N, D] queries sorted by cluster
        k_sorted: [B, N, D] keys sorted by cluster
        v_sorted: [B, N, D] values sorted by cluster
        q_sizes: [B, nqc] tokens per Q-cluster
        k_sizes: [B, nkc] tokens per K-cluster
        dynamic_map: [B, nqc, nkc] bool — which pairs are active
        scale: attention scale factor (typically D^{-0.5})

    Returns:
        out: [B, N, D] attention output (sorted order)
    """
    B, N, D = q_sorted.shape
    nqc = q_sizes.shape[1]
    nkc = k_sizes.shape[1]

    BLOCK_D = _block_d(D)

    qc_cum = torch.cumsum(
        torch.cat([torch.zeros(B, 1, dtype=q_sizes.dtype, device=q_sizes.device), q_sizes], dim=1),
        dim=1,
    ).int()
    kc_cum = torch.cumsum(
        torch.cat([torch.zeros(B, 1, dtype=k_sizes.dtype, device=k_sizes.device), k_sizes], dim=1),
        dim=1,
    ).int()

    out = torch.empty_like(q_sorted)

    def _pow2_floor(n):
        return 1 << (int(n).bit_length() - 1) if n > 0 else 1

    BLOCK_M = _pow2_floor(min(128 if N > 1024 else 64, N))
    BLOCK_N = _pow2_floor(min(64, N))
    BLOCK_M = max(BLOCK_M, 16)
    BLOCK_N = max(BLOCK_N, 16)

    grid = (B * nqc,)

    _block_sparse_fwd_kernel[grid](
        q_sorted, k_sorted, v_sorted, out,
        dynamic_map, qc_cum, kc_cum,
        q_sorted.stride(0), q_sorted.stride(1), q_sorted.stride(2),
        k_sorted.stride(0), k_sorted.stride(1), k_sorted.stride(2),
        v_sorted.stride(0), v_sorted.stride(1), v_sorted.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        dynamic_map.stride(0), dynamic_map.stride(1), dynamic_map.stride(2),
        qc_cum.stride(0), qc_cum.stride(1),
        kc_cum.stride(0), kc_cum.stride(1),
        N, D, BLOCK_D, scale,
        QC_NUM=nqc, KC_NUM=nkc,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return out
