"""SVOO co-clustering Triton kernels (two-pass, low-memory).

Port of: training_free/SVOO/svoo/co_clustering.py
Pass 1: _profile_norm_kernel — compute ||x @ kc^T||_2 per token without materializing [B, N, K]
Pass 2: _fused_cocluster_assign_kernel — argmin in profile space without materializing [B, N, J]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _profile_norm_kernel(
    X,       # [B, N, D]
    KC,      # [B, K, D]
    Norms,   # [B, N] float32 output
    B, N,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_kcb, stride_kck, stride_kcd,
    stride_nb, stride_nn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B:
        return

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, D)

    x_ptrs = X + pid_b * stride_xb + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    sq_norm = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        kc_ptrs = KC + pid_b * stride_kcb + d_offs[:, None] * stride_kcd + k_offs[None, :] * stride_kck
        kc_tile = tl.load(kc_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        dot = tl.dot(x_tile, kc_tile, allow_tf32=True)
        dot = tl.where(k_mask[None, :], dot, 0.0)
        sq_norm += tl.sum(dot * dot, axis=1)

    norms = tl.sqrt(sq_norm + 1e-8)
    norm_ptrs = Norms + pid_b * stride_nb + n_offs * stride_nn
    tl.store(norm_ptrs, norms, mask=n_mask)


@triton.jit
def _fused_cocluster_assign_kernel(
    X,      # [B, N, D]
    KC,     # [B, K, D]
    PC,     # [B, J, K] float32 normalized profile centroids
    Norms,  # [B, N] float32
    Out,    # [B, N] int32
    B, N,
    K: tl.constexpr,
    J: tl.constexpr,
    D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_kcb, stride_kck, stride_kcd,
    stride_pcb, stride_pcj, stride_pck,
    stride_nb, stride_nn,
    stride_ob, stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_b >= B:
        return

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, D)

    x_ptrs = X + pid_b * stride_xb + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    norm_ptrs = Norms + pid_b * stride_nb + n_offs * stride_nn
    norms = tl.load(norm_ptrs, mask=n_mask, other=1.0)

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for j_start in range(0, J, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < J

        dot_nj = tl.zeros((BLOCK_N, BLOCK_J), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            kc_ptrs = KC + pid_b * stride_kcb + d_offs[:, None] * stride_kcd + k_offs[None, :] * stride_kck
            kc_tile = tl.load(kc_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

            dot_xkc = tl.dot(x_tile, kc_tile, allow_tf32=True)
            dot_xkc = dot_xkc / norms[:, None]
            dot_xkc = tl.where(k_mask[None, :], dot_xkc, 0.0)

            pc_ptrs = PC + pid_b * stride_pcb + k_offs[:, None] * stride_pck + j_offs[None, :] * stride_pcj
            pc_tile = tl.load(pc_ptrs, mask=(k_mask[:, None] & j_mask[None, :]), other=0.0)

            dot_nj += tl.dot(dot_xkc, pc_tile, allow_tf32=True)

        dist = 2.0 - 2.0 * dot_nj
        dist = tl.where(j_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, (j_start + curr_idx).to(tl.int32), best_idx)

    out_ptrs = Out + pid_b * stride_ob + n_offs * stride_on
    tl.store(out_ptrs, best_idx, mask=n_mask)


def profile_norm(x: torch.Tensor, kcentroids: torch.Tensor) -> torch.Tensor:
    """Compute profile row norms without materializing [B, N, K].

    norm[b, n] = ||x[b,n,:] @ kcentroids[b,:,:].T||_2

    Args:
        x: [B, N, D] tokens
        kcentroids: [B, K, D] profile-forming centroids
    Returns:
        norms: [B, N] float32
    """
    B, N, D = x.shape
    K = kcentroids.shape[1]

    x = x.contiguous()
    kcentroids = kcentroids.contiguous()
    norms = torch.empty(B, N, device=x.device, dtype=torch.float32)

    BLOCK_N = 32
    BLOCK_K = min(32, K)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _profile_norm_kernel[grid](
        x, kcentroids, norms,
        B, N, K=K, D=D,
        stride_xb=x.stride(0), stride_xn=x.stride(1), stride_xd=x.stride(2),
        stride_kcb=kcentroids.stride(0), stride_kck=kcentroids.stride(1), stride_kcd=kcentroids.stride(2),
        stride_nb=norms.stride(0), stride_nn=norms.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return norms


def co_cluster_assign(
    x: torch.Tensor,
    kcentroids: torch.Tensor,
    profile_centroids: torch.Tensor,
    norms: torch.Tensor,
) -> torch.Tensor:
    """Fused profile-space nearest-centroid without materializing [B, N, K] or [B, N, J].

    labels[b,n] = argmin_j  2 - 2*dot(norm(x[b,n]@kc[b].T), pc_norm[b,j])

    Args:
        x: [B, N, D] tokens
        kcentroids: [B, K, D] profile-forming centroids
        profile_centroids: [B, J, K] float32 L2-normalized profile centroids
        norms: [B, N] float32 from profile_norm()
    Returns:
        labels: [B, N] int64
    """
    B, N, D = x.shape
    K = kcentroids.shape[1]
    J = profile_centroids.shape[1]

    x = x.contiguous()
    kcentroids = kcentroids.contiguous()
    profile_centroids = profile_centroids.float().contiguous()
    norms = norms.contiguous()

    out = torch.empty(B, N, device=x.device, dtype=torch.int32)

    BLOCK_N = 64
    BLOCK_K = min(32, K)
    BLOCK_J = min(32, J)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _fused_cocluster_assign_kernel[grid](
        x, kcentroids, profile_centroids, norms, out,
        B, N, K=K, J=J, D=D,
        stride_xb=x.stride(0), stride_xn=x.stride(1), stride_xd=x.stride(2),
        stride_kcb=kcentroids.stride(0), stride_kck=kcentroids.stride(1), stride_kcd=kcentroids.stride(2),
        stride_pcb=profile_centroids.stride(0), stride_pcj=profile_centroids.stride(1), stride_pck=profile_centroids.stride(2),
        stride_nb=norms.stride(0), stride_nn=norms.stride(1),
        stride_ob=out.stride(0), stride_on=out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_J=BLOCK_J,
    )
    return out.long()
