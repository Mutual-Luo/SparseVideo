"""SVOO co-clustering Triton kernels (two-pass, low-memory).

Port of: training_free/SVOO/svoo/co_clustering.py
Pass 1: _profile_norm_kernel — compute ||x @ kc^T||_2 per token without materializing [B, N, K]
Pass 2: _fused_cocluster_assign_kernel — argmin in profile space without materializing [B, N, J]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _centroid_update_chunk_kernel(
    X,                 # [B, N, D] original-order tokens
    SortedIdx,         # [B, N] token indices sorted by cluster id
    SortedCluster,     # [B, N] cluster ids in sorted order
    Sum,               # [B, K, D] fp32
    Count,             # [B, K] int32
    B,
    N,
    D: tl.constexpr,
    K: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_b = tl.program_id(1)
    chunk_start = pid_chunk * BLOCK_N
    if chunk_start >= N:
        return

    token_offs = chunk_start + tl.arange(0, BLOCK_N)
    valid = token_offs < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    first_id = tl.load(SortedCluster + pid_b * N + first_token_idx)
    last_id = tl.load(SortedCluster + pid_b * N + last_token_idx)
    all_ids = tl.load(SortedCluster + pid_b * N + token_offs, mask=valid, other=-1)
    all_token_idx = tl.load(SortedIdx + pid_b * N + token_offs, mask=valid, other=-1)
    dim_offs = tl.arange(0, D)

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            x_ptrs = (
                X
                + pid_b.to(tl.int64) * stride_xb
                + all_token_idx.to(tl.int64)[:, None] * stride_xn
                + dim_offs.to(tl.int64)[None, :] * stride_xd
            )
            token_valid = all_token_idx[:, None] >= 0
            feats = tl.load(x_ptrs, mask=(cluster_mask[:, None] & token_valid), other=0.0).to(tl.float32)
            sums = tl.sum(feats, axis=0)
            sum_ptrs = Sum + (pid_b * K + cid) * D + dim_offs
            tl.atomic_add(sum_ptrs, sums)
            tl.atomic_add(Count + pid_b * K + cid, cluster_size)


def centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    block_n: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update Euclidean centroids using upstream SVOO's sorted-cluster policy.

    Empty clusters keep their previous centroid, matching
    training_free/SVOO/svoo/co_clustering.py.
    """
    assert x.is_cuda and cluster_ids.is_cuda, "centroid update requires CUDA"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, block_n), B)
    _centroid_update_chunk_kernel[grid](
        x,
        sorted_idx.to(torch.int32),
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        B,
        N,
        D,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_N=block_n,
    )

    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    return centroids.to(x.dtype), centroid_counts.to(torch.long)


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
    BLOCK_K = max(16, min(32, triton.next_power_of_2(K)))

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
    BLOCK_K = max(16, min(32, triton.next_power_of_2(K)))
    BLOCK_J = max(16, min(32, triton.next_power_of_2(J)))

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


def co_cluster_tokens(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    num_q_centroids: int,
    num_k_centroids: int,
    max_iters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster query/key tokens with the SVOO low-memory co-clustering loop.

    This mirrors training_free/SVOO/svoo/co_clustering.py: initialize Q/K
    centroids from random tokens, then alternate K clustering in Q-centroid
    profile space and Q clustering in K-centroid profile space.
    """
    if max_iters <= 0:
        raise ValueError("SVOO co_cluster_tokens requires max_iters > 0")
    if not (q_flat.is_cuda and k_flat.is_cuda):
        raise RuntimeError("SVOO co_cluster_tokens requires CUDA tensors")

    batch_heads, seq_len, head_dim = q_flat.shape
    q_clusters = min(int(num_q_centroids), seq_len)
    k_clusters = min(int(num_k_centroids), seq_len)
    device = q_flat.device

    q_indices = torch.randint(0, seq_len, (batch_heads, q_clusters), device=device)
    q_centroids = torch.gather(q_flat, 1, q_indices.unsqueeze(-1).expand(-1, -1, head_dim))
    k_indices = torch.randint(0, seq_len, (batch_heads, k_clusters), device=device)
    k_centroids = torch.gather(k_flat, 1, k_indices.unsqueeze(-1).expand(-1, -1, head_dim))

    q_labels = q_sizes = k_labels = k_sizes = None
    for _ in range(max_iters):
        profile_centroids_k = torch.matmul(k_centroids, q_centroids.transpose(-2, -1))
        profile_centroids_k = F.normalize(profile_centroids_k.float(), p=2, dim=-1, eps=1e-8).contiguous()
        k_norms = profile_norm(k_flat, q_centroids)
        k_labels = co_cluster_assign(k_flat, q_centroids, profile_centroids_k, k_norms)
        k_centroids, k_sizes = centroid_update_sorted_euclid(k_flat, k_labels, k_centroids, block_n=128)

        profile_centroids_q = torch.matmul(q_centroids, k_centroids.transpose(-2, -1))
        profile_centroids_q = F.normalize(profile_centroids_q.float(), p=2, dim=-1, eps=1e-8).contiguous()
        q_norms = profile_norm(q_flat, k_centroids)
        q_labels = co_cluster_assign(q_flat, k_centroids, profile_centroids_q, q_norms)
        q_centroids, q_sizes = centroid_update_sorted_euclid(q_flat, q_labels, q_centroids, block_n=128)

    return q_labels, q_centroids, q_sizes, k_labels, k_centroids, k_sizes
