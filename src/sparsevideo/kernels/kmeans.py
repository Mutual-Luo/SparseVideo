"""Triton k-means clustering kernels.

Port of: training_free/Adacluster/triton_kernel/fast_kmeans.py
Simplified for [B, N, D] input (B = batch*heads, already folded by method code).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    return 1 << (max(1, int(n)) - 1).bit_length()


def _block_d(head_dim: int) -> int:
    block = _next_pow2(head_dim)
    if block < 16:
        block = 16
    if block > 256:
        raise AssertionError(f"head_dim {head_dim} not supported, must be <= 256")
    return block


@triton.jit
def _norm_sq_kernel(
    Centroids,
    Norms,
    B, K,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_cb, stride_ck, stride_cd,
    stride_nb, stride_nk,
    BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(1)
    block_k = tl.program_id(0)
    if bid >= B:
        return

    k_offs = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    c_ptrs = Centroids + bid * stride_cb + k_offs[:, None] * stride_ck + d_offs[None, :] * stride_cd
    c = tl.load(c_ptrs, mask=(k_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)
    norms = tl.sum(c * c, axis=1)

    n_ptrs = Norms + bid * stride_nb + k_offs * stride_nk
    tl.store(n_ptrs, norms, mask=k_mask)


@triton.jit
def _assign_kernel(
    X,
    Centroids,
    CentroidNorms,
    Labels,
    B, N, K,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_cb, stride_ck, stride_cd,
    stride_nb, stride_nk,
    stride_lb, stride_ln,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bid = tl.program_id(1)
    block_n = tl.program_id(0)
    if bid >= B:
        return

    n_offs = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    x_ptrs = X + bid * stride_xb + n_offs[:, None] * stride_xn + d_offs[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0).to(tl.float32)

    min_dist = tl.full([BLOCK_N], float("inf"), dtype=tl.float32)
    min_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load centroids transposed: [D, BLOCK_K]
        c_ptrs = Centroids + bid * stride_cb + d_offs[:, None] * stride_cd + k_offs[None, :] * stride_ck
        c_t = tl.load(c_ptrs, mask=(d_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)

        # Load norms
        norm_ptrs = CentroidNorms + bid * stride_nb + k_offs * stride_nk
        norms = tl.load(norm_ptrs, mask=k_mask, other=0.0)

        dots = tl.dot(x, c_t, allow_tf32=True)
        dist = norms[None, :] - 2.0 * dots
        dist = tl.where(k_mask[None, :], dist, float("inf"))

        block_min = tl.min(dist, axis=1)
        block_argmin = tl.argmin(dist, axis=1)
        block_argmin = (block_argmin + k_start).to(tl.int32)

        update = block_min < min_dist
        min_dist = tl.where(update, block_min, min_dist)
        min_idx = tl.where(update, block_argmin, min_idx)

    l_ptrs = Labels + bid * stride_lb + n_offs * stride_ln
    tl.store(l_ptrs, min_idx, mask=n_mask)


@triton.jit
def _update_centroids_kernel(
    X,
    Labels,
    NewCentroids,
    Counts,
    B, N, K,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,
    stride_lb, stride_ln,
    stride_cb, stride_ck, stride_cd,
    stride_ctb, stride_ctk,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(1)
    block_n = tl.program_id(0)

    n_range = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    l_ptrs = Labels + bid * stride_lb + n_range * stride_ln
    labels = tl.load(l_ptrs, mask=n_mask, other=0)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    x_ptrs = X + bid * stride_xb + n_range[:, None] * stride_xn + d_range[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0)

    c_ptrs = NewCentroids + bid * stride_cb + labels[:, None] * stride_ck + d_range[None, :] * stride_cd
    tl.atomic_add(c_ptrs, x.to(NewCentroids.type.element_ty), mask=(n_mask[:, None] & d_mask[None, :]))

    ct_ptrs = Counts + bid * stride_ctb + labels * stride_ctk
    tl.atomic_add(ct_ptrs, tl.full([BLOCK_N], 1, dtype=tl.int32), mask=n_mask)


def triton_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 10,
    init_centroids: torch.Tensor | None = None,
    preserve_empty_centroids: bool = True,
    final_reassign: bool = True,
    clamp_clusters: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-accelerated batch k-means.

    Args:
        x: [B, N, D] input tokens
        n_clusters: number of clusters
        max_iters: maximum iterations
        init_centroids: optional [B, K, D] initial centroids
        preserve_empty_centroids: keep the previous centroid for empty clusters.
            AdaCluster upstream resets empty clusters to zero, so that method
            passes False.
        final_reassign: run one final assignment after centroid updates. The
            generic default keeps self-consistent labels for new centroids.
            AdaCluster upstream returns labels/counts from the last update
            iteration instead, so it passes False.
        clamp_clusters: cap cluster count at N. Most methods do this before
            calling k-means; SVOO routing buckets intentionally keep the
            upstream n_clusters value because random initialization may sample
            duplicate centers.

    Returns:
        labels: [B, N] cluster assignments
        centroids: [B, K, D] final centroids
        sizes: [B, K] cluster sizes
    """
    B, N, D = x.shape
    K = min(n_clusters, N) if clamp_clusters else int(n_clusters)
    device = x.device

    BLOCK_D = _block_d(D)

    if init_centroids is not None and init_centroids.shape == (B, K, D):
        centroids = init_centroids.to(device=device, dtype=x.dtype).contiguous()
    else:
        idx = torch.randint(0, N, (B, K), device=device)
        centroids = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()

    def _next_pow2(n):
        return 1 << (max(1, n) - 1).bit_length()

    BLOCK_N = min(64, max(16, _next_pow2(N)))
    BLOCK_K = min(64, max(16, _next_pow2(K)))
    BLOCK_K_NORM = min(64, _next_pow2(K))

    labels = torch.empty(B, N, dtype=torch.int32, device=device)
    norms = torch.empty(B, K, dtype=torch.float32, device=device)
    counts = None

    for _ in range(max_iters):
        # Step 1: compute centroid norms
        grid_norm = (triton.cdiv(K, BLOCK_K_NORM), B)
        _norm_sq_kernel[grid_norm](
            centroids, norms,
            B, K, D=D, BLOCK_D=BLOCK_D,
            stride_cb=centroids.stride(0), stride_ck=centroids.stride(1), stride_cd=centroids.stride(2),
            stride_nb=norms.stride(0), stride_nk=norms.stride(1),
            BLOCK_K=BLOCK_K_NORM,
        )

        # Step 2: assign clusters
        grid_assign = (triton.cdiv(N, BLOCK_N), B)
        _assign_kernel[grid_assign](
            x, centroids, norms, labels,
            B, N, K, D=D, BLOCK_D=BLOCK_D,
            stride_xb=x.stride(0), stride_xn=x.stride(1), stride_xd=x.stride(2),
            stride_cb=centroids.stride(0), stride_ck=centroids.stride(1), stride_cd=centroids.stride(2),
            stride_nb=norms.stride(0), stride_nk=norms.stride(1),
            stride_lb=labels.stride(0), stride_ln=labels.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Step 3: update centroids via atomic add
        new_centroids = torch.zeros(B, K, D, dtype=torch.float32, device=device)
        counts = torch.zeros(B, K, dtype=torch.int32, device=device)

        grid_update = (triton.cdiv(N, BLOCK_N), B)
        _update_centroids_kernel[grid_update](
            x, labels, new_centroids, counts,
            B, N, K, D=D, BLOCK_D=BLOCK_D,
            stride_xb=x.stride(0), stride_xn=x.stride(1), stride_xd=x.stride(2),
            stride_lb=labels.stride(0), stride_ln=labels.stride(1),
            stride_cb=new_centroids.stride(0), stride_ck=new_centroids.stride(1), stride_cd=new_centroids.stride(2),
            stride_ctb=counts.stride(0), stride_ctk=counts.stride(1),
            BLOCK_N=BLOCK_N,
        )

        safe_counts = counts.clamp(min=1).unsqueeze(-1).float()
        new_centroids = new_centroids / safe_counts
        if preserve_empty_centroids:
            empty_mask = (counts == 0).unsqueeze(-1)
            new_centroids = torch.where(empty_mask, centroids.to(torch.float32), new_centroids)
        new_centroids = new_centroids.to(x.dtype)

        if torch.allclose(centroids, new_centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids.contiguous()

    if not final_reassign and counts is not None:
        return labels, centroids, counts.to(torch.long)

    # Final assignment with converged centroids
    _norm_sq_kernel[(triton.cdiv(K, BLOCK_K_NORM), B)](
        centroids, norms,
        B, K, D=D, BLOCK_D=BLOCK_D,
        stride_cb=centroids.stride(0), stride_ck=centroids.stride(1), stride_cd=centroids.stride(2),
        stride_nb=norms.stride(0), stride_nk=norms.stride(1),
        BLOCK_K=BLOCK_K_NORM,
    )
    _assign_kernel[(triton.cdiv(N, BLOCK_N), B)](
        x, centroids, norms, labels,
        B, N, K, D=D, BLOCK_D=BLOCK_D,
        stride_xb=x.stride(0), stride_xn=x.stride(1), stride_xd=x.stride(2),
        stride_cb=centroids.stride(0), stride_ck=centroids.stride(1), stride_cd=centroids.stride(2),
        stride_nb=norms.stride(0), stride_nk=norms.stride(1),
        stride_lb=labels.stride(0), stride_ln=labels.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    sizes = torch.zeros(B, K, dtype=torch.long, device=device)
    sizes.scatter_add_(1, labels.long(), torch.ones(B, N, dtype=torch.long, device=device))

    return labels, centroids, sizes
