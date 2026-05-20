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
def _centroid_update_chunk_kernel(
    x_ptr,
    sorted_idx_ptr,
    sorted_cluster_ptr,
    sum_ptr,
    count_ptr,
    B,
    N,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = pid_chunk * BLOCK_N
    if chunk_start >= N:
        return

    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N
    x_batch_base = x_ptr + b * N * D

    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, BLOCK_D)
    dim_mask = offs_dim < D

    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)
    all_tokens_idxs = tl.load(idx_batch_base + token_idx, mask=valid_tok, other=-1)

    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(
                x_batch_base + load_mask,
                mask=(cluster_mask[:, None] & dim_mask[None, :]),
                other=0.0,
            ).to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats, mask=dim_mask)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
):
    """Sparse-VideoGen sorted centroid update for Euclidean k-means."""
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]
    BLOCK_D = _block_d(D)

    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _centroid_update_chunk_kernel[grid](
        x,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        BLOCK_D,
        K,
        BLOCK_N=BLOCK_N,
    )

    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    return centroids.to(x.dtype), centroid_cnts


_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_stages=4, num_warps=wp)
    for bn in [32, 64, 128]
    for bk in [32, 64, 128]
    for wp in [4, 8]
]


def _cfg_keep(conf):
    bn = conf.kwargs["BLOCK_N"]
    bk = conf.kwargs["BLOCK_K"]
    if bn * bk < 32 * 32 and conf.num_warps > 4:
        return False
    return True


_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))


@triton.autotune(_TUNE_CONFIGS, key=["N", "K"])
@triton.jit
def _euclid_assign_kernel(
    x_ptr,
    c_ptr,
    x_sq_ptr,
    out_ptr,
    B,
    N,
    K,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_x_b,
    stride_x_n,
    stride_x_d,
    stride_c_b,
    stride_c_k,
    stride_c_d,
    stride_xsq_b,
    stride_xsq_n,
    stride_out_b,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=(n_mask[:, None] & d_mask[None, :]), other=0.0)

    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=(d_mask[:, None] & k_mask[None, :]), other=0.0)

        cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)
        cross = tl.dot(x_tile, c_tile).to(tl.float32)
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    B, N, D = x.shape
    K = centroids.shape[1]
    BLOCK_D = _block_d(D)
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int64)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)
    _euclid_assign_kernel[grid](
        x,
        centroids,
        x_sq,
        out,
        B,
        N,
        K,
        D,
        BLOCK_D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        centroids.stride(0),
        centroids.stride(1),
        centroids.stride(2),
        x_sq.stride(0),
        x_sq.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def _euclid_iter(x: torch.Tensor, x_sq: torch.Tensor, centroids: torch.Tensor):
    cluster_ids = euclid_assign_triton(x, centroids, x_sq)
    centroids_new, cluster_sizes = triton_centroid_update_sorted_euclid(
        x, cluster_ids, centroids
    )
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids, cluster_sizes


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    init_centroids: torch.Tensor | None = None,
    verbose: bool = False,
):
    """Sparse-VideoGen SAP Euclidean k-means runtime path."""
    B, N, D = x.shape
    x_sq = (x**2).sum(dim=-1)

    if init_centroids is None:
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids
    centroids = centroids.view(B, n_clusters, D)

    if max_iters <= 0:
        cluster_ids = euclid_assign_triton(x, centroids, x_sq)
        cluster_sizes = _cluster_sizes(cluster_ids, B, n_clusters, x.device)
        return cluster_ids, centroids, cluster_sizes, 0

    for it in range(max_iters):
        centroids_new, center_shift, cluster_ids, cluster_sizes = _euclid_iter(
            x, x_sq, centroids
        )
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            centroids = centroids_new
            break
        centroids = centroids_new

    return cluster_ids, centroids, cluster_sizes.to(torch.long), it + 1


def _cluster_sizes(cluster_ids: torch.Tensor, B: int, K: int, device) -> torch.Tensor:
    cluster_sizes = torch.zeros(B, K, dtype=torch.int64, device=device)
    cluster_sizes.scatter_add_(
        1,
        cluster_ids.long(),
        torch.ones_like(cluster_ids, dtype=torch.int64),
    )
    return cluster_sizes


def triton_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 10,
    init_centroids: torch.Tensor | None = None,
    final_reassign: bool = False,
    clamp_clusters: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SparseVideo adapter around Sparse-VideoGen's SAP k-means call pattern."""
    B, N, D = x.shape
    K = min(int(n_clusters), N) if clamp_clusters else int(n_clusters)
    _block_d(D)

    labels, centroids, sizes, _ = batch_kmeans_Euclid(
        x,
        K,
        max_iters=max_iters,
        init_centroids=init_centroids,
    )

    if final_reassign:
        x_sq = (x**2).sum(dim=-1)
        labels = euclid_assign_triton(x, centroids, x_sq)
        sizes = _cluster_sizes(labels, B, K, x.device)

    return labels, centroids, sizes
