from __future__ import annotations

import csv
from pathlib import Path

import torch

from .text import pad_text_clusters


_SPARSITY_LOOKUP_CACHE = {}


def should_recluster(current_step, start_reuse_step, reuse_interval):
    if current_step is None or start_reuse_step is None or start_reuse_step <= 0:
        return True
    if current_step < start_reuse_step:
        return True
    interval = max(1, int(reuse_interval or 1))
    return (current_step - start_reuse_step) % interval == 0


def load_sparsity_lookup(csv_path):
    path = Path(csv_path).expanduser()
    cache_key = str(path)
    if cache_key in _SPARSITY_LOOKUP_CACHE:
        return _SPARSITY_LOOKUP_CACHE[cache_key]
    if not path.exists():
        _SPARSITY_LOOKUP_CACHE[cache_key] = {}
        return {}
    lookup = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(int(row["Step"]), int(row["Layer"]), int(row["Head"]))] = float(row["Sparsity"])
    _SPARSITY_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def dynamic_min_kc_ratio(cfg, state, current_step, layer_idx, batch, num_heads, device):
    ratio = cfg["min_kc_ratio"]
    path = cfg["sparsity_csv_path"]
    if not cfg["use_dynamic_min_kc_ratio"] or not path or current_step is None:
        return ratio

    path_key = str(Path(path).expanduser())
    if state.get("sparsity_lookup_path") != path_key:
        state["sparsity_lookup"] = load_sparsity_lookup(path)
        state["sparsity_lookup_path"] = path_key

    lookup = state.get("sparsity_lookup") or {}
    values = []
    for head_idx in range(num_heads):
        value = lookup.get((int(current_step), int(layer_idx), int(head_idx)))
        if value is None:
            return ratio
        if cfg["dynamic_min_kc_ratio_min"] is not None:
            value = max(value, float(cfg["dynamic_min_kc_ratio_min"]))
        if cfg["dynamic_min_kc_ratio_max"] is not None:
            value = min(value, float(cfg["dynamic_min_kc_ratio_max"]))
        values.append(value)

    return torch.tensor(values * batch, device=device, dtype=torch.float32)


def weighted_softmax(scores, weights):
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)


def identify_dynamic_map(query_centroids, key_centroids, q_cluster_sizes, k_cluster_sizes, p, min_kc_ratio=0):
    """Select active key clusters with upstream SVOO top-p semantics."""
    batch_heads, q_clusters, head_dim = query_centroids.shape
    k_clusters = key_centroids.shape[1]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (head_dim**0.5)
    weighted_attn_probs = weighted_softmax(attn_scores, k_cluster_sizes.unsqueeze(-2))
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if isinstance(min_kc_ratio, torch.Tensor):
        ratios = min_kc_ratio.flatten().to(device=device, dtype=torch.float32)
        if ratios.numel() == batch_heads:
            for bh in range(batch_heads):
                head_ratio = float(ratios[bh].item())
                if head_ratio > 0:
                    preserve_length = int(head_ratio * k_clusters)
                    remove_indices[bh, :, :preserve_length] = False
        elif ratios.numel() == 1:
            head_ratio = float(ratios.item())
            if head_ratio > 0:
                preserve_length = int(head_ratio * k_clusters)
                remove_indices[..., :preserve_length] = False
        else:
            raise ValueError(
                "min_kc_ratio tensor must be scalar or have one value per batch-head"
            )
    elif isinstance(min_kc_ratio, (list, tuple)):
        if len(min_kc_ratio) == batch_heads:
            for bh, head_ratio in enumerate(min_kc_ratio):
                if float(head_ratio) > 0:
                    preserve_length = int(float(head_ratio) * k_clusters)
                    remove_indices[bh, :, :preserve_length] = False
        else:
            raise ValueError("min_kc_ratio list must have one value per batch-head")
    elif isinstance(min_kc_ratio, (int, float)) and float(min_kc_ratio) > 0:
        preserve_length = int(float(min_kc_ratio) * k_clusters)
        remove_indices[..., :preserve_length] = False

    sorted_clusters_to_keep = ~remove_indices
    dynamic_map = torch.zeros(batch_heads, q_clusters, k_clusters, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map


def svoo_attention(
    query,
    key,
    value,
    cfg,
    state,
    current_step=None,
    layer_idx=0,
    initialize_only=False,
    text_len=0,
):
    """SVOO: Co-clustering + dynamic block-sparse attention.

    query/key/value: [B, N, H, D]
    """
    implementation = cfg.get("implementation", "native")
    if implementation != "native":
        raise NotImplementedError(
            "SVOO implementation must be 'native'. Reference-repo bridges are not used by SparseVideo."
        )

    if not query.is_cuda:
        raise RuntimeError("svoo sparse path requires CUDA")

    from ...kernels.kmeans import triton_kmeans
    from ...kernels.block_sparse_attn import block_sparse_attention
    from ...kernels.co_cluster import co_cluster_tokens
    from ...kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    B, N, H, D = query.shape
    if B != 1:
        raise RuntimeError("SVOO follows the upstream implementation and currently requires batch size 1")
    scale = D ** -0.5
    text_len = max(0, min(int(text_len or 0), N))
    video_N = N - text_len
    if video_N <= 0:
        raise RuntimeError("svoo sparse path requires at least one video token")

    q_flat = query.permute(0, 2, 1, 3).reshape(B * H, N, D)
    k_flat = key.permute(0, 2, 1, 3).reshape(B * H, N, D)
    v_flat = value.permute(0, 2, 1, 3).reshape(B * H, N, D)
    q_video = q_flat[:, :video_N]
    k_video = k_flat[:, :video_N]

    num_q_centroids = cfg["num_q_centroids"]
    num_k_centroids = cfg["num_k_centroids"]
    nqc = min(num_q_centroids, video_N)
    nkc = min(num_k_centroids, video_N)

    cache_key = (B * H, video_N, text_len, D, nqc, nkc, str(q_flat.device))
    cached = state.get("cached_clustering")
    do_recluster = should_recluster(
        current_step, cfg["start_reuse_step"], cfg["reuse_interval"],
    )
    if not do_recluster and cached is not None and cached.get("cache_key") == cache_key:
        q_labels = cached["q_labels"]
        q_centroids_token = cached["q_centroids"]
        q_sizes = cached["q_sizes"]
        k_labels = cached["k_labels"]
        k_centroids = cached["k_centroids"]
        k_sizes = cached["k_sizes"]
    else:
        kmeans_iters = (
            cfg["kmeans_iter_step"] if state["centroids_init"] else cfg["kmeans_iter_init"]
        )
        if cfg["use_svoo"]:
            (
                q_labels,
                q_centroids_token,
                q_sizes,
                k_labels,
                k_centroids,
                k_sizes,
            ) = co_cluster_tokens(
                q_video,
                k_video,
                nqc,
                nkc,
                max_iters=kmeans_iters,
            )
        else:
            q_labels, q_centroids_token, q_sizes = triton_kmeans(
                q_video, nqc, kmeans_iters,
                init_centroids=state.get("prev_q_centroids"),
            )
            k_labels, k_centroids, k_sizes = triton_kmeans(
                k_video, nkc, kmeans_iters,
                init_centroids=state.get("prev_k_centroids"),
            )
            state["prev_q_centroids"] = q_centroids_token.detach()
            state["prev_k_centroids"] = k_centroids.detach()

        state["centroids_init"] = True
        state["cached_clustering"] = {
            "cache_key": cache_key,
            "q_labels": q_labels.detach(),
            "q_centroids": q_centroids_token.detach(),
            "q_sizes": q_sizes.detach(),
            "k_labels": k_labels.detach(),
            "k_centroids": k_centroids.detach(),
            "k_sizes": k_sizes.detach(),
        }

    if initialize_only:
        return None

    min_ratio = dynamic_min_kc_ratio(cfg, state, current_step, layer_idx, B, H, q_flat.device)
    dynamic_map = identify_dynamic_map(
        q_centroids_token,
        k_centroids,
        q_sizes,
        k_sizes,
        cfg["top_p_kmeans"],
        min_ratio,
    )

    q_sorted_idx = q_labels.argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)
    if text_len > 0:
        dynamic_map_4d, q_sizes_4d, k_sizes_4d, q_sorted_idx = pad_text_clusters(
            dynamic_map.unsqueeze(1),
            q_sizes.unsqueeze(1),
            k_sizes.unsqueeze(1),
            q_sorted_idx,
            text_len=text_len,
            prompt_length=cfg.get("prompt_length"),
        )
        dynamic_map = dynamic_map_4d.squeeze(1)
        q_sizes = q_sizes_4d.squeeze(1)
        k_sizes = k_sizes_4d.squeeze(1)
        text_idx = torch.arange(video_N, N, device=q_flat.device, dtype=k_sorted_idx.dtype)
        text_idx = text_idx.expand(B * H, text_len)
        k_sorted_idx = torch.cat([k_sorted_idx, text_idx], dim=-1)

    q_sorted = torch.gather(q_flat, 1, q_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    sparse_backend = cfg.get("sparse_backend", "flashinfer")
    q_sizes_i32 = q_sizes.to(torch.int32)
    k_sizes_i32 = k_sizes.to(torch.int32)
    if sparse_backend == "flashinfer":
        if not HAS_FLASHINFER:
            raise RuntimeError("svoo sparse_backend=flashinfer requires flashinfer.sparse")
        out_sorted = variable_block_sparse_attn(
            q_sorted, k_sorted, v_sorted,
            dynamic_map, q_sizes_i32, k_sizes_i32,
        )
    elif sparse_backend == "triton":
        out_sorted = block_sparse_attention(
            q_sorted, k_sorted, v_sorted,
            q_sizes, k_sizes, dynamic_map, scale,
        )
    else:
        raise ValueError("svoo sparse_backend must be 'flashinfer' or 'triton'")

    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_q_idx.unsqueeze(-1).expand(-1, -1, D))

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)
