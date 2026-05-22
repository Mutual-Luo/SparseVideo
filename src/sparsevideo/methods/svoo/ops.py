from __future__ import annotations

import csv
from pathlib import Path

import torch

from ...kernels.dynamic_map import identify_dynamic_map, identify_dynamic_map_global, weighted_softmax
from ...kernels.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton
from .._layout import infer_video_frame_shape
from .text import pad_text_clusters


_SPARSITY_LOOKUP_CACHE = {}


def resolve_sparsity_csv_path(csv_path, base_dir=None):
    path = Path(str(csv_path)).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    resolved = path.resolve(strict=False)
    if "training_free" in path.parts or "training_free" in resolved.parts:
        raise RuntimeError(
            "Refusing SVOO sparsity_csv_path inside training_free; "
            "SparseVideo runtime sparsity profiles must live under src/sparsevideo."
        )
    return resolved


def should_recluster(current_step, start_reuse_step, reuse_interval):
    if current_step is None or start_reuse_step is None or start_reuse_step <= 0:
        return True
    if current_step < start_reuse_step:
        return True
    interval = max(1, int(reuse_interval or 1))
    return (current_step - start_reuse_step) % interval == 0


def load_sparsity_lookup(csv_path):
    path = resolve_sparsity_csv_path(csv_path)
    cache_key = str(path)
    if cache_key in _SPARSITY_LOOKUP_CACHE:
        return _SPARSITY_LOOKUP_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(
            f"SVOO dynamic_min_kc_ratio requires sparsity_csv_path to exist: {path}"
        )
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
    prompt_length=None,
    model_type="wan",
    scheduler_timestep=None,
    total_layers=0,
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
    scale = D ** -0.5
    text_len = max(0, min(int(text_len or 0), N))
    video_N = N - text_len
    if video_N <= 0:
        raise RuntimeError("svoo sparse path requires at least one video token")

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    # CFG batches are independent after folding batch and head into the leading
    # dimension, matching the block-sparse kernels' batch-head contract.
    q_flat = q_bhsd.reshape(B * H, N, D)
    k_flat = k_bhsd.reshape(B * H, N, D)
    v_flat = v_bhsd.reshape(B * H, N, D)
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
                final_reassign=False,
            )
            k_labels, k_centroids, k_sizes = triton_kmeans(
                k_video, nkc, kmeans_iters,
                init_centroids=state.get("prev_k_centroids"),
                final_reassign=False,
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
    if cfg.get("use_global_constraints"):
        num_frames, frame_size = _svoo_frame_layout(video_N, model_type=model_type)
        dynamic_map = identify_dynamic_map_global(
            q_centroids_token.view(B, H, q_centroids_token.shape[1], D),
            k_centroids.view(B, H, k_centroids.shape[1], D),
            q_sizes.view(B, H, q_sizes.shape[1]),
            k_sizes.view(B, H, k_sizes.shape[1]),
            cfg["top_p_kmeans"],
            min_ratio,
            key_tokens=k_video.view(B, H, video_N, D),
            k_labels=k_labels,
            num_frame=num_frames,
            frame_size=frame_size,
            context_length=0,
            timestep=scheduler_timestep if scheduler_timestep is not None else 0,
            layer_idx=layer_idx,
            num_layers=total_layers,
            lambda_schedule=cfg["lambda_schedule"],
            diverse_top_p_k=cfg["diverse_top_p_k"],
        ).view(B * H, q_centroids_token.shape[1], k_centroids.shape[1])
    else:
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
            prompt_length=prompt_length if prompt_length is not None else cfg.get("prompt_length"),
        )
        dynamic_map = dynamic_map_4d.squeeze(1)
        q_sizes = q_sizes_4d.squeeze(1)
        k_sizes = k_sizes_4d.squeeze(1)
        text_idx = torch.arange(video_N, N, device=q_flat.device, dtype=k_sorted_idx.dtype)
        text_idx = text_idx.expand(B * H, text_len)
        k_sorted_idx = torch.cat([k_sorted_idx, text_idx], dim=-1)

    q_sorted_bhsd, q_sorted_idx = permute_tensor_by_labels_triton(
        q_bhsd, None, dim=2, sorted_indices=q_sorted_idx,
    )
    k_sorted_bhsd, k_sorted_idx = permute_tensor_by_labels_triton(
        k_bhsd, None, dim=2, sorted_indices=k_sorted_idx,
    )
    v_sorted_bhsd, _ = permute_tensor_by_labels_triton(
        v_bhsd, None, dim=2, sorted_indices=k_sorted_idx,
    )
    q_sorted = q_sorted_bhsd.reshape(B * H, N, D)
    k_sorted = k_sorted_bhsd.reshape(B * H, N, D)
    v_sorted = v_sorted_bhsd.reshape(B * H, N, D)
    if cfg.get("enable_mem_save", True):
        del query, key, value
        del q_flat, k_flat, v_flat
        del q_bhsd, k_bhsd, v_bhsd
        del q_sorted_bhsd, k_sorted_bhsd, v_sorted_bhsd

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
    if cfg.get("enable_mem_save", True):
        del q_sorted, k_sorted, v_sorted

    out_bhsd = out_sorted.reshape(B, H, N, D)
    out_bhsd = apply_inverse_permutation_triton(out_bhsd, q_sorted_idx, dim=2)
    if cfg.get("enable_mem_save", True):
        del out_sorted

    return out_bhsd.permute(0, 2, 1, 3)


def _svoo_frame_layout(video_len, model_type):
    num_frames, frame_h, frame_w = infer_video_frame_shape(video_len, model_type=model_type)
    return num_frames, frame_h * frame_w

