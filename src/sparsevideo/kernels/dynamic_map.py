from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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
    """Select active key clusters with Sparse-VideoGen/SVOO top-p semantics."""
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


def identify_dynamic_map_global(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
    key_tokens=None,
    k_labels=None,
    num_frame=0,
    frame_size=0,
    context_length=0,
    timestep=0,
    layer_idx=0,
    num_layers=0,
    lambda_schedule="linear",
    diverse_top_p_k=0.0,
):
    """SVOO dynamic map with optional global diversity constraints."""
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()
    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)

    if key_tokens is not None and k_labels is not None:
        k_diversity = compute_cluster_diversity_vectorized(
            key_tokens, k_labels, k_cluster_sizes,
            num_frame, frame_size, context_length,
            kc_num, device,
        )
        diversity_gains = k_diversity.unsqueeze(2).expand(-1, -1, qc_num, -1)
    else:
        diversity_gains = torch.zeros_like(weighted_attn_probs)

    if diversity_gains.max() > diversity_gains.min():
        diversity_gains = (diversity_gains - diversity_gains.min()) / (
            diversity_gains.max() - diversity_gains.min() + 1e-8
        )

    if diverse_top_p_k > 0 and diverse_top_p_k < p:
        attn_p = p - diverse_top_p_k
        sorted_attn, sorted_attn_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        remove_attn = cumsum_attn > attn_p
        remove_attn[..., 1:] = remove_attn[..., :-1].clone()
        remove_attn[..., 0] = False
        _preserve_min_ratio(remove_attn, min_kc_ratio, H, kc_num)

        selected_by_attn_sorted = ~remove_attn
        attn_selected_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
        attn_selected_map.scatter_(-1, sorted_attn_indices, selected_by_attn_sorted)

        remaining_diversity = diversity_gains.clone().masked_fill(attn_selected_map, float("-inf"))
        sorted_diversity, sorted_diversity_indices = torch.sort(remaining_diversity, dim=-1, descending=True)
        remaining_diversity_valid = sorted_diversity.clone().masked_fill(sorted_diversity == float("-inf"), 0.0)
        remaining_diversity_sum = remaining_diversity_valid.sum(dim=-1, keepdim=True)
        remaining_diversity_probs = remaining_diversity_valid / (remaining_diversity_sum + 1e-8)

        cumsum_diversity = torch.cumsum(remaining_diversity_probs, dim=-1)
        remove_diversity = cumsum_diversity > diverse_top_p_k
        remove_diversity[..., 1:] = remove_diversity[..., :-1].clone()
        remove_diversity[..., 0] = False
        sorted_attn_selected = torch.gather(attn_selected_map, dim=-1, index=sorted_diversity_indices)
        remove_diversity = remove_diversity | sorted_attn_selected

        selected_by_diversity_sorted = ~remove_diversity
        diversity_selected_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
        diversity_selected_map.scatter_(-1, sorted_diversity_indices, selected_by_diversity_sorted)
        return attn_selected_map | diversity_selected_map

    lambda_weight = compute_lambda_schedule(timestep, layer_idx, num_layers, lambda_schedule)
    combined_scores = lambda_weight * weighted_attn_probs + (1 - lambda_weight) * diversity_gains
    sorted_scores, sorted_indices = torch.sort(combined_scores, dim=-1, descending=True)
    cumsum_scores = torch.cumsum(sorted_scores, dim=-1)
    remove_indices = cumsum_scores > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False
    _preserve_min_ratio(remove_indices, min_kc_ratio, H, kc_num)

    sorted_clusters_to_keep = ~remove_indices
    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map


def _preserve_min_ratio(remove_indices, min_kc_ratio, num_heads, k_clusters):
    if isinstance(min_kc_ratio, torch.Tensor):
        ratios = min_kc_ratio.flatten().to(device=remove_indices.device, dtype=torch.float32)
        if ratios.numel() == num_heads:
            for h in range(num_heads):
                head_ratio = float(ratios[h].item())
                if head_ratio > 0:
                    remove_indices[:, h, :, : int(head_ratio * k_clusters)] = False
        elif ratios.numel() == 1:
            head_ratio = float(ratios.item())
            if head_ratio > 0:
                remove_indices[..., : int(head_ratio * k_clusters)] = False
        elif ratios.numel() == remove_indices.shape[0] * num_heads:
            ratios = ratios.view(remove_indices.shape[0], num_heads)
            for b in range(remove_indices.shape[0]):
                for h in range(num_heads):
                    head_ratio = float(ratios[b, h].item())
                    if head_ratio > 0:
                        remove_indices[b, h, :, : int(head_ratio * k_clusters)] = False
        else:
            raise ValueError("min_kc_ratio tensor must be scalar or have one value per head")
    elif isinstance(min_kc_ratio, (list, tuple)):
        if len(min_kc_ratio) == num_heads:
            for h, head_ratio in enumerate(min_kc_ratio):
                if float(head_ratio) > 0:
                    remove_indices[:, h, :, : int(float(head_ratio) * k_clusters)] = False
        else:
            raise ValueError("min_kc_ratio list must have one value per head")
    elif isinstance(min_kc_ratio, (int, float)) and float(min_kc_ratio) > 0:
        remove_indices[..., : int(float(min_kc_ratio) * k_clusters)] = False


def compute_cluster_diversity_vectorized(
    tokens,
    labels,
    cluster_sizes,
    num_frame,
    frame_size,
    context_length,
    num_clusters,
    device,
):
    B, H, S, _ = tokens.shape
    video_length = num_frame * frame_size if num_frame > 0 and frame_size > 0 else S

    if labels.shape[0] == B * H:
        labels_reshaped = labels.view(B, H, S)
        labels_for_coverage = labels_reshaped[:, 0, :]
        labels_for_variance = labels_reshaped
    else:
        labels_for_coverage = labels
        labels_for_variance = labels.unsqueeze(1).expand(-1, H, -1)

    spatiotemporal_coverage = compute_spatiotemporal_coverage_fully_vectorized(
        labels_for_coverage, num_frame, frame_size, context_length,
        video_length, S, num_clusters, device,
    )
    cluster_variance = compute_cluster_variance_fully_vectorized(
        tokens, labels_for_variance, cluster_sizes, num_clusters,
    )
    spatiotemporal_coverage = spatiotemporal_coverage.unsqueeze(1).expand(-1, H, -1)
    return spatiotemporal_coverage + cluster_variance


def compute_spatiotemporal_coverage_fully_vectorized(
    labels,
    num_frame,
    frame_size,
    context_length,
    video_length,
    seq_len,
    num_clusters,
    device,
):
    B, S = labels.shape
    if num_frame == 0 or frame_size == 0:
        cluster_mask = (labels.unsqueeze(-1) == torch.arange(num_clusters, device=device)).float()
        return cluster_mask.sum(dim=1) / seq_len

    frame_indices = torch.arange(S, device=device)
    video_mask = frame_indices < video_length
    frame_ids = torch.where(video_mask, frame_indices // frame_size, torch.full_like(frame_indices, -1))
    cluster_mask = (labels.unsqueeze(-1) == torch.arange(num_clusters, device=device)).float()

    video_mask_expanded = video_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, num_clusters).float()
    video_cluster_mask = cluster_mask * video_mask_expanded
    spatial_coverage = video_cluster_mask.sum(dim=1) / video_length

    frame_ids_expanded = frame_ids.unsqueeze(0).unsqueeze(-1).expand(B, -1, num_clusters)
    valid_frame_mask = (frame_ids_expanded >= 0).float()
    valid_cluster_mask = video_cluster_mask * valid_frame_mask
    frame_ids_clamped = torch.clamp(frame_ids_expanded, 0, num_frame - 1)
    frame_onehot = F.one_hot(frame_ids_clamped.long(), num_classes=num_frame).float()
    frame_onehot = frame_onehot * valid_cluster_mask.unsqueeze(-1)
    unique_frames_per_cluster = (frame_onehot.sum(dim=1) > 0).float().sum(dim=-1)
    temporal_coverage = unique_frames_per_cluster / num_frame
    return (temporal_coverage + spatial_coverage) / 2.0


def compute_cluster_variance_fully_vectorized(tokens, labels, cluster_sizes, num_clusters):
    B, H, S, D = tokens.shape
    if labels.dim() == 2:
        labels_expanded = labels.unsqueeze(1).expand(-1, H, -1)
    else:
        labels_expanded = labels

    labels_onehot = F.one_hot(labels_expanded, num_classes=num_clusters).float()
    tokens_float = tokens.float()
    cluster_sums = torch.einsum("bhsd,bhsk->bhkd", tokens_float, labels_onehot)
    cluster_sizes_expanded = cluster_sizes.unsqueeze(-1).float().clamp(min=1.0)
    centroids = cluster_sums / cluster_sizes_expanded

    labels_gather_idx = labels_expanded.long().unsqueeze(-1).expand(-1, -1, -1, D)
    token_centroids = torch.gather(centroids, dim=2, index=labels_gather_idx)
    distances = torch.norm(tokens_float - token_centroids, dim=-1)
    cluster_variance_sum = torch.einsum("bhs,bhsk->bhk", distances, labels_onehot)
    cluster_variance = cluster_variance_sum / cluster_sizes_expanded.squeeze(-1).clamp(min=1.0)
    return torch.clamp(cluster_variance / 10.0, 0.0, 1.0)


def compute_lambda_schedule(timestep, layer_idx, num_layers, schedule_type="linear"):
    if schedule_type == "constant":
        return 0.5
    if isinstance(timestep, torch.Tensor):
        timestep_val = timestep.item() if timestep.numel() == 1 else timestep.flatten()[0].item()
    else:
        timestep_val = timestep
    if timestep_val is None:
        timestep_val = 0

    if schedule_type == "linear":
        timestep_weight = float(timestep_val) / 1000.0
    elif schedule_type == "cosine":
        import math

        timestep_weight = 0.5 * (1 - math.cos(float(timestep_val) / 1000.0 * math.pi))
    else:
        timestep_weight = float(timestep_val) / 1000.0

    layer_weight = layer_idx / num_layers if num_layers > 0 else 0.5
    lambda_weight = 0.6 * timestep_weight + 0.4 * layer_weight
    return max(0.0, min(1.0, lambda_weight))


# ---------------------------------------------------------------------------
# Error-Aware Reduction (EAR) block selection
#
# Ported from Sparse-VideoGen (SVG-EAR, branch ear-wan22-support). Instead of
# selecting blocks purely by centroid attention mass (identify_dynamic_map),
# EAR estimates the squared output error each (query-cluster, key-cluster) block
# would incur if approximated by its centroid, and keeps the most cost-effective
# blocks within a centroid-attention budget. Operates on [B, H, ...] tensors.
# ---------------------------------------------------------------------------


@triton.jit
def _precompute_v_stats_kernel(
    value_ptr,
    value_centroids_ptr,
    kc_offsets_ptr,
    value_norm_sq_ptr,
    centroid_value_dot_ptr,
    centroid_value_sq_ptr,
    kc_num,
    num_heads,
    D: tl.constexpr,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_vcb,
    stride_vch,
    stride_vckc,
    stride_vcd,
    stride_norm_b, stride_norm_h, stride_norm_s,
    stride_vc_sq_b, stride_vc_sq_h, stride_vc_sq_kc,
    BLOCK_S: tl.constexpr,
):
    # Grid layout:
    # - axis 0: key-cluster index
    # - axis 1: flattened (batch, head) index
    pid_bh = tl.program_id(axis=1)
    kc_idx = tl.program_id(axis=0)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    offset_base = pid_bh * (kc_num + 1)
    token_start = tl.load(kc_offsets_ptr + offset_base + kc_idx)
    token_end = tl.load(kc_offsets_ptr + offset_base + kc_idx + 1)

    offs_d = tl.arange(0, D)
    value_centroid = tl.load(
        value_centroids_ptr
        + batch_idx * stride_vcb
        + head_idx * stride_vch
        + kc_idx * stride_vckc
        + offs_d * stride_vcd,
        mask=offs_d < D,
        other=0.0,
    )

    # Cache ||v_c||^2 once per key cluster so the main kernel only consumes scalars.
    centroid_value_sq = tl.sum(value_centroid * value_centroid)
    tl.store(
        centroid_value_sq_ptr + batch_idx * stride_vc_sq_b + head_idx * stride_vc_sq_h + kc_idx,
        centroid_value_sq,
    )

    offs_s = tl.arange(0, BLOCK_S)
    for token_offset in range(token_start, token_end, BLOCK_S):
        block_size = tl.minimum(BLOCK_S, token_end - token_offset)
        token_mask = offs_s < block_size

        value_block = tl.load(
            value_ptr
            + batch_idx * stride_vb
            + head_idx * stride_vh
            + (token_offset + offs_s)[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=token_mask[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )

        value_norm_sq = tl.sum(value_block * value_block, axis=1)
        centroid_value_dot = tl.sum(value_centroid[None, :] * value_block, axis=1)

        tl.store(
            value_norm_sq_ptr + batch_idx * stride_norm_b + head_idx * stride_norm_h + token_offset + offs_s,
            value_norm_sq,
            mask=token_mask,
        )
        tl.store(
            centroid_value_dot_ptr + batch_idx * stride_norm_b + head_idx * stride_norm_h + token_offset + offs_s,
            centroid_value_dot,
            mask=token_mask,
        )


@triton.jit
def _error_estimation_kernel(
    query_centroids_ptr,
    key_tokens_ptr,
    kc_offsets_ptr,
    qc_sz_ptr,
    kc_sz_ptr,
    efficiency_scores_ptr,
    value_norm_sq_ptr,
    centroid_value_dot_ptr,
    centroid_logits_ptr,
    centroid_value_sq_ptr,
    qc_num,
    kc_num,
    scale,
    B,
    num_heads,
    gamma,
    eps,
    stride_qb,
    stride_qh,
    stride_qqc,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_qc_sz_b, stride_qc_sz_h, stride_qc_sz_qc,
    stride_kc_sz_b, stride_kc_sz_h, stride_kc_sz_kc,
    stride_eb, stride_eh, stride_eqc, stride_ekc,
    stride_norm_b, stride_norm_h, stride_norm_s,
    stride_cb, stride_ch, stride_cqc, stride_ckc,
    stride_vc_sq_b, stride_vc_sq_h, stride_vc_sq_kc,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    D: tl.constexpr,
):
    # Grid layout:
    # - axis 0: query-cluster block
    # - axis 1: flattened (batch, head) index
    pid_bh = tl.program_id(axis=1)
    pid_q_block = tl.program_id(axis=0)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    query_cluster_idx = pid_q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    query_cluster_mask = query_cluster_idx < qc_num

    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    query_centroids = tl.load(
        query_centroids_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + query_cluster_idx[:, None] * stride_qqc
        + offs_d[None, :] * stride_qd,
        mask=query_cluster_mask[:, None],
        other=0.0,
    )

    query_cluster_sizes = tl.load(
        qc_sz_ptr
        + batch_idx * stride_qc_sz_b
        + head_idx * stride_qc_sz_h
        + query_cluster_idx * stride_qc_sz_qc,
        mask=query_cluster_mask,
        other=0.0,
    ).to(tl.float32)

    offset_base = pid_bh * (kc_num + 1)
    key_cluster_start = tl.load(kc_offsets_ptr + offset_base)
    centroid_logits_base = (
        centroid_logits_ptr
        + batch_idx * stride_cb
        + head_idx * stride_ch
        + query_cluster_idx * stride_cqc
    )

    for kc_idx in range(kc_num):
        key_cluster_end = tl.load(kc_offsets_ptr + offset_base + kc_idx + 1)

        # All value-dependent terms are scalar loads from the precompute kernel.
        centroid_logit = tl.load(
            centroid_logits_base + kc_idx * stride_ckc,
            mask=query_cluster_mask,
            other=0.0,
        ).to(tl.float32)
        centroid_value_sq = tl.load(
            centroid_value_sq_ptr + batch_idx * stride_vc_sq_b + head_idx * stride_vc_sq_h + kc_idx
        )
        key_cluster_size = tl.load(
            kc_sz_ptr + batch_idx * stride_kc_sz_b + head_idx * stride_kc_sz_h + kc_idx
        ).to(tl.float32)

        running_max = centroid_logit
        squared_sum = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)
        cross_sum = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32)

        for token_offset in range(key_cluster_start, key_cluster_end, BLOCK_SIZE_K):
            block_size = tl.minimum(BLOCK_SIZE_K, key_cluster_end - token_offset)
            key_token_mask = offs_k < block_size

            key_block = tl.load(
                key_tokens_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + (token_offset + offs_k)[:, None] * stride_ks
                + offs_d[None, :] * stride_kd,
                mask=key_token_mask[:, None],
                other=0.0,
            )
            value_norm_sq = tl.load(
                value_norm_sq_ptr + batch_idx * stride_norm_b + head_idx * stride_norm_h + token_offset + offs_k,
                mask=key_token_mask,
                other=0.0,
            )
            centroid_value_dot = tl.load(
                centroid_value_dot_ptr + batch_idx * stride_norm_b + head_idx * stride_norm_h + token_offset + offs_k,
                mask=key_token_mask,
                other=0.0,
            )

            attn = tl.dot(query_centroids, tl.trans(key_block)) * scale
            attn = tl.where(
                query_cluster_mask[:, None] & key_token_mask[None, :],
                attn,
                -float("inf"),
            )

            block_max = tl.max(attn, axis=1)
            new_running_max = tl.maximum(running_max, block_max)

            squared_rescale = tl.exp(2.0 * (running_max - new_running_max))
            cross_rescale = tl.exp(running_max - new_running_max)

            attn_prob = tl.exp(attn - new_running_max[:, None])
            attn_prob = tl.where(
                query_cluster_mask[:, None] & key_token_mask[None, :],
                attn_prob,
                0.0,
            )

            # Keep the online softmax update explicit: one term for ||A||^2 and one for <A, v_c>.
            squared_sum = squared_sum * squared_rescale + tl.sum(attn_prob * attn_prob * value_norm_sq[None, :], axis=1)
            cross_sum = cross_sum * cross_rescale + tl.sum(attn_prob * centroid_value_dot[None, :], axis=1)

            running_max = new_running_max

        centroid_prob = tl.exp(centroid_logit - running_max)
        acc_error = squared_sum - 2.0 * centroid_prob * cross_sum
        acc_error += (centroid_prob * centroid_prob) * centroid_value_sq * key_cluster_size
        acc_error = acc_error * tl.exp(2.0 * running_max)
        acc_error = tl.maximum(acc_error, eps)

        block_area = tl.maximum(query_cluster_sizes * key_cluster_size, eps)
        efficiency_scores = tl.log(acc_error) - gamma * tl.log(block_area)

        tl.store(
            efficiency_scores_ptr
            + batch_idx * stride_eb
            + head_idx * stride_eh
            + query_cluster_idx * stride_eqc
            + kc_idx * stride_ekc,
            efficiency_scores,
            mask=query_cluster_mask,
        )

        key_cluster_start = key_cluster_end


def error_estimation_triton(
    query_centroids: torch.Tensor,
    key_centroids: torch.Tensor,
    value_centroids: torch.Tensor,
    permuted_K: torch.Tensor,
    permuted_V: torch.Tensor,
    qc_sz: torch.Tensor,
    kc_sz: torch.Tensor,
    c_logits: torch.Tensor,
    gamma: float,
    eps: float,
    BLOCK_SIZE_Q: int = 64,
    BLOCK_SIZE_K: int = 32,
):
    """Estimate log-efficiency scores for all `(query cluster, key cluster)` pairs.

    The Triton path is split into two kernels:
    1. Precompute value-side statistics that only depend on the key-cluster partition.
    2. Reuse those statistics while scanning key tokens to estimate the block error.

    All tensors use the [B, H, ...] layout.
    """

    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    seq_len = permuted_K.shape[2]
    device = query_centroids.device
    scale = D ** -0.5

    # Offsets map each key cluster to a contiguous token range in the permuted sequence.
    kc_offsets = torch.zeros((B, H, kc_num + 1), device=device, dtype=torch.long)
    torch.cumsum(kc_sz, dim=-1, out=kc_offsets[..., 1:])

    value_norm_sq = torch.empty((B, H, seq_len), device=device, dtype=torch.float32)
    centroid_value_dot = torch.empty((B, H, seq_len), device=device, dtype=torch.float32)
    centroid_value_sq = torch.empty((B, H, kc_num), device=device, dtype=torch.float32)

    precompute_grid = (kc_num, B * H)
    _precompute_v_stats_kernel[precompute_grid](
        permuted_V,
        value_centroids,
        kc_offsets,
        value_norm_sq,
        centroid_value_dot,
        centroid_value_sq,
        kc_num,
        H,
        D,
        permuted_V.stride(0),
        permuted_V.stride(1),
        permuted_V.stride(2),
        permuted_V.stride(3),
        value_centroids.stride(0),
        value_centroids.stride(1),
        value_centroids.stride(2),
        value_centroids.stride(3),
        value_norm_sq.stride(0),
        value_norm_sq.stride(1),
        value_norm_sq.stride(2),
        centroid_value_sq.stride(0),
        centroid_value_sq.stride(1),
        centroid_value_sq.stride(2),
        BLOCK_S=64,
        num_warps=4,
    )

    efficiency_scores = torch.empty((B, H, qc_num, kc_num), device=device, dtype=torch.float32)
    grid = ((qc_num + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q, B * H)

    _error_estimation_kernel[grid](
        query_centroids,
        permuted_K,
        kc_offsets,
        qc_sz,
        kc_sz,
        efficiency_scores,
        value_norm_sq,
        centroid_value_dot,
        c_logits,
        centroid_value_sq,
        qc_num,
        kc_num,
        scale,
        B,
        H,
        gamma,
        eps,
        query_centroids.stride(0),
        query_centroids.stride(1),
        query_centroids.stride(2),
        query_centroids.stride(3),
        permuted_K.stride(0),
        permuted_K.stride(1),
        permuted_K.stride(2),
        permuted_K.stride(3),
        qc_sz.stride(0),
        qc_sz.stride(1),
        qc_sz.stride(2),
        kc_sz.stride(0),
        kc_sz.stride(1),
        kc_sz.stride(2),
        efficiency_scores.stride(0),
        efficiency_scores.stride(1),
        efficiency_scores.stride(2),
        efficiency_scores.stride(3),
        value_norm_sq.stride(0),
        value_norm_sq.stride(1),
        value_norm_sq.stride(2),
        c_logits.stride(0),
        c_logits.stride(1),
        c_logits.stride(2),
        c_logits.stride(3),
        centroid_value_sq.stride(0),
        centroid_value_sq.stride(1),
        centroid_value_sq.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        D=D,
        num_warps=4,
    )

    return efficiency_scores


@torch.inference_mode()
def identify_dynamic_map_estimated(
    q_perm,
    k_perm,
    v_perm,
    qc_sz,
    kc_sz,
    qcentroids,
    kcentroids,
    vcentroids,
    top_p=0.9,
    gamma=1.0,
    min_kc_ratio=0.0,
):
    """Estimate the dynamic block map under a budget derived from centroid attention.

    The selection happens in three stages:
    1. Estimate an efficiency score for every `(query cluster, key cluster)` pair.
    2. Derive a per-query-cluster area budget from centroid attention probabilities.
    3. Keep the highest-efficiency blocks that fit within the remaining budget after
       reserving the mandatory minimum key-cluster coverage.

    All tensors use the [B, H, ...] layout; returns a [B, H, qc, kc] bool map.
    """

    B, H, _, D = q_perm.shape
    device = q_perm.device
    scale = D ** -0.5
    eps = 1e-8

    qc_num = qcentroids.shape[2]
    kc_num = kcentroids.shape[2]

    # Centroid logits are reused both for error estimation and for budget construction.
    centroid_logits = torch.einsum("bhqd,bhkd->bhqk", qcentroids, kcentroids) * scale

    efficiency_scores = error_estimation_triton(
        qcentroids,
        kcentroids,
        vcentroids,
        k_perm,
        v_perm,
        qc_sz,
        kc_sz,
        centroid_logits,
        gamma,
        eps,
    )

    block_area = (qc_sz.unsqueeze(-1) * kc_sz.unsqueeze(-2)).to(torch.float32)

    # Approximate the reference top-p budget with centroid attention weighted by key-cluster size.
    key_cluster_log_sizes = torch.log(kc_sz.to(qcentroids.dtype) + eps)
    budget_logits = centroid_logits + key_cluster_log_sizes.unsqueeze(2)

    budget_probs = torch.softmax(budget_logits, dim=-1)
    sorted_budget_probs, budget_sorted_indices = torch.sort(budget_probs, dim=-1, descending=True)
    cumulative_budget_probs = torch.cumsum(sorted_budget_probs, dim=-1)
    budget_keep_mask = cumulative_budget_probs <= top_p
    budget_keep_mask = F.pad(budget_keep_mask[..., :-1], (1, 0), value=True)

    preserve_len = 0
    if min_kc_ratio > 0:
        preserve_len = max(1, int(min_kc_ratio * kc_num))
        budget_preserve_len = max(1, int(min_kc_ratio * kc_num * 2))
        budget_keep_mask[..., :budget_preserve_len] = True

    sorted_block_area = torch.gather(block_area, -1, budget_sorted_indices)
    budget_per_qc = (budget_keep_mask.to(block_area.dtype) * sorted_block_area).sum(dim=-1, keepdim=True)

    # Reserve a minimum number of key clusters before ranking the remaining candidates.
    guaranteed_mask = torch.zeros((B, H, qc_num, kc_num), dtype=torch.bool, device=device)
    if preserve_len > 0:
        guaranteed_mask.scatter_(-1, budget_sorted_indices[..., :preserve_len], True)

    guaranteed_cost = (guaranteed_mask.to(block_area.dtype) * block_area).sum(dim=-1, keepdim=True)
    remaining_budget = (budget_per_qc - guaranteed_cost).clamp_(min=0.0)

    efficiency_scores.masked_fill_(guaranteed_mask, -1e10)

    _, sorted_efficiency_indices = torch.sort(efficiency_scores, dim=-1, descending=True)
    sorted_candidate_area = torch.gather(block_area, -1, sorted_efficiency_indices)
    cumulative_candidate_cost = torch.cumsum(sorted_candidate_area, dim=-1)

    dynamic_keep_mask = (cumulative_candidate_cost - sorted_candidate_area) < remaining_budget
    dynamic_mask = torch.zeros_like(guaranteed_mask)
    dynamic_mask.scatter_(-1, sorted_efficiency_indices, dynamic_keep_mask)

    return guaranteed_mask | dynamic_mask
