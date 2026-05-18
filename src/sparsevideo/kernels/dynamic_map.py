from __future__ import annotations

import torch
import torch.nn.functional as F


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
