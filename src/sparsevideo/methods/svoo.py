from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ..kernels.kmeans import triton_kmeans
from ..kernels.block_sparse_attn import block_sparse_attention
from ..kernels.co_cluster import profile_norm, co_cluster_assign


class SVOOMethod(SparseMethod):
    """SVOO: QK co-clustering + dynamic block-sparse attention.

    Clusters Q in K-centroid "profile space" (what K-clusters each Q attends to),
    then computes attention only between active cluster pairs.

    Port of: training_free/SVOO/svoo/co_clustering.py
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "num_q_centroids": 50,
        "num_k_centroids": 200,
        "kmeans_iters": 10,
        "min_kc_ratio": 0.0,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svoo not yet supported for {self.model_info.model_type}")

        cfg = self.config
        skip_steps = cfg["skip_first_steps"]
        skip_layers = cfg["skip_first_layers"]

        state = {"prev_k_centroids": None, "prev_q_profile_centroids": None}

        def attn_fn(query, key, value, attention_mask, **kwargs):
            use_sparse = (
                layer_idx >= skip_layers
                and step_tracker.step > skip_steps
            )
            if not use_sparse:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            if not query.is_cuda:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            return _svoo_attention(
                query, key, value,
                budget=cfg["budget"],
                num_q_centroids=cfg["num_q_centroids"],
                num_k_centroids=cfg["num_k_centroids"],
                kmeans_iters=cfg["kmeans_iters"],
                min_kc_ratio=cfg["min_kc_ratio"],
                state=state,
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _svoo_attention(query, key, value, budget, num_q_centroids, num_k_centroids,
                    kmeans_iters, min_kc_ratio, state):
    """SVOO: Co-clustering + dynamic block-sparse attention.

    query/key/value: [B, N, H, D]
    """
    if not query.is_cuda:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    B, N, H, D = query.shape
    scale = D ** -0.5

    q_flat = query.permute(0, 2, 1, 3).reshape(B * H, N, D)
    k_flat = key.permute(0, 2, 1, 3).reshape(B * H, N, D)
    v_flat = value.permute(0, 2, 1, 3).reshape(B * H, N, D)

    nqc = min(num_q_centroids, N)
    nkc = min(num_k_centroids, N)

    # Step 1: K-means on K tokens (warm-start from previous step if available)
    k_labels, k_centroids, k_sizes = triton_kmeans(
        k_flat, nkc, kmeans_iters,
        init_centroids=state.get("prev_k_centroids"),
    )
    state["prev_k_centroids"] = k_centroids.detach()

    # Step 2: Co-cluster Q in K-centroid profile space via Triton
    # Profile = softmax(Q @ K_centroids^T) — which K-clusters each Q attends to
    # Two-pass Triton approach avoids materializing [B*H, N, nkc]
    norms = profile_norm(q_flat, k_centroids)

    # Compute or warm-start Q profile centroids
    prev_pc = state.get("prev_q_profile_centroids")
    if prev_pc is not None and prev_pc.shape == (B * H, nqc, nkc):
        q_profile_centroids = prev_pc.float()
    else:
        # Initialize profile centroids by sampling Q profiles
        sample_idx = torch.randint(0, N, (B * H, nqc), device=q_flat.device)
        sample_q = torch.gather(q_flat, 1, sample_idx.unsqueeze(-1).expand(-1, -1, D))
        sample_profiles = torch.matmul(sample_q, k_centroids.transpose(-2, -1)) * scale
        sample_profiles = F.softmax(sample_profiles, dim=-1)
        profile_norms = sample_profiles.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_profile_centroids = (sample_profiles / profile_norms).float()

    # Fused co-cluster assignment
    q_labels = co_cluster_assign(q_flat, k_centroids, q_profile_centroids, norms)

    # Update profile centroids for next step
    new_pc = torch.zeros(B * H, nqc, nkc, device=q_flat.device, dtype=torch.float32)
    pc_counts = torch.zeros(B * H, nqc, device=q_flat.device, dtype=torch.float32)
    # Compute profiles for centroid update (small: [B*H, nqc, nkc])
    q_centroids_token = torch.zeros(B * H, nqc, D, device=q_flat.device, dtype=torch.float32)
    q_sizes = torch.zeros(B * H, nqc, dtype=torch.long, device=q_flat.device)
    q_sizes.scatter_add_(1, q_labels, torch.ones(B * H, N, dtype=torch.long, device=q_flat.device))
    q_centroids_token.scatter_add_(1, q_labels.unsqueeze(-1).expand(-1, -1, D), q_flat.float())
    safe_qsizes = q_sizes.clamp(min=1).unsqueeze(-1).float()
    q_centroids_token = q_centroids_token / safe_qsizes

    # Profile centroids = normalized softmax(q_centroid @ k_centroids^T)
    q_profiles = torch.matmul(q_centroids_token, k_centroids.float().transpose(-2, -1)) * scale
    q_profiles = F.softmax(q_profiles, dim=-1)
    profile_norms_new = q_profiles.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    state["prev_q_profile_centroids"] = (q_profiles / profile_norms_new).detach()

    # Step 3: Dynamic map — cumulative probability thresholding
    cluster_scores = torch.matmul(q_centroids_token, k_centroids.float().transpose(-2, -1)) * scale
    k_cluster_sizes_f = k_sizes.float().clamp(min=1)
    cluster_scores = cluster_scores + k_cluster_sizes_f.unsqueeze(1).log()
    cluster_attn = F.softmax(cluster_scores, dim=-1)

    sorted_attn, sorted_idx = cluster_attn.sort(dim=-1, descending=True)
    cumsum_attn = sorted_attn.cumsum(dim=-1)

    k_keep = (cumsum_attn < budget).sum(dim=-1) + 1
    k_keep = k_keep.clamp(max=nkc, min=max(1, int(nkc * min_kc_ratio)))

    dynamic_map = torch.zeros(B * H, nqc, nkc, dtype=torch.bool, device=q_flat.device)
    for qi in range(nqc):
        max_keep = k_keep[:, qi].max().item()
        mask = torch.arange(nkc, device=q_flat.device).unsqueeze(0) < k_keep[:, qi].unsqueeze(1)
        idx = sorted_idx[:, qi, :max_keep]
        row_mask = mask[:, :max_keep]
        dynamic_map[:, qi, :].scatter_(1, idx, row_mask)

    # Step 4: Sort + block-sparse attention + unsort
    q_sorted_idx = q_labels.argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)

    q_sorted = torch.gather(q_flat, 1, q_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    out_sorted = block_sparse_attention(
        q_sorted, k_sorted, v_sorted,
        q_sizes, k_sizes, dynamic_map, scale,
    )

    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_q_idx.unsqueeze(-1).expand(-1, -1, D))

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)
