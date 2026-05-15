from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class AdaClusterMethod(SparseMethod):
    """AdaCluster: Triton k-means clustering + block-sparse attention.

    Uses configurable distance metric for clustering, then computes
    sparse attention between active cluster pairs.

    Port of: training_free/Adacluster/triton_kernel/fast_kmeans.py
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "num_clusters": 200,
        "distance": "euclidean",
        "kmeans_iters": 10,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"adacluster not yet supported for {self.model_info.model_type}")

        cfg = self.config
        skip_steps = cfg["skip_first_steps"]
        skip_layers = cfg["skip_first_layers"]

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
            try:
                return _adacluster_attention(
                    query, key, value,
                    budget=cfg["budget"],
                    num_clusters=cfg["num_clusters"],
                    distance=cfg["distance"],
                    kmeans_iters=cfg["kmeans_iters"],
                )
            except Exception:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _adacluster_attention(query, key, value, budget, num_clusters, distance, kmeans_iters):
    """AdaCluster: Triton k-means + block-sparse attention.

    query/key/value: [B, N, H, D]
    """
    if not query.is_cuda:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    from ..kernels.kmeans import triton_kmeans
    from ..kernels.block_sparse_attn import block_sparse_attention

    B, N, H, D = query.shape
    scale = D ** -0.5

    q_flat = query.permute(0, 2, 1, 3).reshape(B * H, N, D)
    k_flat = key.permute(0, 2, 1, 3).reshape(B * H, N, D)
    v_flat = value.permute(0, 2, 1, 3).reshape(B * H, N, D)

    nc = min(num_clusters, N)

    # Cluster K tokens with Triton k-means (Euclidean)
    k_labels, k_centroids, k_sizes = triton_kmeans(k_flat, nc, kmeans_iters)

    # Assign Q tokens to nearest K-centroid using configured distance
    if distance == "cosine":
        q_norm = F.normalize(q_flat, dim=-1)
        kc_norm = F.normalize(k_centroids, dim=-1)
        sim = torch.matmul(q_norm, kc_norm.transpose(-2, -1))
        q_labels = sim.argmax(dim=-1).int()
    elif distance == "dot":
        sim = torch.matmul(q_flat, k_centroids.transpose(-2, -1))
        q_labels = sim.argmax(dim=-1).int()
    else:
        dists = torch.cdist(q_flat.float(), k_centroids.float())
        q_labels = dists.argmin(dim=-1).int()

    # Compute Q centroids for dynamic map
    q_sizes = torch.zeros(B * H, nc, dtype=torch.long, device=query.device)
    q_sizes.scatter_add_(1, q_labels.long(), torch.ones(B * H, N, dtype=torch.long, device=query.device))

    q_centroids = torch.zeros(B * H, nc, D, device=query.device, dtype=torch.float32)
    q_centroids.scatter_add_(1, q_labels.long().unsqueeze(-1).expand(-1, -1, D), q_flat.float())
    safe_qsizes = q_sizes.clamp(min=1).unsqueeze(-1).float()
    q_centroids = q_centroids / safe_qsizes

    # Cluster-level attention → top-k → dynamic map
    cluster_scores = torch.matmul(q_centroids, k_centroids.float().transpose(-2, -1)) * scale
    cluster_attn = F.softmax(cluster_scores, dim=-1)

    k_keep = max(1, int(nc * budget))
    _, topk_idx = torch.topk(cluster_attn, k=k_keep, dim=-1)
    dynamic_map = torch.zeros(B * H, nc, nc, dtype=torch.bool, device=query.device)
    dynamic_map.scatter_(dim=-1, index=topk_idx, value=True)

    # Sort by cluster and compute block-sparse attention
    q_sorted_idx = q_labels.long().argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)

    q_sorted = torch.gather(q_flat, 1, q_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    out_sorted = block_sparse_attention(
        q_sorted, k_sorted, v_sorted,
        q_sizes, k_sizes, dynamic_map, scale,
    )

    # Unsort
    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_q_idx.unsqueeze(-1).expand(-1, -1, D))

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)
