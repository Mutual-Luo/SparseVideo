from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ..kernels.kmeans import triton_kmeans
from ..kernels.block_sparse_attn import block_sparse_attention


class SVG2Method(SparseMethod):
    """SVG2: k-means clustering + block-sparse attention.

    Approximate port of the second Sparse-VideoGen method.
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "num_q_centroids": 50,
        "num_k_centroids": 200,
        "kmeans_iters": 10,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svg2 not yet supported for {self.model_info.model_type}")

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
            return _svg2_attention(
                query, key, value,
                budget=cfg["budget"],
                num_q_centroids=cfg["num_q_centroids"],
                num_k_centroids=cfg["num_k_centroids"],
                kmeans_iters=cfg["kmeans_iters"],
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _svg2_attention(query, key, value, budget, num_q_centroids, num_k_centroids, kmeans_iters):
    """SVG2: Triton k-means clustering + block-sparse attention.

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

    q_labels, q_centroids, q_sizes = triton_kmeans(q_flat, nqc, kmeans_iters)
    k_labels, k_centroids, k_sizes = triton_kmeans(k_flat, nkc, kmeans_iters)

    # Cluster-level attention → top-k → dynamic map
    cluster_scores = torch.matmul(q_centroids, k_centroids.transpose(-2, -1)) * scale
    cluster_attn = F.softmax(cluster_scores, dim=-1)

    k_keep = max(1, int(nkc * budget))
    _, topk_k = torch.topk(cluster_attn, k=k_keep, dim=-1)
    dynamic_map = torch.zeros(B * H, nqc, nkc, dtype=torch.bool, device=query.device)
    dynamic_map.scatter_(dim=-1, index=topk_k, value=True)

    # Sort tokens by cluster
    q_sorted_idx = q_labels.long().argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)

    q_sorted = torch.gather(q_flat, 1, q_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    # Block-sparse attention via Triton
    out_sorted = block_sparse_attention(
        q_sorted, k_sorted, v_sorted,
        q_sizes, k_sizes, dynamic_map, scale,
    )

    # Unsort
    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_q_idx.unsqueeze(-1).expand(-1, -1, D))

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)
