from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class AdaClusterMethod(SparseMethod):
    """AdaCluster: Triton k-means clustering + block-sparse attention.

    Uses upstream topk_num/q_kernel_num/kv_kernel_num naming, including the
    Hunyuan late-layer cluster-count overrides.

    Port of: training_free/Adacluster/triton_kernel/fast_kmeans.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"adacluster not yet supported for {self.model_info.model_type}")

        cfg = self.config
        state = {
            "centroids_init": False,
            "prev_q_centroids": None,
            "prev_k_centroids": None,
        }
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            if not query.is_cuda or attention_mask is not None:
                raise RuntimeError("adacluster sparse path requires CUDA self-attention without an attention mask")
            topk_num = cfg["topk_num"]
            q_kernel_num = cfg["q_kernel_num"]
            kv_kernel_num = cfg["kv_kernel_num"]
            if model_type == "hunyuan_video" and layer_idx >= cfg["late_layer_start"]:
                topk_num = cfg["late_topk_num"]
                q_kernel_num = cfg["late_q_kernel_num"]
                kv_kernel_num = cfg["late_kv_kernel_num"]
            return _adacluster_attention(
                query, key, value,
                topk_num=topk_num,
                q_kernel_num=q_kernel_num,
                kv_kernel_num=kv_kernel_num,
                kmeans_iter_init=cfg["kmeans_iter_init"],
                kmeans_iter_step=cfg["kmeans_iter_step"],
                state=state,
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _adacluster_attention(query, key, value, topk_num, q_kernel_num, kv_kernel_num,
                          kmeans_iter_init, kmeans_iter_step, state):
    """AdaCluster: Triton k-means + block-sparse attention.

    query/key/value: [B, N, H, D]
    """
    from ...kernels.kmeans import triton_kmeans
    from ...kernels.block_sparse_attn import block_sparse_attention

    B, N, H, D = query.shape
    scale = D ** -0.5

    q_flat = query.permute(0, 2, 1, 3).reshape(B * H, N, D)
    k_flat = key.permute(0, 2, 1, 3).reshape(B * H, N, D)
    v_flat = value.permute(0, 2, 1, 3).reshape(B * H, N, D)

    nqc = min(q_kernel_num, N)
    nkc = min(kv_kernel_num, N)
    kmeans_iters = kmeans_iter_step if state["centroids_init"] else kmeans_iter_init

    q_labels, q_centroids, q_sizes = triton_kmeans(
        q_flat, nqc, kmeans_iters, init_centroids=state.get("prev_q_centroids"),
    )
    k_labels, k_centroids, k_sizes = triton_kmeans(
        k_flat, nkc, kmeans_iters, init_centroids=state.get("prev_k_centroids"),
    )
    state["centroids_init"] = True
    state["prev_q_centroids"] = q_centroids.detach()
    state["prev_k_centroids"] = k_centroids.detach()

    # Cluster-level attention → top-k → dynamic map
    cluster_scores = torch.matmul(q_centroids.float(), k_centroids.float().transpose(-2, -1)) * scale
    cluster_scores = cluster_scores + k_sizes.float().clamp(min=1).unsqueeze(1).log()
    cluster_attn = F.softmax(cluster_scores, dim=-1)

    k_keep = min(nkc, max(1, int(topk_num)))
    _, topk_idx = torch.topk(cluster_attn, k=k_keep, dim=-1)
    dynamic_map = torch.zeros(B * H, nqc, nkc, dtype=torch.bool, device=query.device)
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
