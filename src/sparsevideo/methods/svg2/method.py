from __future__ import annotations

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from .._schedule import resolve_first_layers, resolve_first_steps
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class SVG2Method(SparseMethod):
    """SVG2: k-means clustering + block-sparse attention.

    Port of the second Sparse-VideoGen method.
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svg2 not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = resolve_first_layers(cfg["first_layers_fp"], total_layers)
        first_step_count = resolve_first_steps(cfg["first_times_fp"], cfg["num_inference_steps"])
        state = {
            "centroids_init": False,
            "prev_q_centroids": None,
            "prev_k_centroids": None,
        }

        def attn_fn(query, key, value, attention_mask, **kwargs):
            full_attention = (
                layer_idx < first_layer_count
                or step_tracker.step <= first_step_count
            )
            if full_attention:
                if cfg["zero_step_kmeans_init"] and query.is_cuda and attention_mask is None:
                    _svg2_attention(
                        query, key, value,
                        top_p_kmeans=cfg["top_p_kmeans"],
                        min_kc_ratio=cfg["min_kc_ratio"],
                        num_q_centroids=cfg["num_q_centroids"],
                        num_k_centroids=cfg["num_k_centroids"],
                        kmeans_iter_init=cfg["kmeans_iter_init"],
                        kmeans_iter_step=cfg["kmeans_iter_step"],
                        state=state,
                        initialize_only=True,
                    )
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            if not query.is_cuda or attention_mask is not None:
                raise RuntimeError("svg2 sparse path requires CUDA self-attention without an attention mask")
            return _svg2_attention(
                query, key, value,
                top_p_kmeans=cfg["top_p_kmeans"],
                min_kc_ratio=cfg["min_kc_ratio"],
                num_q_centroids=cfg["num_q_centroids"],
                num_k_centroids=cfg["num_k_centroids"],
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


def _svg2_attention(query, key, value, top_p_kmeans, min_kc_ratio,
                    num_q_centroids, num_k_centroids, kmeans_iter_init,
                    kmeans_iter_step, state, initialize_only=False):
    """SVG2: k-means clustering + block-sparse attention.

    query/key/value: [B, N, H, D]
    Primary backend: flashinfer VariableBlockSparseAttentionWrapper.
    Fallback: Triton block_sparse_attention.
    """
    from ...kernels.kmeans import triton_kmeans
    from ...kernels.block_sparse_attn import block_sparse_attention
    from ...kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    B, N, H, D = query.shape
    scale = D ** -0.5

    q_flat = query.permute(0, 2, 1, 3).reshape(B * H, N, D)
    k_flat = key.permute(0, 2, 1, 3).reshape(B * H, N, D)
    v_flat = value.permute(0, 2, 1, 3).reshape(B * H, N, D)

    nqc = min(num_q_centroids, N)
    nkc = min(num_k_centroids, N)

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

    if initialize_only:
        return None

    # Cluster-level weighted softmax -> top-p dynamic map.
    cluster_scores = torch.matmul(q_centroids, k_centroids.transpose(-2, -1)).float() * scale
    max_score = cluster_scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(cluster_scores - max_score)
    weighted_exp = exp_scores * k_sizes.float().unsqueeze(1)
    cluster_attn = weighted_exp / weighted_exp.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    sorted_attn, sorted_idx = cluster_attn.sort(dim=-1, descending=True)
    cumsum_attn = sorted_attn.cumsum(dim=-1)
    k_keep = (cumsum_attn < float(top_p_kmeans)).sum(dim=-1) + 1
    k_keep = k_keep.clamp(max=nkc, min=max(1, int(nkc * float(min_kc_ratio))))
    dynamic_map = torch.zeros(B * H, nqc, nkc, dtype=torch.bool, device=query.device)
    for qi in range(nqc):
        max_keep = k_keep[:, qi].max().item()
        mask = torch.arange(nkc, device=query.device).unsqueeze(0) < k_keep[:, qi].unsqueeze(1)
        idx = sorted_idx[:, qi, :max_keep]
        dynamic_map[:, qi, :].scatter_(1, idx, mask[:, :max_keep])

    # Sort tokens by cluster
    q_sorted_idx = q_labels.long().argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)

    q_sorted = torch.gather(q_flat, 1, q_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, k_sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    # Block-sparse attention — flashinfer primary, Triton fallback
    q_sizes_i32 = q_sizes.to(torch.int32)
    k_sizes_i32 = k_sizes.to(torch.int32)
    if HAS_FLASHINFER:
        out_sorted = variable_block_sparse_attn(
            q_sorted, k_sorted, v_sorted,
            dynamic_map, q_sizes_i32, k_sizes_i32,
        )
    else:
        out_sorted = block_sparse_attention(
            q_sorted, k_sorted, v_sorted,
            q_sizes, k_sizes, dynamic_map, scale,
        )

    # Unsort
    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_q_idx.unsqueeze(-1).expand(-1, -1, D))

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)
