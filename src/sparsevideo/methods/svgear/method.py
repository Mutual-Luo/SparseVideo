from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._schedule import (
    configured_dense_warmup_layer_count,
    configured_dense_warmup_requires_dense,
    runtime_or_config_num_inference_steps,
    scheduler_timestep_from_tracker,
)
from ...kernels.dynamic_map import identify_dynamic_map_estimated
from ...kernels.scatter_mean import scatter_mean_fused
from ...kernels.flashinfer_block_sparse import dynamic_block_sparse_prune_fwd_flashinfer
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor

# EAR reuses SVG2/SAP infrastructure verbatim: k-means, token permutation, dense
# fallback, the Hunyuan text-tail append, and the per-call runtime state.
from ..svg2.method import (
    _new_runtime_state,
    _state_for_cache_suffix,
    _svg2_attention,
    _svg2_dense_attention,
    _svg2_dense_backend_name,
    _svg2_permute_by_labels,
    _svg2_permute_by_sorted_indices,
    _svg2_inverse_permutation,
    _svg2_append_hunyuan_text_clusters,
    _resolve_svg2_prompt_length,
)
from . import config as method_config

_TEXT_TAIL_MODELS = ("hunyuan_video", "cogvideox", "mochi", "easyanimate")


class SVGEARMethod(SparseMethod):
    """SVG-EAR: Error-Aware Reduction sparse attention.

    Builds on SVG2/SAP (k-means clustering + variable block-sparse attention) but
    selects blocks by their estimated output error and recovers pruned blocks with
    a centroid-attention approximation instead of dropping them.
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in (
            "wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate",
        ):
            raise NotImplementedError(f"svg-ear not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = configured_dense_warmup_layer_count(cfg, total_layers)
        state = _new_runtime_state()

        def attn_fn(query, key, value, attention_mask, **kwargs):
            scheduler_timestep = scheduler_timestep_from_tracker(step_tracker, kwargs)
            runtime_state = _state_for_cache_suffix(state, kwargs.get("cache_key_suffix"))
            prompt_length = kwargs.get("prompt_length")
            if prompt_length is None:
                prompt_length = cfg.get("prompt_length")
            full_attention = (
                layer_idx < first_layer_count
                or configured_dense_warmup_requires_dense(
                    cfg,
                    runtime_or_config_num_inference_steps(step_tracker, cfg),
                    step_tracker.step,
                    scheduler_timestep,
                    notifier=self.warmup_notifier,
                )
            )
            if full_attention:
                if (
                    cfg["zero_step_kmeans_init"]
                    and query.is_cuda
                    and (
                        attention_mask is None
                        or self.model_info.model_type in _TEXT_TAIL_MODELS
                    )
                ):
                    # Seed k-means centroids during warmup, reusing the SVG2 path
                    # (EAR shares the same runtime state shape).
                    _svg2_attention(
                        query, key, value,
                        top_p_kmeans=cfg["top_p_kmeans"],
                        min_kc_ratio=cfg["min_kc_ratio"],
                        num_q_centroids=cfg["num_q_centroids"],
                        num_k_centroids=cfg["num_k_centroids"],
                        kmeans_iter_init=cfg["kmeans_iter_init"],
                        kmeans_iter_step=cfg["kmeans_iter_step"],
                        state=runtime_state,
                        initialize_only=True,
                        model_type=self.model_info.model_type,
                        text_len=kwargs.get("text_len", 0),
                        prompt_length=prompt_length,
                        context_length=cfg.get("context_length"),
                    )
                    self.record_runtime_dispatch(
                        "initialize_only",
                        backend="triton_kmeans",
                        layer_idx=layer_idx,
                        step=getattr(step_tracker, "step", None),
                    )
                out = _svg2_dense_attention(
                    query, key, value, attention_mask,
                    model_type=self.model_info.model_type,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_svg2_dense_backend_name(query, attention_mask, self.model_info.model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("svg-ear sparse path requires CUDA self-attention without an attention mask")
            if (
                attention_mask is not None
                and self.model_info.model_type not in _TEXT_TAIL_MODELS
            ):
                raise RuntimeError("svg-ear sparse path requires CUDA self-attention without an attention mask")
            out = _svgear_attention(
                query, key, value,
                top_p_kmeans=cfg["top_p_kmeans"],
                min_kc_ratio=cfg["min_kc_ratio"],
                gamma=cfg["gamma"],
                num_q_centroids=cfg["num_q_centroids"],
                num_k_centroids=cfg["num_k_centroids"],
                kmeans_iter_init=cfg["kmeans_iter_init"],
                kmeans_iter_step=cfg["kmeans_iter_step"],
                state=runtime_state,
                model_type=self.model_info.model_type,
                text_len=kwargs.get("text_len", 0),
                prompt_length=prompt_length,
                context_length=cfg.get("context_length"),
            )
            self.record_runtime_dispatch(
                "sparse",
                backend="flashinfer_ear",
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "cogvideox":
            return SparseCogVideoXAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "ltx_video":
            return SparseLTXVideoAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "allegro":
            return SparseAllegroAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "mochi":
            return SparseMochiAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "easyanimate":
            return SparseEasyAnimateAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        return SparseHunyuanVideoAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)


def _svgear_attention(query, key, value, top_p_kmeans, min_kc_ratio, gamma,
                      num_q_centroids, num_k_centroids, kmeans_iter_init,
                      kmeans_iter_step, state, model_type="wan",
                      text_len=0, prompt_length=None, context_length=None):
    """SVG-EAR: k-means clustering + error-aware block selection + pruned block-sparse
    attention with centroid recovery.

    query: [B, Q, H, D], key/value: [B, KV, H, D]. Mirrors svg2's `_svg2_attention`
    but swaps the block selection (error estimation) and the block-sparse kernel
    (centroid-recovering prune).
    """
    from ..svg2.kmeans import triton_kmeans

    B, q_len, H, D = query.shape
    kv_len = key.shape[1]
    if value.shape[1] != kv_len:
        raise RuntimeError(
            "svg-ear sparse path requires key/value lengths to match; "
            f"got key_len={kv_len}, value_len={value.shape[1]}"
        )

    q_full = query.permute(0, 2, 1, 3).contiguous().reshape(B * H, q_len, D)
    k_full = key.permute(0, 2, 1, 3).contiguous().reshape(B * H, kv_len, D)
    v_full = value.permute(0, 2, 1, 3).contiguous().reshape(B * H, kv_len, D)

    text_len = int(text_len or 0)
    if model_type in _TEXT_TAIL_MODELS and text_len > 0:
        if q_len != kv_len:
            raise RuntimeError(
                "svg-ear text-tail sparse path requires matching query/key lengths; "
                f"got query_len={q_len}, key_len={kv_len}"
            )
        if context_length is not None and int(context_length) != text_len:
            raise RuntimeError(
                "svg-ear context_length must match the text token tail length "
                f"seen by the processor; got context_length={int(context_length)}, text_len={text_len}"
            )
        q_video_len = q_len - text_len
        kv_video_len = kv_len - text_len
        if q_video_len <= 0 or kv_video_len <= 0:
            raise RuntimeError("svg-ear hunyuan sparse path could not find video tokens")
        prompt_length = _resolve_svg2_prompt_length(prompt_length, text_len)
        q_flat = q_full[:, :q_video_len, :].contiguous()
        k_flat = k_full[:, :kv_video_len, :].contiguous()
        v_flat = v_full[:, :kv_video_len, :].contiguous()
    else:
        q_video_len = q_len
        kv_video_len = kv_len
        prompt_length = 0
        q_flat = q_full
        k_flat = k_full
        v_flat = v_full

    nqc = min(num_q_centroids, q_video_len)
    nkc = min(num_k_centroids, kv_video_len)

    kmeans_iters = kmeans_iter_step if state["centroids_init"] else kmeans_iter_init
    q_labels, q_centroids, q_sizes = triton_kmeans(
        q_flat, nqc, kmeans_iters,
        init_centroids=state.get("prev_q_centroids"), final_reassign=False,
    )
    k_labels, k_centroids, k_sizes = triton_kmeans(
        k_flat, nkc, kmeans_iters,
        init_centroids=state.get("prev_k_centroids"), final_reassign=False,
    )
    state["centroids_init"] = True
    state["prev_q_centroids"] = q_centroids.detach()
    state["prev_k_centroids"] = k_centroids.detach()

    # Sort tokens by cluster (package-owned Triton permutation kernels).
    q_sorted, q_sorted_idx = _svg2_permute_by_labels(q_flat, q_labels)
    k_sorted, k_sorted_idx = _svg2_permute_by_labels(k_flat, k_labels)
    v_sorted, _ = _svg2_permute_by_sorted_indices(v_flat, k_sorted_idx)

    # Value-cluster centroids: mean of V within each (cluster-contiguous) key cluster.
    v_centroids = scatter_mean_fused(
        v_sorted.unsqueeze(1), k_sizes.unsqueeze(1),
    ).squeeze(1)

    # Error-aware block selection on the video clusters.
    dynamic_map = identify_dynamic_map_estimated(
        q_sorted.unsqueeze(1),
        k_sorted.unsqueeze(1),
        v_sorted.unsqueeze(1),
        q_sizes.unsqueeze(1),
        k_sizes.unsqueeze(1),
        q_centroids.unsqueeze(1),
        k_centroids.unsqueeze(1),
        v_centroids.unsqueeze(1),
        top_p=top_p_kmeans,
        gamma=gamma,
        min_kc_ratio=min_kc_ratio,
    ).squeeze(1)

    prune_mask = None
    if model_type in _TEXT_TAIL_MODELS and text_len > 0:
        q_sorted, k_sorted, v_sorted, dynamic_map, q_sizes, k_sizes, q_sorted_idx = _svg2_append_hunyuan_text_clusters(
            q_sorted, k_sorted, v_sorted,
            q_full, k_full, v_full,
            dynamic_map, q_sizes, k_sizes, q_sorted_idx,
            video_len=q_video_len, text_len=text_len, prompt_length=prompt_length,
        )
        # Pad centroids for the two appended text clusters (never read because the
        # text columns are excluded from the centroid-approximation step below).
        k_centroids = F.pad(k_centroids, (0, 0, 0, 2))
        v_centroids = F.pad(v_centroids, (0, 0, 0, 2))
        # FlashInfer handles the text columns exactly (per svg2 selection); force the
        # centroid step to skip them so padding-text exclusion is preserved.
        prune_mask = dynamic_map.clone()
        prune_mask[..., -2:] = True

    q_sizes_i32 = q_sizes.to(torch.int32)
    k_sizes_i32 = k_sizes.to(torch.int32)
    prune_mask_4d = None if prune_mask is None else prune_mask.unsqueeze(1)
    out_sorted = dynamic_block_sparse_prune_fwd_flashinfer(
        q_sorted.unsqueeze(1),
        k_sorted.unsqueeze(1),
        v_sorted.unsqueeze(1),
        k_centroids.unsqueeze(1),
        v_centroids.unsqueeze(1),
        dynamic_map.unsqueeze(1),
        q_sizes_i32.unsqueeze(1),
        k_sizes_i32.unsqueeze(1),
        prune_mask=prune_mask_4d,
    ).squeeze(1)

    out_flat = _svg2_inverse_permutation(out_sorted, q_sorted_idx)
    return out_flat.reshape(B, H, q_len, D).permute(0, 2, 1, 3)
