from __future__ import annotations

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from .._schedule import resolve_first_layers, resolve_first_steps
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as svoo_config
from .ops import svoo_attention


class SVOOMethod(SparseMethod):
    """SVOO: QK co-clustering + dynamic block-sparse attention.

    Clusters Q in K-centroid profile space, then computes attention only
    between active cluster pairs.
    """

    CONFIG_DEFAULTS = svoo_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = svoo_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        normalized_config = self.normalize_config(config)
        model_key = None
        if model_info.model_type == "wan" and len(getattr(model_info, "transformers", [])) > 1:
            model_key = "wan22-t2v-a14b"
        self.config = {
            **svoo_config.default_config(model_family=model_info.model_type, model_key=model_key),
            **normalized_config,
        }
        self.model_info = model_info

        if self.config["implementation"] != "native":
            raise NotImplementedError(
                "SVOO implementation='upstream' is not a SparseVideo-owned runtime path. "
                "Port the needed code into src/sparsevideo before enabling it."
            )
        if self.config["sparse_backend"] not in ("flashinfer", "triton"):
            raise ValueError("svoo sparse_backend must be 'flashinfer' or 'triton'")

        unsupported = [
            name
            for name, default in svoo_config.UNPORTED_OPTION_DEFAULTS.items()
            if self.config[name] != default
        ]
        if unsupported:
            raise NotImplementedError(
                "These upstream SVOO options are recognized but not ported in "
                f"SparseVideo yet: {unsupported}"
            )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svoo not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = resolve_first_layers(cfg["first_layers_fp"], total_layers)
        first_step_count = resolve_first_steps(cfg["first_times_fp"], cfg["num_inference_steps"])

        state = {
            "centroids_init": False,
            "prev_q_centroids": None,
            "prev_k_centroids": None,
            "prev_q_profile_centroids": None,
            "cached_clustering": None,
            "sparsity_lookup": None,
            "sparsity_lookup_path": None,
        }

        def attn_fn(query, key, value, attention_mask, **kwargs):
            full_attention = (
                layer_idx < first_layer_count
                or step_tracker.step <= first_step_count
            )
            if full_attention:
                if cfg["zero_step_kmeans_init"] and query.is_cuda and attention_mask is None:
                    svoo_attention(
                        query, key, value, cfg, state,
                        current_step=step_tracker.step,
                        layer_idx=layer_idx,
                        initialize_only=True,
                        text_len=kwargs.get("text_len", 0),
                    )
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            if not query.is_cuda or attention_mask is not None:
                raise RuntimeError("svoo sparse path requires CUDA self-attention without an attention mask")
            return svoo_attention(
                query, key, value, cfg, state,
                current_step=step_tracker.step,
                layer_idx=layer_idx,
                text_len=kwargs.get("text_len", 0),
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )
