from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._schedule import first_times_fp_requires_dense, resolve_first_layers, scheduler_timestep_from_tracker
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as svoo_config
from .ops import resolve_sparsity_csv_path, svoo_attention
from .sparsity import log_attention_sparsity, prepare_sparsity_output


class SVOOMethod(SparseMethod):
    """SVOO: QK co-clustering + dynamic block-sparse attention.

    Clusters Q in K-centroid profile space, then computes attention only
    between active cluster pairs.
    """

    CONFIG_DEFAULTS = svoo_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = svoo_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        normalized_config = self.normalize_config(config)
        model_key = getattr(model_info, "model_key", None)
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
        if model_info.model_type == "hunyuan_video" and self.config["sparse_backend"] != "flashinfer":
            raise ValueError(
                "svoo Hunyuan upstream path uses FlashInfer for varlen dense gates and sparse attention; "
                "sparse_backend='triton' is only available for the Wan fallback path."
            )
        if self.config.get("use_routing_transformer_strategy"):
            missing = [
                name for name in ("mq1", "mk1", "mq2", "mk2")
                if self.config.get(name) is None
            ]
            if missing:
                raise ValueError(
                    "svoo use_routing_transformer_strategy requires mq1, mk1, mq2, and mk2; "
                    f"missing {missing}"
                )
            for name in ("mq1", "mk1", "mq2", "mk2"):
                if int(self.config[name]) <= 0:
                    raise ValueError(f"svoo routing parameter {name} must be a positive integer")
        if self.config.get("use_dynamic_min_kc_ratio"):
            sparsity_csv_path = self.config.get("sparsity_csv_path")
            if not sparsity_csv_path:
                raise ValueError("svoo use_dynamic_min_kc_ratio requires sparsity_csv_path")
            resolved_sparsity_csv_path = resolve_sparsity_csv_path(sparsity_csv_path)
            if not resolved_sparsity_csv_path.exists():
                raise FileNotFoundError(
                    "svoo use_dynamic_min_kc_ratio requires an existing "
                    f"sparsity_csv_path: {resolved_sparsity_csv_path}"
                )
            self.config["sparsity_csv_path"] = str(resolved_sparsity_csv_path)

        if self.config.get("measure_attention_sparsity"):
            prepare_sparsity_output(self.config.get("sparsity_output_file"))

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svoo not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = resolve_first_layers(cfg["first_layers_fp"], total_layers)

        state = {
            "centroids_init": False,
            "prev_q_centroids": None,
            "prev_k_centroids": None,
            "prev_q_profile_centroids": None,
            "cached_clustering": None,
            "sparsity_lookup": None,
            "sparsity_lookup_path": None,
            "last_logged_sparsity_step": None,
        }

        def attn_fn(query, key, value, attention_mask, **kwargs):
            scheduler_timestep = scheduler_timestep_from_tracker(step_tracker, kwargs)
            prompt_length = kwargs.get("prompt_length")
            if prompt_length is None:
                prompt_length = cfg.get("prompt_length")
            text_len = kwargs.get("text_len", 0)
            full_attention = (
                layer_idx < first_layer_count
                or first_times_fp_requires_dense(
                    cfg["first_times_fp"],
                    cfg["num_inference_steps"],
                    step_tracker.step,
                    scheduler_timestep,
                )
            )
            log_attention_sparsity(
                query, key, cfg, state,
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            if full_attention:
                can_initialize = (
                    cfg["zero_step_kmeans_init"]
                    and query.is_cuda
                    and (attention_mask is None or self.model_info.model_type == "hunyuan_video")
                )
                if can_initialize:
                    svoo_attention(
                        query, key, value, cfg, state,
                        current_step=step_tracker.step,
                        layer_idx=layer_idx,
                        initialize_only=True,
                        text_len=text_len,
                        prompt_length=prompt_length,
                        model_type=self.model_info.model_type,
                        scheduler_timestep=scheduler_timestep,
                        total_layers=total_layers,
                    )
                    self.record_runtime_dispatch(
                        "initialize_only",
                        backend=f"svoo_{cfg['sparse_backend']}",
                        layer_idx=layer_idx,
                        step=getattr(step_tracker, "step", None),
                    )
                out = _svoo_dense_attention(
                    query, key, value, attention_mask,
                    model_type=self.model_info.model_type,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_svoo_dense_backend_name(query, attention_mask, self.model_info.model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("svoo sparse path requires CUDA")
            if attention_mask is not None and self.model_info.model_type != "hunyuan_video":
                raise RuntimeError("svoo sparse path only supports attention_mask for Hunyuan prompt padding")
            out = svoo_attention(
                query, key, value, cfg, state,
                current_step=step_tracker.step,
                layer_idx=layer_idx,
                text_len=text_len,
                prompt_length=prompt_length,
                model_type=self.model_info.model_type,
                scheduler_timestep=scheduler_timestep,
                total_layers=total_layers,
            )
            self.record_runtime_dispatch(
                "sparse",
                backend=f"svoo_{cfg['sparse_backend']}",
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
                use_fused_qk_norm=True,
                use_fused_rope=self.config.get("use_fused_rope", True),
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )

    def install_model_patches(self, model_info):
        if model_info.model_type == "hunyuan_video":
            from ...processors.hunyuan_sparse_forward import install_hunyuan_sparse_forward_patch

            return [install_hunyuan_sparse_forward_patch()]
        return super().install_model_patches(model_info)


def _svoo_dense_backend_name(query, attention_mask, model_type):
    if model_type == "hunyuan_video" and attention_mask is not None and query.is_cuda:
        return "svoo_flashinfer_varlen"
    if attention_mask is not None:
        return "diffusers_dispatch"
    return "torch_sdpa"


def _svoo_dense_attention(query, key, value, attention_mask, *, model_type):
    if model_type == "hunyuan_video" and attention_mask is not None and query.is_cuda:
        return _svoo_hunyuan_flashinfer_varlen(query, key, value, attention_mask)
    if attention_mask is not None:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        return dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        dropout_p=0.0, is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _svoo_hunyuan_flashinfer_varlen(query, key, value, attention_mask):
    if query.shape[0] != 1:
        raise RuntimeError("SVOO Hunyuan FlashInfer varlen path follows upstream batch size 1")

    batch, seq_len, heads, dim = query.shape
    valid_len = int(attention_mask.sum().item())
    total_len = int(attention_mask.numel())
    if total_len != seq_len:
        raise RuntimeError(
            "SVOO Hunyuan FlashInfer varlen path requires a total-sequence attention_mask; "
            f"got attention_mask length {total_len} for sequence length {seq_len}."
        )

    from ...kernels.flashinfer_block_sparse import hunyuan_flashinfer_varlen_attn

    q = query.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    hidden_states = hunyuan_flashinfer_varlen_attn(
        q, k, v,
        valid_len=valid_len,
    )
    return hidden_states.reshape(batch, heads, seq_len, dim).permute(0, 2, 1, 3).contiguous()
