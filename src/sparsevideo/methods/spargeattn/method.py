from __future__ import annotations

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class SpargeAttnMethod(SparseMethod):
    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if self.config["mode"] not in ("full", "cdfthreshd", "topk"):
            raise ValueError("spargeattn mode must be one of: full, cdfthreshd, topk")
        unsupported = [
            name
            for name, default in method_config.UNPORTED_OPTION_DEFAULTS.items()
            if self.config[name] != default
        ]
        if unsupported:
            raise NotImplementedError(
                "These upstream SpargeAttn tuning options are recognized but not "
                f"ported in SparseVideo yet: {unsupported}"
            )
        self._sparge_cdf_fn = None
        self._sparge_topk_fn = None
        if self.config["mode"] != "full":
            if self.config["value"] is None:
                raise ValueError("spargeattn value is required when mode is cdfthreshd or topk")
            try:
                from spas_sage_attn import spas_sage2_attn_meansim_cuda
                from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
                self._sparge_cdf_fn = spas_sage2_attn_meansim_cuda
                self._sparge_topk_fn = spas_sage2_attn_meansim_topk_cuda
            except ImportError:
                raise ImportError(
                    "spargeattn method requires the spas_sage_attn package when mode is not full. "
                    "Install from: https://github.com/thu-ml/SpargeAttn or via "
                    "pip install sparsevideo[spargeattn]"
                )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"spargeattn not yet supported for {self.model_info.model_type}")

        mode = self.config["mode"]
        sparge_value = self.config["value"]
        sparge_cdf_fn = self._sparge_cdf_fn
        sparge_topk_fn = self._sparge_topk_fn

        def attn_fn(query, key, value, attention_mask, **kwargs):
            use_sparse = (
                query.is_cuda
                and query.shape[1] >= 128  # seq_len
                and query.shape[-1] in (64, 128)  # head_dim
                and attention_mask is None
                and mode != "full"
            )

            if not use_sparse:
                if mode != "full":
                    raise RuntimeError(
                        "spargeattn sparse mode requires CUDA, seq_len >= 128, "
                        "head_dim in (64, 128), and no attention mask"
                    )
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )

            # Diffusers layout: [B, N, H, D] → SpargeAttn layout: [B, H, N, D]
            q_hnd = query.permute(0, 2, 1, 3).contiguous()
            k_hnd = key.permute(0, 2, 1, 3).contiguous()
            v_hnd = value.permute(0, 2, 1, 3).contiguous()

            if mode == "cdfthreshd":
                o_hnd = sparge_cdf_fn(q_hnd, k_hnd, v_hnd, cdfthreshd=sparge_value)
            else:
                o_hnd = sparge_topk_fn(q_hnd, k_hnd, v_hnd, topk=sparge_value)

            return o_hnd.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )
