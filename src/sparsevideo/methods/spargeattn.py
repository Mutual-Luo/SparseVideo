from __future__ import annotations

from typing import Any

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor


class SpargeAttnMethod(SparseMethod):
    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def __init__(self, config, model_info):
        try:
            from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
            self._sparge_fn = spas_sage2_attn_meansim_topk_cuda
        except ImportError:
            raise ImportError(
                "spargeattn method requires the spas_sage_attn package. "
                "Install from: https://github.com/thu-ml/SpargeAttn or via "
                "pip install sparsevideo[spargeattn]"
            )
        super().__init__(config, model_info)

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type == "wan":
            topk = self.config["budget"]
            skip_steps = self.config["skip_first_steps"]
            skip_layers = self.config["skip_first_layers"]
            sparge_fn = self._sparge_fn

            def attn_fn(query, key, value, attention_mask):
                use_sparse = (
                    query.is_cuda
                    and query.shape[1] >= 128  # seq_len
                    and query.shape[-1] in (64, 128)  # head_dim
                    and attention_mask is None
                    and layer_idx >= skip_layers
                    and step_tracker.step > skip_steps
                )

                if not use_sparse:
                    return dispatch_attention_fn(
                        query, key, value,
                        attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                    )

                # Diffusers layout: [B, N, H, D] → SpargeAttn layout: [B, H, N, D]
                q_hnd = query.permute(0, 2, 1, 3).contiguous()
                k_hnd = key.permute(0, 2, 1, 3).contiguous()
                v_hnd = value.permute(0, 2, 1, 3).contiguous()

                o_hnd = sparge_fn(q_hnd, k_hnd, v_hnd, topk=topk)

                return o_hnd.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]

            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        raise NotImplementedError(f"spargeattn not yet supported for {self.model_info.model_type}")
