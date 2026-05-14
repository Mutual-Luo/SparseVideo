from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class TopKMethod(SparseMethod):
    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"topk not yet supported for {self.model_info.model_type}")

        budget = self.config["budget"]
        skip_steps = self.config["skip_first_steps"]
        skip_layers = self.config["skip_first_layers"]

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

            # query/key/value: [B, seq, heads, head_dim]
            B, N, H, D = query.shape
            scale = D ** -0.5

            q = query.permute(0, 2, 1, 3)  # [B, H, N, D]
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

            k_val = max(1, int(N * budget))
            _, topk_indices = torch.topk(scores, k=k_val, dim=-1)

            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(dim=-1, index=topk_indices, value=0.0)
            scores = scores + mask

            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)  # [B, H, N, D]
            return out.permute(0, 2, 1, 3)  # [B, N, H, D]

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )
