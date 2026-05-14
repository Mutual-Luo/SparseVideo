from __future__ import annotations

from typing import Any

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod


class DenseMethod(SparseMethod):
    CONFIG_DEFAULTS = {}

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"dense not yet supported for {self.model_info.model_type}")
        return original_processor
