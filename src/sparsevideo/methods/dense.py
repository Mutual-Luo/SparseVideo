from __future__ import annotations

from typing import Any

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod


class DenseMethod(SparseMethod):
    CONFIG_DEFAULTS = {}

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        return original_processor
