from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

from diffusers.models.transformers.transformer_ltx import apply_rotary_emb

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseLTXVideoAttnProcessor:
    """LTX Video self-attention processor with pluggable sparse attention.

    LTX transformer blocks use ``attn1`` for video self-attention and ``attn2``
    for text cross-attention. SparseVideo only installs this processor on
    ``attn1`` so text attention stays on the original dense Diffusers path.
    """

    def __init__(
        self,
        attn_fn: Callable,
        layer_idx: int,
        step_tracker: "StepTracker",
    ):
        self.attn_fn = attn_fn
        self.layer_idx = layer_idx
        self.step_tracker = step_tracker

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise RuntimeError(
                "SparseVideo LTX processor is only installed for video self-attention; "
                "LTX text cross-attention must remain dense."
            )
        if attention_mask is not None:
            raise RuntimeError(
                "SparseVideo LTX self-attention does not support attention_mask; "
                "the standard LTX video attn1 path is unmasked."
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = self.attn_fn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
