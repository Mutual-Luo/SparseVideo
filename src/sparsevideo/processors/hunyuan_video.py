from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseHunyuanVideoAttnProcessor:
    """Drop-in replacement for HunyuanVideoAttnProcessor2_0 with pluggable sparse attention.

    Reproduces all stock logic (QKV projection, QK norm, selective RoPE,
    encoder concatenation, output split) and delegates only the main
    attention call to `attn_fn`.
    """

    _attention_backend = None
    _parallel_config = None

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
        # Single blocks: concatenate encoder before projection
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. RoPE (selective: single blocks apply only to video portion)
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(
                            query[:, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                            sequence_dim=1,
                        ),
                        query[:, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=1,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(
                            key[:, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                            sequence_dim=1,
                        ),
                        key[:, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=1,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 4. Dual blocks: separate encoder QKV then concatenate
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        # 5. Attention (sparse or dense via attn_fn)
        text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        hidden_states = self.attn_fn(query, key, value, attention_mask, text_len=text_len)

        # 6. Output projection and split
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
