from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseAllegroAttnProcessor:
    """Allegro video self-attention processor with pluggable sparse attention."""

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
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise RuntimeError(
                "SparseVideo Allegro processor is only installed for video self-attention; "
                "Allegro text cross-attention must remain dense."
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if image_rotary_emb is not None and not attn.is_cross_attention:
            from diffusers.models.embeddings import apply_rotary_emb_allegro

            query_bhsd = query.transpose(1, 2)
            key_bhsd = key.transpose(1, 2)
            query = apply_rotary_emb_allegro(
                query_bhsd, image_rotary_emb[0], image_rotary_emb[1]
            ).transpose(1, 2)
            key = apply_rotary_emb_allegro(
                key_bhsd, image_rotary_emb[0], image_rotary_emb[1]
            ).transpose(1, 2)

        hidden_states = self.attn_fn(query, key, value, attention_mask)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
