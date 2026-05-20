from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseCogVideoXAttnProcessor:
    """CogVideoX processor with SparseVideo's video-first internal layout.

    Diffusers CogVideoX attention concatenates tokens as [text, video]. Existing
    SparseVideo text-aware kernels expect [video, text], so this processor
    reorders Q/K/V before calling the method and restores the stock order before
    output projection.
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
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        video_seq_length = hidden_states.size(1)

        if attention_mask is not None:
            raise RuntimeError(
                "CogVideoX sparse attention with attention_mask is not implemented; "
                "the sparse processor supports the standard unmasked CogVideoX path."
            )

        text_first_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, _, _ = text_first_states.shape

        query = attn.to_q(text_first_states)
        key = attn.to_k(text_first_states)
        value = attn.to_v(text_first_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        query = _text_first_to_video_first(query, text_seq_length)
        key = _text_first_to_video_first(key, text_seq_length)
        value = _text_first_to_video_first(value, text_seq_length)

        hidden_states = self.attn_fn(
            query,
            key,
            value,
            None,
            text_len=text_seq_length,
            prompt_length=text_seq_length,
        )

        hidden_states = _video_first_to_text_first(
            hidden_states,
            video_seq_length=video_seq_length,
            text_seq_length=text_seq_length,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, video_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def _text_first_to_video_first(hidden_states: torch.Tensor, text_seq_length: int) -> torch.Tensor:
    if text_seq_length <= 0:
        return hidden_states
    return torch.cat(
        [hidden_states[:, text_seq_length:], hidden_states[:, :text_seq_length]],
        dim=1,
    )


def _video_first_to_text_first(
    hidden_states: torch.Tensor,
    *,
    video_seq_length: int,
    text_seq_length: int,
) -> torch.Tensor:
    if text_seq_length <= 0:
        return hidden_states
    return torch.cat(
        [
            hidden_states[:, video_seq_length:video_seq_length + text_seq_length],
            hidden_states[:, :video_seq_length],
        ],
        dim=1,
    )
