from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseEasyAnimateAttnProcessor:
    """EasyAnimate joint text/video processor with SparseVideo's video-first layout."""

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
        if attention_mask is not None:
            raise RuntimeError(
                "SparseVideo EasyAnimate does not support text attention masks yet; "
                "run without prompt_attention_mask or keep EasyAnimate dense."
            )

        text_seq_length = encoder_hidden_states.size(1)
        video_seq_length = hidden_states.size(1)

        if attn.add_q_proj is None:
            return self._concat_projection_attention(
                attn,
                hidden_states,
                encoder_hidden_states,
                text_seq_length=text_seq_length,
                video_seq_length=video_seq_length,
                image_rotary_emb=image_rotary_emb,
            )
        return self._added_projection_attention(
            attn,
            hidden_states,
            encoder_hidden_states,
            text_seq_length=text_seq_length,
            video_seq_length=video_seq_length,
            image_rotary_emb=image_rotary_emb,
        )

    def _concat_projection_attention(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        *,
        text_seq_length,
        video_seq_length,
        image_rotary_emb,
    ):
        text_first_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        query = attn.to_q(text_first_states)
        key = attn.to_k(text_first_states)
        value = attn.to_v(text_first_states)

        query, key, value = _to_heads(attn, query, key, value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query, key = _apply_video_rotary(
            attn,
            query,
            key,
            text_seq_length=text_seq_length,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = self._dispatch_video_first(
            query,
            key,
            value,
            text_seq_length=text_seq_length,
            video_seq_length=video_seq_length,
        )
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, video_seq_length], dim=1
        )
        return _project_outputs(attn, hidden_states, encoder_hidden_states)

    def _added_projection_attention(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        *,
        text_seq_length,
        video_seq_length,
        image_rotary_emb,
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query, key, value = _to_heads(attn, query, key, value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
        encoder_query, encoder_key, encoder_value = _to_heads(
            attn, encoder_query, encoder_key, encoder_value
        )
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)
        query, key = _apply_video_rotary(
            attn,
            query,
            key,
            text_seq_length=text_seq_length,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = self._dispatch_video_first(
            query,
            key,
            value,
            text_seq_length=text_seq_length,
            video_seq_length=video_seq_length,
        )
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, video_seq_length], dim=1
        )
        return _project_outputs(attn, hidden_states, encoder_hidden_states)

    def _dispatch_video_first(self, query, key, value, *, text_seq_length, video_seq_length):
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
        return hidden_states.type_as(query)


def _to_heads(attn, query, key, value):
    query = query.unflatten(2, (attn.heads, -1)).contiguous()
    key = key.unflatten(2, (attn.heads, -1)).contiguous()
    value = value.unflatten(2, (attn.heads, -1)).contiguous()
    return query, key, value


def _apply_video_rotary(attn, query, key, *, text_seq_length, image_rotary_emb):
    if image_rotary_emb is None:
        return query, key

    from diffusers.models.embeddings import apply_rotary_emb

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    query[:, :, text_seq_length:] = apply_rotary_emb(
        query[:, :, text_seq_length:], image_rotary_emb
    )
    if not attn.is_cross_attention:
        key[:, :, text_seq_length:] = apply_rotary_emb(
            key[:, :, text_seq_length:], image_rotary_emb
        )
    return query.transpose(1, 2).contiguous(), key.transpose(1, 2).contiguous()


def _project_outputs(attn, hidden_states, encoder_hidden_states):
    if getattr(attn, "to_out", None) is not None:
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
    if getattr(attn, "to_add_out", None) is not None:
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
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
