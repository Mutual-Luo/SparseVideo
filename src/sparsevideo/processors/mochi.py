from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .._step_tracker import StepTracker


class SparseMochiAttnProcessor:
    """Mochi joint video/text processor with SparseVideo's video-first layout.

    Diffusers Mochi attends over ``[video, valid_text]`` per batch item, where
    invalid/padded prompt tokens are removed before attention and padded back
    afterward. This processor keeps that policy and delegates only the valid
    joint attention call to the selected SparseVideo method.
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
        attention_mask: torch.Tensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

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

        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, *image_rotary_emb)
            key = _apply_rotary_emb(key, *image_rotary_emb)

        batch_size, sequence_length, heads, dim = query.shape
        encoder_sequence_length = encoder_query.shape[1]
        total_length = sequence_length + encoder_sequence_length

        if _can_batch_prompt_mask(attention_mask, batch_size):
            valid_text_indices = _valid_text_indices(attention_mask, 0)
            valid_text_length = int(valid_text_indices.numel())
            valid_query = torch.cat([query, encoder_query[:, valid_text_indices]], dim=1)
            valid_key = torch.cat([key, encoder_key[:, valid_text_indices]], dim=1)
            valid_value = torch.cat([value, encoder_value[:, valid_text_indices]], dim=1)

            hidden_states = self.attn_fn(
                valid_query,
                valid_key,
                valid_value,
                None,
                text_len=valid_text_length,
                prompt_length=valid_text_length,
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
            hidden_states = F.pad(hidden_states, (0, 0, 0, total_length - hidden_states.size(2)))
        else:
            hidden_states = self._per_batch_attention(
                query,
                key,
                value,
                encoder_query,
                encoder_key,
                encoder_value,
                attention_mask,
                batch_size=batch_size,
                total_length=total_length,
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            (sequence_length, encoder_sequence_length), dim=1
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

    def _per_batch_attention(
        self,
        query,
        key,
        value,
        encoder_query,
        encoder_key,
        encoder_value,
        attention_mask,
        *,
        batch_size,
        total_length,
    ):
        attn_outputs = []
        for batch_idx in range(batch_size):
            valid_text_indices = _valid_text_indices(attention_mask, batch_idx)
            valid_text_length = int(valid_text_indices.numel())

            valid_query = torch.cat(
                [query[batch_idx:batch_idx + 1], encoder_query[batch_idx:batch_idx + 1, valid_text_indices]],
                dim=1,
            )
            valid_key = torch.cat(
                [key[batch_idx:batch_idx + 1], encoder_key[batch_idx:batch_idx + 1, valid_text_indices]],
                dim=1,
            )
            valid_value = torch.cat(
                [value[batch_idx:batch_idx + 1], encoder_value[batch_idx:batch_idx + 1, valid_text_indices]],
                dim=1,
            )

            out = self.attn_fn(
                valid_query,
                valid_key,
                valid_value,
                None,
                text_len=valid_text_length,
                prompt_length=valid_text_length,
                cache_key_suffix=batch_idx,
            )
            out = out.permute(0, 2, 1, 3).contiguous()
            out = F.pad(out, (0, 0, 0, total_length - out.size(2)))
            attn_outputs.append(out)

        return torch.cat(attn_outputs, dim=0)


def _apply_rotary_emb(x, freqs_cos, freqs_sin):
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
    sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

    return torch.stack([cos, sin], dim=-1).flatten(-2)


def _valid_text_indices(attention_mask: torch.Tensor, batch_idx: int) -> torch.Tensor:
    mask = attention_mask[batch_idx]
    return torch.nonzero(mask.flatten(), as_tuple=False).flatten()


def _can_batch_prompt_mask(attention_mask: torch.Tensor, batch_size: int) -> bool:
    if batch_size == 1:
        return True
    if attention_mask.is_cuda:
        return False
    return bool(torch.equal(attention_mask, attention_mask[:1].expand_as(attention_mask)))
