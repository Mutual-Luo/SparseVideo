from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

if TYPE_CHECKING:
    from .._step_tracker import StepTracker

try:
    from ..kernels.fused_norm_rope import triton_rmsnorm_inplace, triton_rope_hyvideo_inplace
    _HAS_FUSED_KERNELS = True
except Exception:
    _HAS_FUSED_KERNELS = False


def _norm_weight_is_materialized(norm) -> bool:
    weight = getattr(norm, "weight", None)
    return weight is not None and not bool(getattr(weight, "is_meta", False))


def _can_use_fused_qk_norm(norm_q, norm_k) -> bool:
    return _norm_weight_is_materialized(norm_q) and _norm_weight_is_materialized(norm_k)


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
        use_fused_qk_norm_rope: bool = True,
        use_fused_qk_norm: Optional[bool] = None,
        use_fused_rope: Optional[bool] = None,
        query_projection_fn: Optional[Callable] = None,
        output_projection_fn: Optional[Callable] = None,
    ):
        self.attn_fn = attn_fn
        self.layer_idx = layer_idx
        self.step_tracker = step_tracker
        self.use_fused_qk_norm = use_fused_qk_norm_rope if use_fused_qk_norm is None else use_fused_qk_norm
        self.use_fused_rope = use_fused_qk_norm_rope if use_fused_rope is None else use_fused_rope
        self.use_fused_qk_norm_rope = self.use_fused_qk_norm and self.use_fused_rope
        self.query_projection_fn = query_projection_fn
        self.output_projection_fn = output_projection_fn

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        cache_dic: Optional[dict] = None,
        current: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Single blocks: concatenate encoder before projection
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        if self.query_projection_fn is not None and attn.add_q_proj is None:
            query = self.query_projection_fn(attn.to_q, hidden_states, attn.heads)
        else:
            query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization — Triton inplace when available
        if (
            self.use_fused_qk_norm
            and _HAS_FUSED_KERNELS
            and query.is_cuda
            and _can_use_fused_qk_norm(attn.norm_q, attn.norm_k)
        ):
            query = triton_rmsnorm_inplace(query, attn.norm_q.weight, attn.norm_q.eps)
            key   = triton_rmsnorm_inplace(key,   attn.norm_k.weight, attn.norm_k.eps)
        else:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        # 3. RoPE (selective: single blocks apply only to video portion)
        if image_rotary_emb is not None:
            if self.use_fused_rope and _HAS_FUSED_KERNELS and query.is_cuda:
                cos, sin = image_rotary_emb
                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    # Single block: text tokens at end, skip them for RoPE
                    txt_len = encoder_hidden_states.shape[1]
                    query, key = triton_rope_hyvideo_inplace(query, key, cos, sin, txt_len=txt_len)
                else:
                    # Dual block: all tokens are video, txt_len=0
                    query, key = triton_rope_hyvideo_inplace(query, key, cos, sin, txt_len=0)
            else:
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
        prompt_length = _prompt_length_from_attention_mask(attention_mask, query.shape[1], text_len)
        method_kwargs = {
            "text_len": text_len,
            "prompt_length": prompt_length,
            "timestep": timestep,
        }
        if cache_dic is not None:
            method_kwargs["cache_dic"] = cache_dic
        if current is not None:
            method_kwargs["current"] = current
        hidden_states = self.attn_fn(
            query, key, value, attention_mask,
            **method_kwargs,
        )

        # 6. Output projection and split
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                if self.output_projection_fn is not None:
                    hidden_states = self.output_projection_fn(attn.to_out[0], hidden_states, attn.heads)
                else:
                    hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def _prompt_length_from_attention_mask(
    attention_mask: Optional[torch.Tensor],
    total_seq_len: int,
    text_len: int,
) -> Optional[int]:
    if text_len <= 0:
        return 0
    if attention_mask is None:
        return None

    if attention_mask.shape[-1] == int(text_len):
        flat_mask = attention_mask.reshape(attention_mask.shape[0], -1)
        prompt_lengths = flat_mask.to(torch.int64).sum(dim=-1).clamp(min=0, max=int(text_len))
        return int(prompt_lengths.min().item())

    if attention_mask.shape[-1] != total_seq_len:
        return None

    video_len = max(0, int(total_seq_len) - int(text_len))
    flat_mask = attention_mask.reshape(attention_mask.shape[0], -1)
    valid_lengths = flat_mask.to(torch.int64).sum(dim=-1)
    prompt_lengths = (valid_lengths - video_len).clamp(min=0, max=int(text_len))
    return int(prompt_lengths.min().item())
