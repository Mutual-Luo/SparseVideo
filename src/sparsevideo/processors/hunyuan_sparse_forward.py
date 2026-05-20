from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


_PATCH_REFCOUNT = 0
_ORIGINALS = {}


def _normalize_token_replace_forward_args(timestep, token_replace_emb, first_frame_num_tokens):
    if first_frame_num_tokens is None and isinstance(token_replace_emb, int) and torch.is_tensor(timestep):
        return None, timestep, token_replace_emb
    return timestep, token_replace_emb, first_frame_num_tokens


def install_hunyuan_sparse_forward_patch():
    """Install SVOO's package-owned Hunyuan forward patch.

    Upstream SVOO patches Hunyuan block/model forwards so attention processors
    receive the scheduler timestep and the text context is capped to the 256
    tokens expected by the Hunyuan10 sparse path. SparseVideo keeps the patch
    reversible through the public apply/restore handle.
    """
    from diffusers.models.transformers import transformer_hunyuan_video as hunyuan

    patch_map = {
        hunyuan.HunyuanVideoSingleTransformerBlock: _svoo_hunyuan_single_block_forward,
        hunyuan.HunyuanVideoTransformerBlock: _svoo_hunyuan_double_block_forward,
        hunyuan.HunyuanVideoTransformer3DModel: _svoo_hunyuan_transformer_forward,
    }
    token_block = getattr(hunyuan, "HunyuanVideoTokenReplaceTransformerBlock", None)
    if token_block is not None:
        patch_map[token_block] = _svoo_hunyuan_token_replace_block_forward
    token_single = getattr(hunyuan, "HunyuanVideoTokenReplaceSingleTransformerBlock", None)
    if token_single is not None:
        patch_map[token_single] = _svoo_hunyuan_token_replace_single_block_forward

    global _PATCH_REFCOUNT, _ORIGINALS
    if _PATCH_REFCOUNT == 0:
        _ORIGINALS = {cls: cls.forward for cls in patch_map}
        for cls, forward in patch_map.items():
            cls.forward = forward
    _PATCH_REFCOUNT += 1

    restored = False

    def restore():
        nonlocal restored
        global _PATCH_REFCOUNT, _ORIGINALS
        if restored:
            return
        restored = True
        _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
        if _PATCH_REFCOUNT == 0:
            for cls, forward in _ORIGINALS.items():
                cls.forward = forward
            _ORIGINALS = {}

    return restore


def _svoo_hunyuan_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_hidden_states, norm_encoder_hidden_states = (
        norm_hidden_states[:, :-text_seq_length, :],
        norm_hidden_states[:, -text_seq_length:, :],
    )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
        timestep=timestep,
    )
    attn_output = torch.cat([attn_output, context_attn_output], dim=1)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = gate.unsqueeze(1) * hidden_states
    hidden_states = hidden_states + residual

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )
    return hidden_states, encoder_hidden_states


def _svoo_hunyuan_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=freqs_cis,
        timestep=timestep,
    )

    hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

    norm_hidden_states = self.norm2(hidden_states)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    return hidden_states, encoder_hidden_states


def _svoo_hunyuan_token_replace_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep: Optional[torch.Tensor] = None,
    token_replace_emb: torch.Tensor = None,
    first_frame_num_tokens: int = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    timestep, token_replace_emb, first_frame_num_tokens = _normalize_token_replace_forward_args(
        timestep, token_replace_emb, first_frame_num_tokens,
    )
    num_tokens = first_frame_num_tokens
    (
        norm_hidden_states,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        tr_gate_msa,
        tr_shift_mlp,
        tr_scale_mlp,
        tr_gate_mlp,
    ) = self.norm1(hidden_states, temb, token_replace_emb, num_tokens)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=freqs_cis,
        timestep=timestep,
    )

    hidden_states_zero = hidden_states[:, :num_tokens] + attn_output[:, :num_tokens] * tr_gate_msa.unsqueeze(1)
    hidden_states_orig = hidden_states[:, num_tokens:] + attn_output[:, num_tokens:] * gate_msa.unsqueeze(1)
    hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
    encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

    norm_hidden_states = self.norm2(hidden_states)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

    hidden_states_zero = norm_hidden_states[:, :num_tokens] * (1 + tr_scale_mlp[:, None]) + tr_shift_mlp[:, None]
    hidden_states_orig = norm_hidden_states[:, num_tokens:] * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    norm_hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states_zero = hidden_states[:, :num_tokens] + ff_output[:, :num_tokens] * tr_gate_mlp.unsqueeze(1)
    hidden_states_orig = hidden_states[:, num_tokens:] + ff_output[:, num_tokens:] * gate_mlp.unsqueeze(1)
    hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    return hidden_states, encoder_hidden_states


def _svoo_hunyuan_token_replace_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep: Optional[torch.Tensor] = None,
    token_replace_emb: torch.Tensor = None,
    first_frame_num_tokens: int = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    timestep, token_replace_emb, first_frame_num_tokens = _normalize_token_replace_forward_args(
        timestep, token_replace_emb, first_frame_num_tokens,
    )
    num_tokens = first_frame_num_tokens
    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate, tr_gate = self.norm(hidden_states, temb, token_replace_emb, num_tokens)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_hidden_states, norm_encoder_hidden_states = (
        norm_hidden_states[:, :-text_seq_length, :],
        norm_hidden_states[:, -text_seq_length:, :],
    )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
        timestep=timestep,
    )
    attn_output = torch.cat([attn_output, context_attn_output], dim=1)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    proj_output = self.proj_out(hidden_states)
    hidden_states_zero = proj_output[:, :num_tokens] * tr_gate.unsqueeze(1)
    hidden_states_orig = proj_output[:, num_tokens:] * gate.unsqueeze(1)
    hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
    hidden_states = hidden_states + residual

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )
    return hidden_states, encoder_hidden_states


def _svoo_hunyuan_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    context_length_limit = 256
    if encoder_hidden_states.shape[1] > context_length_limit:
        encoder_hidden_states = encoder_hidden_states[:, :context_length_limit, :]
        encoder_attention_mask = encoder_attention_mask[:, :context_length_limit]

    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool)
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                token_replace_emb,
                first_frame_num_tokens,
            )
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                timestep,
                token_replace_emb,
                first_frame_num_tokens,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)
    return Transformer2DModelOutput(sample=hidden_states)
