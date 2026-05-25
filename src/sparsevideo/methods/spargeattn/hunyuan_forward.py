from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


_PATCH_REFCOUNT = 0
_ORIGINALS = {}


def install_spargeattn_hunyuan_forward_patch(model_info):
    """Install SpargeAttn's owned Hunyuan forward patch.

    Upstream SpargeAttn's Hunyuan path trims padded text tokens before the
    attention blocks and then passes ``attention_mask=None`` because the
    spas_sage_attn sparse kernels do not support Hunyuan's padding mask.
    """
    from diffusers.models.transformers import transformer_hunyuan_video as hunyuan

    patch_map = {
        hunyuan.HunyuanVideoTransformer3DModel: spargeattn_hunyuan_forward,
    }

    old_forward_patches = []
    for transformer in model_info.transformers:
        if hasattr(transformer, "_old_forward"):
            old_forward_patches.append((transformer, transformer._old_forward))
            transformer._old_forward = spargeattn_hunyuan_forward.__get__(transformer, type(transformer))

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
        for transformer, old_forward in reversed(old_forward_patches):
            transformer._old_forward = old_forward
        _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
        if _PATCH_REFCOUNT == 0:
            for cls, forward in _ORIGINALS.items():
                cls.forward = forward
            _ORIGINALS = {}

    return restore


def spargeattn_hunyuan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, _num_channels, num_frames, height, width = hidden_states.shape
    if batch_size != 1:
        raise RuntimeError(
            "SpargeAttn Hunyuan forward follows upstream batch size 1 because "
            "the sparse kernel path does not support attention_mask."
        )

    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    effective_condition_sequence_length = _spargeattn_hunyuan_effective_condition_length(
        encoder_attention_mask,
        batch_size=batch_size,
        max_length=encoder_hidden_states.shape[1],
        device=encoder_hidden_states.device,
    )
    encoder_hidden_states = encoder_hidden_states[:, :effective_condition_sequence_length]
    attention_mask = None

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
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


def _spargeattn_hunyuan_effective_condition_length(
    encoder_attention_mask,
    *,
    batch_size: int,
    max_length: int,
    device,
) -> int:
    if encoder_attention_mask is None:
        return int(max_length)

    mask = encoder_attention_mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    else:
        mask = mask.reshape(mask.shape[0], -1)

    if mask.shape[0] != int(batch_size):
        raise RuntimeError(
            f"SpargeAttn Hunyuan attention_mask batch mismatch: got {mask.shape[0]}, expected {int(batch_size)}"
        )
    if int(batch_size) != 1:
        raise RuntimeError(
            "SpargeAttn Hunyuan forward follows upstream batch size 1 because "
            "the sparse kernel path does not support attention_mask."
        )

    valid_len = int(mask[0].to(torch.int64).sum().item())
    return max(0, min(valid_len, int(max_length)))
