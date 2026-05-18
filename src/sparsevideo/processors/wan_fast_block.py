from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch


_PATCH_REFCOUNT = 0
_ORIGINAL_WAN_BLOCK_FORWARD = None
_ORIGINAL_WAN_MODEL_FORWARD = None


def install_wan_fast_block_patch():
    """Install SVOO's package-owned Wan forward patch.

    Upstream SVOO patches both WanTransformerBlock.forward and
    WanTransformer3DModel.forward so the surrounding transformer path passes
    timestep through the patched block and uses Triton layernorm/modulation
    kernels, not only a sparse attention processor. The restore callback keeps
    SparseVideo's apply/restore API reversible.
    """
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock

    global _PATCH_REFCOUNT, _ORIGINAL_WAN_BLOCK_FORWARD, _ORIGINAL_WAN_MODEL_FORWARD
    if _PATCH_REFCOUNT == 0:
        _ORIGINAL_WAN_BLOCK_FORWARD = WanTransformerBlock.forward
        _ORIGINAL_WAN_MODEL_FORWARD = WanTransformer3DModel.forward
        WanTransformerBlock.forward = _svoo_wan_block_forward
        WanTransformer3DModel.forward = _svoo_wan_model_forward
    _PATCH_REFCOUNT += 1

    restored = False

    def restore():
        nonlocal restored
        global _PATCH_REFCOUNT, _ORIGINAL_WAN_BLOCK_FORWARD, _ORIGINAL_WAN_MODEL_FORWARD
        if restored:
            return
        restored = True
        _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
        if _PATCH_REFCOUNT == 0 and _ORIGINAL_WAN_BLOCK_FORWARD is not None:
            WanTransformerBlock.forward = _ORIGINAL_WAN_BLOCK_FORWARD
            WanTransformer3DModel.forward = _ORIGINAL_WAN_MODEL_FORWARD
            _ORIGINAL_WAN_BLOCK_FORWARD = None
            _ORIGINAL_WAN_MODEL_FORWARD = None

    return restore


def _fast_kernel_enabled() -> bool:
    return os.environ.get("ENABLE_FAST_KERNEL", "1") == "1"


def _layernorm_forward(norm, hidden_states):
    from diffusers.models.normalization import FP32LayerNorm

    if not isinstance(norm, FP32LayerNorm):
        raise ValueError(f"Unsupported norm type: {type(norm)}")
    if _fast_kernel_enabled() and hidden_states.is_cuda:
        from ..kernels.layernorm import triton_layernorm_forward

        return triton_layernorm_forward(
            hidden_states,
            norm.weight,
            norm.bias,
            norm.eps,
            norm.elementwise_affine,
        )
    return norm(hidden_states.float())


def _modulate_shift(norm_hidden_states, scale, shift, output_dtype):
    if _fast_kernel_enabled() and norm_hidden_states.is_cuda:
        from ..kernels.modulate import triton_modulate_shift_forward

        return triton_modulate_shift_forward(
            norm_hidden_states,
            scale,
            shift,
            output_dtype=output_dtype,
        )
    return (norm_hidden_states * (1 + scale) + shift).to(output_dtype)


def _modulate_gate_residual(residual, x, gate, output_dtype):
    if _fast_kernel_enabled() and residual.is_cuda:
        from ..kernels.modulate import triton_modulate_gate_residual_forward

        return triton_modulate_gate_residual_forward(
            residual,
            x,
            gate,
            output_dtype=output_dtype,
        )
    return (residual.float() + x.float() * gate).to(output_dtype)


def _svoo_wan_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """WanTransformerBlock.forward ported from training_free/SVOO.

    The sparse attention processor is still responsible for SVOO itself. This
    function only aligns the surrounding Wan block fast kernels with the
    upstream SVOO inference path.
    """
    if temb.ndim == 4:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    norm_hidden_states = _layernorm_forward(self.norm1, hidden_states)
    norm_hidden_states = _modulate_shift(
        norm_hidden_states,
        scale_msa,
        shift_msa,
        hidden_states.dtype,
    )
    attn_output = self.attn1(
        hidden_states=norm_hidden_states,
        rotary_emb=rotary_emb,
        timestep=timestep,
    )
    hidden_states = _modulate_gate_residual(
        hidden_states,
        attn_output,
        gate_msa,
        hidden_states.dtype,
    )

    norm_hidden_states = _layernorm_forward(self.norm2, hidden_states).type_as(hidden_states)
    attn_output = self.attn2(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
    )
    hidden_states = hidden_states + attn_output

    norm_hidden_states = _layernorm_forward(self.norm3, hidden_states)
    norm_hidden_states = _modulate_shift(
        norm_hidden_states,
        c_scale_msa,
        c_shift_msa,
        hidden_states.dtype,
    )
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = _modulate_gate_residual(
        hidden_states,
        ff_output,
        c_gate_msa,
        hidden_states.dtype,
    )
    return hidden_states


def _svoo_wan_model_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
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
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )

    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, timestep=timestep
            )

    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
