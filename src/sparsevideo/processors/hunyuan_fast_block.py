from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch


_PATCH_REFCOUNT = 0
_ORIGINALS = {}


def install_hunyuan_fast_block_patch():
    """Install Triton-accelerated block forward for HunyuanVideo.

    Patches HunyuanVideoTransformerBlock (double block) and
    HunyuanVideoSingleTransformerBlock (single block) to use Triton kernels
    for LayerNorm and modulation operations surrounding attention/FFN,
    while preserving the AdaLayerNormZero path (norm1) unchanged.

    Returns a restore callable for SparseVideo's apply/restore API.
    """
    from diffusers.models.transformers import transformer_hunyuan_video as hunyuan

    global _PATCH_REFCOUNT, _ORIGINALS
    if _PATCH_REFCOUNT == 0:
        _ORIGINALS = {
            hunyuan.HunyuanVideoTransformerBlock: hunyuan.HunyuanVideoTransformerBlock.forward,
            hunyuan.HunyuanVideoSingleTransformerBlock: hunyuan.HunyuanVideoSingleTransformerBlock.forward,
        }
        hunyuan.HunyuanVideoTransformerBlock.forward = _fast_hunyuan_double_block_forward
        hunyuan.HunyuanVideoSingleTransformerBlock.forward = _fast_hunyuan_single_block_forward
    _PATCH_REFCOUNT += 1

    restored = False

    def restore():
        nonlocal restored
        global _PATCH_REFCOUNT, _ORIGINALS
        if restored:
            return
        restored = True
        _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
        if _PATCH_REFCOUNT == 0 and _ORIGINALS:
            for cls, forward in _ORIGINALS.items():
                cls.forward = forward
            _ORIGINALS = {}

    return restore


def _fast_kernel_enabled() -> bool:
    return os.environ.get("ENABLE_FAST_KERNEL", "1") == "1"


def _fast_layernorm_noaffine(hidden_states, eps):
    if _fast_kernel_enabled() and hidden_states.is_cuda:
        from ..kernels.layernorm import triton_layernorm_noparam_forward

        out = triton_layernorm_noparam_forward(hidden_states, eps)
        return out.to(hidden_states.dtype)
    return torch.nn.functional.layer_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)


def _fast_modulate_shift(x, scale, shift):
    if _fast_kernel_enabled() and x.is_cuda:
        from ..kernels.modulate import triton_modulate_shift_forward

        return triton_modulate_shift_forward(x, scale, shift, output_dtype=x.dtype)
    return x * (1 + scale[:, None]) + shift[:, None]


def _fast_gate_residual(residual, x, gate):
    if _fast_kernel_enabled() and residual.is_cuda:
        from ..kernels.modulate import triton_modulate_gate_residual_forward

        return triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=residual.dtype)
    return residual + gate.unsqueeze(1) * x


def _fast_hunyuan_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HunyuanVideoTransformerBlock.forward with Triton norm/modulate kernels."""
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=freqs_cis,
        **kwargs,
    )

    hidden_states = _fast_gate_residual(hidden_states, attn_output, gate_msa)
    encoder_hidden_states = _fast_gate_residual(encoder_hidden_states, context_attn_output, c_gate_msa)

    norm_hidden_states = _fast_layernorm_noaffine(hidden_states, eps=1e-6)
    norm_encoder_hidden_states = _fast_layernorm_noaffine(encoder_hidden_states, eps=1e-6)

    norm_hidden_states = _fast_modulate_shift(norm_hidden_states, scale_mlp, shift_mlp)
    norm_encoder_hidden_states = _fast_modulate_shift(norm_encoder_hidden_states, c_scale_mlp, c_shift_mlp)

    ff_output = self.ff(norm_hidden_states)
    context_ff_output = self.ff_context(norm_encoder_hidden_states)

    hidden_states = _fast_gate_residual(hidden_states, ff_output, gate_mlp)
    encoder_hidden_states = _fast_gate_residual(encoder_hidden_states, context_ff_output, c_gate_mlp)
    return hidden_states, encoder_hidden_states


def _fast_hunyuan_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HunyuanVideoSingleTransformerBlock.forward with Triton gate kernel."""
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
        **kwargs,
    )
    attn_output = torch.cat([attn_output, context_attn_output], dim=1)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
    hidden_states = hidden_states + residual

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )
    return hidden_states, encoder_hidden_states
