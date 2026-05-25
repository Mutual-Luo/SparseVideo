"""Tests for fused kernel equivalence: Triton vs PyTorch reference paths.

Each test verifies that the Triton (or native C++) fused kernel produces
results within numerical tolerance of the PyTorch reference implementation.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA/Triton")


# ============================================================
# RMSNorm inplace equivalence
# ============================================================

@skip_no_cuda
def test_triton_rmsnorm_matches_pytorch(monkeypatch):
    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    from sparsevideo.kernels.fused_norm_rope import triton_rmsnorm_inplace

    torch.manual_seed(42)
    hidden_dim = 128
    x = torch.randn(2, 64, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
    eps = 1e-6

    x_ref = x.clone()
    variance = x_ref.float().pow(2).mean(-1, keepdim=True)
    x_normed = (x_ref.float() * torch.rsqrt(variance + eps)).to(x_ref.dtype)
    expected = x_normed * weight

    actual = triton_rmsnorm_inplace(x.clone(), weight, eps)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@skip_no_cuda
def test_triton_rmsnorm_matches_pytorch_float32(monkeypatch):
    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    from sparsevideo.kernels.fused_norm_rope import triton_rmsnorm_inplace

    torch.manual_seed(7)
    hidden_dim = 64
    x = torch.randn(1, 32, hidden_dim, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    eps = 1e-6

    variance = x.pow(2).mean(-1, keepdim=True)
    expected = (x * torch.rsqrt(variance + eps)) * weight

    actual = triton_rmsnorm_inplace(x.clone(), weight, eps)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ============================================================
# RoPE Wan inplace equivalence
# ============================================================

@skip_no_cuda
def test_triton_rope_wan_matches_pytorch(monkeypatch):
    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    from sparsevideo.kernels.fused_norm_rope import triton_rope_wan_inplace

    torch.manual_seed(11)
    B, S, H, D = 1, 16, 4, 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    freqs = torch.randn(1, S, 1, D, device="cuda", dtype=torch.float16)
    cos_f = freqs.cos()
    sin_f = freqs.sin()

    def pytorch_rope_wan(x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[..., 0::2]
        s = sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out.type_as(x)

    expected_q = pytorch_rope_wan(q, cos_f, sin_f)
    expected_k = pytorch_rope_wan(k, cos_f, sin_f)

    actual_q, actual_k = triton_rope_wan_inplace(q.clone(), k.clone(), cos_f, sin_f)

    torch.testing.assert_close(actual_q, expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-3, rtol=1e-3)


@skip_no_cuda
def test_triton_rope_wan_larger_sequence(monkeypatch):
    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    from sparsevideo.kernels.fused_norm_rope import triton_rope_wan_inplace

    torch.manual_seed(99)
    B, S, H, D = 1, 256, 8, 128
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    freqs = torch.randn(1, S, 1, D, device="cuda", dtype=torch.float16)
    cos_f = freqs.cos()
    sin_f = freqs.sin()

    def pytorch_rope_wan(x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[..., 0::2]
        s = sin[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out.type_as(x)

    expected_q = pytorch_rope_wan(q, cos_f, sin_f)
    expected_k = pytorch_rope_wan(k, cos_f, sin_f)

    actual_q, actual_k = triton_rope_wan_inplace(q.clone(), k.clone(), cos_f, sin_f)

    torch.testing.assert_close(actual_q, expected_q, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=2e-3, rtol=2e-3)


# ============================================================
# RoPE HunyuanVideo inplace equivalence
# ============================================================

@skip_no_cuda
def test_triton_rope_hyvideo_matches_pytorch(monkeypatch):
    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    from sparsevideo.kernels.fused_norm_rope import triton_rope_hyvideo_inplace, _rope_hyvideo_pytorch

    torch.manual_seed(22)
    B, S, H, D = 1, 20, 4, 64
    txt_len = 5
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(S - txt_len, D, device="cuda", dtype=torch.float16)
    sin = torch.randn(S - txt_len, D, device="cuda", dtype=torch.float16)

    expected_q, expected_k = _rope_hyvideo_pytorch(q.clone().cpu(), k.clone().cpu(), cos.cpu(), sin.cpu(), txt_len)
    actual_q, actual_k = triton_rope_hyvideo_inplace(q.clone(), k.clone(), cos, sin, txt_len)

    torch.testing.assert_close(actual_q.cpu(), expected_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual_k.cpu(), expected_k, atol=1e-3, rtol=1e-3)


# ============================================================
# LayerNorm (fast-block patch) equivalence
# ============================================================

@skip_no_cuda
def test_triton_layernorm_matches_pytorch():
    from sparsevideo.kernels.layernorm import triton_layernorm_forward

    torch.manual_seed(33)
    hidden_dim = 128
    x = torch.randn(2, 64, hidden_dim, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    eps = 1e-5

    layer_norm = nn.LayerNorm(hidden_dim, eps=eps).cuda()
    layer_norm.weight.data.copy_(weight)
    layer_norm.bias.data.copy_(bias)
    expected = layer_norm(x)

    actual = triton_layernorm_forward(x, weight, bias, eps, elementwise_affine=True)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@skip_no_cuda
def test_triton_layernorm_no_affine_matches_pytorch():
    from sparsevideo.kernels.layernorm import triton_layernorm_noparam_forward

    torch.manual_seed(44)
    hidden_dim = 256
    x = torch.randn(1, 32, hidden_dim, device="cuda", dtype=torch.float32)
    eps = 1e-6

    layer_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps).cuda()
    expected = layer_norm(x)

    actual = triton_layernorm_noparam_forward(x, eps)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ============================================================
# Modulate shift/gate (fast-block patch) equivalence
# ============================================================

@skip_no_cuda
def test_triton_modulate_shift_matches_pytorch():
    from sparsevideo.kernels.modulate import triton_modulate_shift_forward

    torch.manual_seed(55)
    B, S, D = 2, 64, 128
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.randn(B, D, device="cuda", dtype=torch.float32)
    shift = torch.randn(B, D, device="cuda", dtype=torch.float32)

    expected = (x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)).to(torch.float16)
    actual = triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.float16)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@skip_no_cuda
def test_triton_modulate_shift_matches_wan_broadcast_shape():
    from sparsevideo.kernels.modulate import triton_modulate_shift_forward

    torch.manual_seed(56)
    B, S, D = 2, 64, 128
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float32)
    scale = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)
    shift = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)

    expected = (x * (1 + scale) + shift).to(torch.bfloat16)
    actual = triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.bfloat16)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@skip_no_cuda
def test_triton_modulate_gate_residual_matches_pytorch():
    from sparsevideo.kernels.modulate import triton_modulate_gate_residual_forward

    torch.manual_seed(66)
    B, S, D = 2, 64, 128
    residual = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
    gate = torch.randn(B, D, device="cuda", dtype=torch.float16)

    expected = (residual.float() + x.float() * gate.unsqueeze(1).float()).to(torch.float16)
    actual = triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.float16)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@skip_no_cuda
def test_triton_modulate_gate_residual_matches_wan_broadcast_shape():
    from sparsevideo.kernels.modulate import triton_modulate_gate_residual_forward

    torch.manual_seed(67)
    B, S, D = 2, 64, 128
    residual = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(B, 1, D, device="cuda", dtype=torch.float32)

    expected = (residual.float() + x.float() * gate).to(torch.bfloat16)
    actual = triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.bfloat16)

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# ============================================================
# Fast-block patch end-to-end equivalence
# ============================================================

@skip_no_cuda
def test_wan_fast_block_forward_matches_stock_forward():
    """Verify fast-block patched forward produces same output as stock."""
    from diffusers.models.normalization import FP32LayerNorm
    from sparsevideo.processors.wan_fast_block import (
        _layernorm_forward,
        _modulate_shift,
        _modulate_gate_residual,
    )

    torch.manual_seed(77)
    hidden_dim = 128
    x = torch.randn(1, 16, hidden_dim, device="cuda", dtype=torch.float16)
    scale = torch.randn(1, 1, hidden_dim, device="cuda", dtype=torch.float16)
    shift = torch.randn(1, 1, hidden_dim, device="cuda", dtype=torch.float16)
    gate = torch.randn(1, 1, hidden_dim, device="cuda", dtype=torch.float16)

    norm = FP32LayerNorm(hidden_dim).cuda()

    norm_out_stock = norm(x.float())
    norm_out_fast = _layernorm_forward(norm, x)
    torch.testing.assert_close(norm_out_fast, norm_out_stock, atol=1e-5, rtol=1e-5)

    modulated_stock = (norm_out_stock * (1 + scale) + shift).to(x.dtype)
    modulated_fast = _modulate_shift(norm_out_fast, scale, shift, x.dtype)
    torch.testing.assert_close(modulated_fast, modulated_stock, atol=1e-3, rtol=1e-3)

    residual = x.clone()
    attn_output = torch.randn_like(x)
    gated_stock = (residual.float() + attn_output.float() * gate).to(x.dtype)
    gated_fast = _modulate_gate_residual(residual, attn_output, gate, x.dtype)
    torch.testing.assert_close(gated_fast, gated_stock, atol=1e-3, rtol=1e-3)


# ============================================================
# Spargeattn fused path does not change attention output
# ============================================================

@skip_no_cuda
def test_spargeattn_fused_vs_unfused_processor_same_output():
    """Verify spargeattn with use_fused_qk_norm_rope=True vs False produces same Q/K."""
    from sparsevideo.processors.wan import SparseWanAttnProcessor

    class _StepTracker:
        step = 0

    class _WanRMSNormAttention:
        def __init__(self, hidden_dim=128, heads=4):
            self.heads = heads
            self.fused_projections = False
            self.cross_attention_dim_head = None
            self.add_k_proj = None
            self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda().half()
            self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda().half()
            self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda().half()
            self.norm_q = nn.RMSNorm(hidden_dim, eps=1e-6).cuda().half()
            self.norm_k = nn.RMSNorm(hidden_dim, eps=1e-6).cuda().half()
            self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False).cuda().half(), nn.Identity()]

    torch.manual_seed(88)
    attn = _WanRMSNormAttention()
    hidden_states = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)

    captured_fused = {}
    captured_unfused = {}

    def capture_fn(name):
        def fn(query, key, value, attention_mask, **kwargs):
            captured = captured_fused if name == "fused" else captured_unfused
            captured["q"] = query.clone()
            captured["k"] = key.clone()
            captured["v"] = value.clone()
            return query
        return fn

    SparseWanAttnProcessor(
        attn_fn=capture_fn("fused"), layer_idx=0, step_tracker=_StepTracker(),
        use_fused_qk_norm_rope=True,
    )(attn, hidden_states.clone())

    SparseWanAttnProcessor(
        attn_fn=capture_fn("unfused"), layer_idx=0, step_tracker=_StepTracker(),
        use_fused_qk_norm_rope=False,
    )(attn, hidden_states.clone())

    torch.testing.assert_close(
        captured_fused["q"], captured_unfused["q"], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_fused["k"], captured_unfused["k"], atol=1e-2, rtol=1e-2
    )
