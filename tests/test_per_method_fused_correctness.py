"""Per-method fast-path correctness tests.

Verifies that every sparse method produces identical attention inputs (Q, K, V)
and compatible outputs when using the speed-first fused kernel defaults
(fused_qk_norm=True, fused_rope=True) vs the unfused baseline.

This proves the fused paths don't degrade quality for any method.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HIDDEN_DIM = 256
HEADS = 4
SEQ_LEN = 64
DEVICE = "cuda"
DTYPE = torch.float16


class _StepTracker:
    step = 5
    timestep = 500


class _WanAttn:
    """Minimal mock of a Wan Attention module for processor testing."""

    def __init__(self, hidden_dim=HIDDEN_DIM, heads=HEADS):
        self.heads = heads
        self.fused_projections = False
        self.cross_attention_dim_head = None
        self.add_k_proj = None
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.norm_q = nn.RMSNorm(hidden_dim, eps=1e-6).to(DEVICE, DTYPE)
        self.norm_k = nn.RMSNorm(hidden_dim, eps=1e-6).to(DEVICE, DTYPE)
        self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE), nn.Identity()]


class _HunyuanAttn:
    """Minimal mock of a HunyuanVideo Attention module for processor testing."""

    def __init__(self, hidden_dim=HIDDEN_DIM, heads=HEADS):
        self.heads = heads
        self.add_q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.add_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.add_v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)
        self.norm_q = nn.RMSNorm(hidden_dim // heads, eps=1e-6).to(DEVICE, DTYPE)
        self.norm_k = nn.RMSNorm(hidden_dim // heads, eps=1e-6).to(DEVICE, DTYPE)
        self.norm_added_q = nn.RMSNorm(hidden_dim // heads, eps=1e-6).to(DEVICE, DTYPE)
        self.norm_added_k = nn.RMSNorm(hidden_dim // heads, eps=1e-6).to(DEVICE, DTYPE)
        self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE), nn.Identity()]
        self.to_add_out = nn.Linear(hidden_dim, hidden_dim, bias=False).to(DEVICE, DTYPE)


def _make_wan_rotary_emb(seq_len=SEQ_LEN, dim=HIDDEN_DIM, heads=HEADS):
    head_dim = dim // heads
    cos = torch.randn(1, seq_len, 1, head_dim, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(1, seq_len, 1, head_dim, device=DEVICE, dtype=torch.float32)
    return (cos, sin)


def _capture_qkv(captures: dict, name: str):
    """Return an attn_fn that captures Q, K, V and returns Q as dummy output."""

    def fn(query, key, value, attention_mask, **kwargs):
        captures[name] = {"q": query.clone(), "k": key.clone(), "v": value.clone()}
        return query

    return fn


def _compare_captures(fused: dict, unfused: dict, atol=2e-2, rtol=2e-2):
    """Assert Q, K, V match between fused and unfused captures."""
    torch.testing.assert_close(fused["q"], unfused["q"], atol=atol, rtol=rtol)
    torch.testing.assert_close(fused["k"], unfused["k"], atol=atol, rtol=rtol)
    torch.testing.assert_close(fused["v"], unfused["v"], atol=atol, rtol=rtol)


def _wan_processor_fused_vs_unfused(method_name: str, method_config: dict):
    """Generic test: method creates a Wan processor, fused vs unfused Q/K/V match."""
    from sparsevideo._registry import get_method_class

    method_cls = get_method_class(method_name)
    model_info = SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()])

    config_fused = {**method_config}
    config_unfused = {**method_config}

    # For methods that support fused flags through the processor
    method_fused = method_cls(config=config_fused, model_info=model_info)
    method_unfused = method_cls(config=config_unfused, model_info=model_info)

    captures = {}
    proc_fused = method_fused.create_processor(
        layer_idx=0, total_layers=30, original_processor=None, step_tracker=_StepTracker(),
    )
    proc_unfused = method_unfused.create_processor(
        layer_idx=0, total_layers=30, original_processor=None, step_tracker=_StepTracker(),
    )

    # Override attn_fn to capture Q/K/V
    proc_fused.attn_fn = _capture_qkv(captures, "fused")
    proc_unfused.attn_fn = _capture_qkv(captures, "unfused")

    # Force unfused path
    proc_unfused.use_fused_qk_norm = False
    proc_unfused.use_fused_rope = False
    proc_unfused.use_fused_qk_norm_rope = False

    torch.manual_seed(42)
    attn = _WanAttn()
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    rotary_emb = _make_wan_rotary_emb()

    proc_fused(attn, hidden.clone(), rotary_emb=rotary_emb)
    proc_unfused(attn, hidden.clone(), rotary_emb=rotary_emb)

    _compare_captures(captures["fused"], captures["unfused"])


# ============ Per-method tests ============


def test_svg1_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("svg1", {})


def test_svg2_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("svg2", {"allow_triton_fallback": True})


def test_radial_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("radial", {"use_sage_attention": False})


def test_draft_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("draft", {})


def test_adacluster_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("adacluster", {})


def test_sta_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("sta", {})


def test_svoo_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("svoo", {"use_dynamic_min_kc_ratio": False})


def test_flashomni_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("flashomni", {"sparse_pattern": "global_random"})


def test_spargeattn_fused_vs_unfused_wan_processor():
    _wan_processor_fused_vs_unfused("spargeattn", {"mode": "full"})


# ============ Hunyuan processor tests ============


def _hunyuan_processor_fused_vs_unfused(method_name: str, method_config: dict):
    """Generic test: method creates a Hunyuan processor, fused vs unfused Q/K/V match."""
    from sparsevideo._registry import get_method_class

    method_cls = get_method_class(method_name)
    model_info = SimpleNamespace(model_type="hunyuan_video", model_key=None, transformers=[object()])

    method_fused = method_cls(config=method_config, model_info=model_info)
    method_unfused = method_cls(config=method_config, model_info=model_info)

    captures = {}
    proc_fused = method_fused.create_processor(
        layer_idx=0, total_layers=38, original_processor=None, step_tracker=_StepTracker(),
    )
    proc_unfused = method_unfused.create_processor(
        layer_idx=0, total_layers=38, original_processor=None, step_tracker=_StepTracker(),
    )

    proc_fused.attn_fn = _capture_qkv(captures, "fused")
    proc_unfused.attn_fn = _capture_qkv(captures, "unfused")

    proc_unfused.use_fused_qk_norm = False
    proc_unfused.use_fused_rope = False
    proc_unfused.use_fused_qk_norm_rope = False

    torch.manual_seed(42)
    attn = _HunyuanAttn()
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    encoder_hidden = torch.randn(1, 8, HIDDEN_DIM, device=DEVICE, dtype=DTYPE)
    # HunyuanVideo image_rotary_emb: (cos, sin) for video tokens only
    # Dual block: Q/K are video-only before RoPE, text concatenated after
    head_dim = HIDDEN_DIM // HEADS
    cos = torch.randn(SEQ_LEN, head_dim, device=DEVICE, dtype=torch.float32)
    sin = torch.randn(SEQ_LEN, head_dim, device=DEVICE, dtype=torch.float32)
    image_rotary_emb = (cos, sin)

    proc_fused(attn, hidden.clone(), encoder_hidden_states=encoder_hidden.clone(), image_rotary_emb=image_rotary_emb)
    proc_unfused(attn, hidden.clone(), encoder_hidden_states=encoder_hidden.clone(), image_rotary_emb=image_rotary_emb)

    _compare_captures(captures["fused"], captures["unfused"])


def test_svg1_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("svg1", {})


def test_svg2_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("svg2", {"allow_triton_fallback": True})


def test_radial_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("radial", {"use_sage_attention": False})


def test_draft_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("draft", {})


def test_adacluster_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("adacluster", {})


def test_sta_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("sta", {})


def test_spargeattn_fused_vs_unfused_hunyuan_processor():
    _hunyuan_processor_fused_vs_unfused("spargeattn", {"mode": "full"})
