from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from sparsevideo.processors.hunyuan_video import (
    SparseHunyuanVideoAttnProcessor,
    _can_use_fused_qk_norm as _hunyuan_can_use_fused_qk_norm,
)
from sparsevideo.processors.wan import (
    SparseWanAttnProcessor,
    _can_use_fused_qk_norm as _wan_can_use_fused_qk_norm,
)


class _StepTracker:
    step = 0


class _WanAttention:
    def __init__(self, hidden_dim=16, heads=4):
        self.heads = heads
        self.fused_projections = False
        self.cross_attention_dim_head = None
        self.add_k_proj = None
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Identity()]


class _HunyuanAttention:
    def __init__(self, hidden_dim=16, heads=4):
        self.heads = heads
        self.add_q_proj = None
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Identity()]
        self.to_add_out = None


class _MetaWeightNorm(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, device="meta"))
        self.eps = 1e-6

    def forward(self, hidden_states):
        return hidden_states


def _dense_attn_fn(query, key, value, attention_mask, **kwargs):
    return dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )


def test_fused_qk_norm_is_disabled_for_offloaded_meta_weights():
    norm_q = _MetaWeightNorm()
    norm_k = _MetaWeightNorm()

    assert not _wan_can_use_fused_qk_norm(norm_q, norm_k)
    assert not _hunyuan_can_use_fused_qk_norm(norm_q, norm_k)


def test_sparse_wan_processor_matches_diffusers_dense_processor():
    torch.manual_seed(0)
    attn = _WanAttention()
    hidden_states = torch.randn(2, 7, 16)

    expected = WanAttnProcessor()(attn, hidden_states)
    actual = SparseWanAttnProcessor(
        attn_fn=_dense_attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_wan_processor_matches_diffusers_dense_processor_with_rotary():
    torch.manual_seed(2)
    attn = _WanAttention()
    hidden_states = torch.randn(2, 7, 16)
    freqs = torch.randn(1, 7, 1, 4)
    rotary_emb = (freqs.cos(), freqs.sin())

    expected = WanAttnProcessor()(attn, hidden_states, rotary_emb=rotary_emb)
    actual = SparseWanAttnProcessor(
        attn_fn=_dense_attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, rotary_emb=rotary_emb)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_wan_processor_forwards_timestep_kwargs_to_method_attention():
    torch.manual_seed(3)
    attn = _WanAttention()
    hidden_states = torch.randn(1, 5, 16)
    timestep = torch.tensor([925])
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["timestep"] = kwargs.get("timestep")
        return torch.zeros_like(query)

    actual = SparseWanAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, timestep=timestep)

    assert calls["timestep"] is timestep
    assert actual.shape == hidden_states.shape


def test_sparse_hunyuan_processor_matches_diffusers_dense_processor():
    torch.manual_seed(1)
    attn = _HunyuanAttention()
    hidden_states = torch.randn(2, 7, 16)

    expected = HunyuanVideoAttnProcessor2_0()(attn, hidden_states)
    actual = SparseHunyuanVideoAttnProcessor(
        attn_fn=_dense_attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states)

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    assert actual[1] is None
    assert expected[1] is None


def test_hunyuan_rope_pytorch_fallback_matches_diffusers_full_dim_rotary():
    from diffusers.models.embeddings import apply_rotary_emb
    from sparsevideo.kernels.fused_norm_rope import _rope_hyvideo_pytorch

    torch.manual_seed(5)
    q = torch.randn(1, 6, 2, 4)
    k = torch.randn(1, 6, 2, 4)
    cos = torch.randn(4, 4)
    sin = torch.randn(4, 4)

    actual_q, actual_k = _rope_hyvideo_pytorch(q, k, cos, sin, txt_len=2)
    expected_q = torch.cat(
        [apply_rotary_emb(q[:, :4], (cos, sin), sequence_dim=1), q[:, 4:]],
        dim=1,
    )
    expected_k = torch.cat(
        [apply_rotary_emb(k[:, :4], (cos, sin), sequence_dim=1), k[:, 4:]],
        dim=1,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_hunyuan_rope_triton_fallback_matches_diffusers_full_dim_rotary(monkeypatch):
    from diffusers.models.embeddings import apply_rotary_emb
    from sparsevideo.kernels.fused_norm_rope import triton_rope_hyvideo_inplace

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")

    torch.manual_seed(6)
    q = torch.randn(1, 6, 2, 4, device="cuda")
    k = torch.randn(1, 6, 2, 4, device="cuda")
    cos = torch.randn(4, 4, device="cuda")
    sin = torch.randn(4, 4, device="cuda")

    actual_q, actual_k = triton_rope_hyvideo_inplace(q.clone(), k.clone(), cos, sin, txt_len=2)
    expected_q = torch.cat(
        [apply_rotary_emb(q[:, :4], (cos, sin), sequence_dim=1), q[:, 4:]],
        dim=1,
    )
    expected_k = torch.cat(
        [apply_rotary_emb(k[:, :4], (cos, sin), sequence_dim=1), k[:, 4:]],
        dim=1,
    )

    torch.testing.assert_close(actual_q, expected_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, atol=1e-5, rtol=1e-5)


def test_sparse_hunyuan_processor_forwards_timestep_kwargs_to_method_attention():
    torch.manual_seed(4)
    attn = _HunyuanAttention()
    hidden_states = torch.randn(1, 5, 16)
    timestep = torch.tensor([925])
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["timestep"] = kwargs.get("timestep")
        return torch.zeros_like(query)

    actual = SparseHunyuanVideoAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, timestep=timestep)

    assert calls["timestep"] is timestep
    assert actual[0].shape == hidden_states.shape
    assert actual[1] is None


def test_sparse_hunyuan_processor_forwards_flashomni_cache_kwargs_to_method_attention():
    torch.manual_seed(5)
    attn = _HunyuanAttention()
    hidden_states = torch.randn(1, 5, 16)
    cache_dic = {"cache": {}}
    current = {"step": 0}
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["key_is_tensor"] = torch.is_tensor(key)
        calls["key_shape"] = tuple(key.shape)
        calls["cache_dic"] = kwargs.get("cache_dic")
        calls["current"] = kwargs.get("current")
        return torch.zeros_like(query)

    SparseHunyuanVideoAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, cache_dic=cache_dic, current=current)

    assert calls["key_is_tensor"] is True
    assert calls["key_shape"] == (1, 5, 4, 4)
    assert calls["cache_dic"] is cache_dic
    assert calls["current"] is current


def test_sparse_hunyuan_processor_exposes_flashomni_cache_kwargs_for_diffusers_filtering():
    signature = inspect.signature(SparseHunyuanVideoAttnProcessor.__call__)

    assert "cache_dic" in signature.parameters
    assert "current" in signature.parameters


def test_sparse_hunyuan_processor_forwards_text_and_prompt_lengths_from_total_mask():
    torch.manual_seed(7)
    attn = _HunyuanAttention()
    hidden_states = torch.randn(1, 5, 16)
    encoder_hidden_states = torch.randn(1, 4, 16)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["query_len"] = query.shape[1]
        calls["text_len"] = kwargs.get("text_len")
        calls["prompt_length"] = kwargs.get("prompt_length")
        return torch.zeros_like(query)

    actual = SparseHunyuanVideoAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

    assert calls == {"query_len": 9, "text_len": 4, "prompt_length": 2}
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape
