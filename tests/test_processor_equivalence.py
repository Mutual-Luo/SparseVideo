from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from sparsevideo.processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from sparsevideo.processors.wan import SparseWanAttnProcessor


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


def _dense_attn_fn(query, key, value, attention_mask, **kwargs):
    return dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )


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
