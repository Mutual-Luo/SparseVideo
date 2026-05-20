from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import (
    AllegroAttnProcessor2_0,
    Attention,
    MochiAttention,
    MochiAttnProcessor2_0,
)
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXAttnProcessor2_0
from diffusers.models.transformers.transformer_easyanimate import EasyAnimateAttnProcessor2_0
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0
from diffusers.models.transformers.transformer_ltx import LTXAttention, LTXVideoAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from sparsevideo.processors.allegro import SparseAllegroAttnProcessor
from sparsevideo.processors.cogvideox import SparseCogVideoXAttnProcessor
from sparsevideo.processors.easyanimate import SparseEasyAnimateAttnProcessor
from sparsevideo.processors.hunyuan_video import (
    SparseHunyuanVideoAttnProcessor,
    _can_use_fused_qk_norm as _hunyuan_can_use_fused_qk_norm,
)
from sparsevideo.processors.ltx_video import SparseLTXVideoAttnProcessor
from sparsevideo.processors.mochi import SparseMochiAttnProcessor
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


class _CogVideoXAttention:
    def __init__(self, hidden_dim=16, heads=4):
        self.heads = heads
        self.is_cross_attention = False
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.to_out = [nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Identity()]

    def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
        return attention_mask


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


def _dense_video_first_sdpa(query, key, value, attention_mask, **kwargs):
    assert attention_mask is None
    out = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).contiguous(),
        key.permute(0, 2, 1, 3).contiguous(),
        value.permute(0, 2, 1, 3).contiguous(),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


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


def test_sparse_cogvideox_processor_matches_diffusers_dense_processor():
    torch.manual_seed(6)
    attn = _CogVideoXAttention()
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 3, 16)

    expected = CogVideoXAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
    )
    actual = SparseCogVideoXAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_cogvideox_processor_matches_diffusers_dense_processor_with_rotary():
    torch.manual_seed(8)
    attn = _CogVideoXAttention()
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 3, 16)
    cos = torch.randn(hidden_states.shape[1], 4)
    sin = torch.randn(hidden_states.shape[1], 4)
    image_rotary_emb = (cos, sin)

    expected = CogVideoXAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    actual = SparseCogVideoXAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_cogvideox_processor_passes_text_tail_to_method_attention():
    torch.manual_seed(7)
    attn = _CogVideoXAttention()
    hidden_states = torch.randn(1, 5, 16)
    encoder_hidden_states = torch.randn(1, 2, 16)
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["query_len"] = query.shape[1]
        calls["text_len"] = kwargs["text_len"]
        calls["prompt_length"] = kwargs["prompt_length"]
        return torch.zeros_like(query)

    actual = SparseCogVideoXAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states)

    assert calls == {"query_len": 7, "text_len": 2, "prompt_length": 2}
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape


def test_sparse_ltx_video_processor_matches_diffusers_dense_processor():
    torch.manual_seed(9)
    attn = LTXAttention(query_dim=16, heads=4, kv_heads=4, dim_head=4)
    hidden_states = torch.randn(2, 7, 16)

    expected = LTXVideoAttnProcessor()(attn, hidden_states)
    actual = SparseLTXVideoAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_ltx_video_processor_matches_diffusers_dense_processor_with_rotary():
    torch.manual_seed(10)
    attn = LTXAttention(query_dim=16, heads=4, kv_heads=4, dim_head=4)
    hidden_states = torch.randn(2, 7, 16)
    cos = torch.randn(1, hidden_states.shape[1], 16)
    sin = torch.randn(1, hidden_states.shape[1], 16)
    image_rotary_emb = (cos, sin)

    expected = LTXVideoAttnProcessor()(
        attn,
        hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    actual = SparseLTXVideoAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_ltx_video_processor_rejects_cross_attention():
    attn = LTXAttention(query_dim=16, heads=4, kv_heads=4, dim_head=4)
    processor = SparseLTXVideoAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )

    with pytest.raises(RuntimeError, match="text cross-attention must remain dense"):
        processor(
            attn,
            torch.randn(1, 5, 16),
            encoder_hidden_states=torch.randn(1, 3, 16),
        )


def test_sparse_allegro_processor_matches_diffusers_dense_processor():
    torch.manual_seed(19)
    attn = Attention(
        query_dim=16,
        heads=4,
        dim_head=4,
        bias=True,
        cross_attention_dim=None,
        processor=AllegroAttnProcessor2_0(),
    )
    hidden_states = torch.randn(2, 7, 16)

    expected = AllegroAttnProcessor2_0()(attn, hidden_states)
    actual = SparseAllegroAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_sparse_allegro_processor_rejects_cross_attention():
    attn = Attention(
        query_dim=16,
        heads=4,
        dim_head=4,
        bias=True,
        cross_attention_dim=None,
        processor=AllegroAttnProcessor2_0(),
    )
    processor = SparseAllegroAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )

    with pytest.raises(RuntimeError, match="text cross-attention must remain dense"):
        processor(
            attn,
            torch.randn(1, 5, 16),
            encoder_hidden_states=torch.randn(1, 3, 16),
        )


def test_sparse_mochi_processor_matches_diffusers_dense_processor():
    torch.manual_seed(11)
    attn = MochiAttention(
        query_dim=16,
        added_kv_proj_dim=12,
        processor=MochiAttnProcessor2_0(),
        heads=4,
        dim_head=4,
        out_context_dim=12,
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

    expected = MochiAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
    )
    actual = SparseMochiAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_mochi_processor_matches_diffusers_dense_processor_with_rotary():
    torch.manual_seed(12)
    attn = MochiAttention(
        query_dim=16,
        added_kv_proj_dim=12,
        processor=MochiAttnProcessor2_0(),
        heads=4,
        dim_head=4,
        out_context_dim=12,
    )
    hidden_states = torch.randn(1, 7, 16)
    encoder_hidden_states = torch.randn(1, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    cos = torch.randn(1, 7, 1, 2)
    sin = torch.randn(1, 7, 1, 2)
    image_rotary_emb = (cos, sin)

    expected = MochiAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    actual = SparseMochiAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        image_rotary_emb=image_rotary_emb,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_mochi_processor_passes_valid_text_tail_to_method_attention():
    torch.manual_seed(13)
    attn = MochiAttention(
        query_dim=16,
        added_kv_proj_dim=12,
        processor=MochiAttnProcessor2_0(),
        heads=4,
        dim_head=4,
        out_context_dim=12,
    )
    hidden_states = torch.randn(1, 7, 16)
    encoder_hidden_states = torch.randn(1, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["query_len"] = query.shape[1]
        calls["text_len"] = kwargs["text_len"]
        calls["prompt_length"] = kwargs["prompt_length"]
        return torch.zeros_like(query)

    actual = SparseMochiAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states, attention_mask)

    assert calls == {"query_len": 10, "text_len": 3, "prompt_length": 3}
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape


def test_sparse_mochi_processor_batches_equal_prompt_masks():
    torch.manual_seed(14)
    attn = MochiAttention(
        query_dim=16,
        added_kv_proj_dim=12,
        processor=MochiAttnProcessor2_0(),
        heads=4,
        dim_head=4,
        out_context_dim=12,
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    calls = []

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls.append(
            {
                "batch": query.shape[0],
                "query_len": query.shape[1],
                "text_len": kwargs["text_len"],
                "cache_key_suffix": kwargs.get("cache_key_suffix"),
            }
        )
        return torch.zeros_like(query)

    actual = SparseMochiAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states, attention_mask)

    assert calls == [{"batch": 2, "query_len": 10, "text_len": 3, "cache_key_suffix": None}]
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape


def test_sparse_mochi_processor_isolates_variable_prompt_masks():
    torch.manual_seed(15)
    attn = MochiAttention(
        query_dim=16,
        added_kv_proj_dim=12,
        processor=MochiAttnProcessor2_0(),
        heads=4,
        dim_head=4,
        out_context_dim=12,
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 5, 12)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    calls = []

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls.append(
            {
                "batch": query.shape[0],
                "query_len": query.shape[1],
                "text_len": kwargs["text_len"],
                "cache_key_suffix": kwargs.get("cache_key_suffix"),
            }
        )
        return torch.zeros_like(query)

    actual = SparseMochiAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states, attention_mask)

    assert calls == [
        {"batch": 1, "query_len": 10, "text_len": 3, "cache_key_suffix": 0},
        {"batch": 1, "query_len": 9, "text_len": 2, "cache_key_suffix": 1},
    ]
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape


def test_sparse_easyanimate_processor_matches_dense_added_projection_processor():
    torch.manual_seed(16)
    attn = Attention(
        query_dim=16,
        dim_head=4,
        heads=4,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=True,
        added_proj_bias=True,
        added_kv_proj_dim=16,
        context_pre_only=False,
        processor=EasyAnimateAttnProcessor2_0(),
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 3, 16)

    expected = EasyAnimateAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
    )
    actual = SparseEasyAnimateAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_easyanimate_processor_matches_dense_concat_processor():
    torch.manual_seed(17)
    attn = Attention(
        query_dim=16,
        dim_head=4,
        heads=4,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=True,
        added_proj_bias=True,
        added_kv_proj_dim=None,
        context_pre_only=None,
        processor=EasyAnimateAttnProcessor2_0(),
    )
    hidden_states = torch.randn(2, 7, 16)
    encoder_hidden_states = torch.randn(2, 3, 16)

    expected = EasyAnimateAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
    )
    actual = SparseEasyAnimateAttnProcessor(
        attn_fn=_dense_video_first_sdpa,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(
        attn,
        hidden_states,
        encoder_hidden_states,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_sparse_easyanimate_processor_passes_text_tail_to_method_attention():
    torch.manual_seed(18)
    attn = Attention(
        query_dim=16,
        dim_head=4,
        heads=4,
        qk_norm="layer_norm",
        eps=1e-6,
        bias=True,
        added_proj_bias=True,
        added_kv_proj_dim=16,
        context_pre_only=False,
        processor=EasyAnimateAttnProcessor2_0(),
    )
    hidden_states = torch.randn(1, 7, 16)
    encoder_hidden_states = torch.randn(1, 3, 16)
    calls = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        calls["query_len"] = query.shape[1]
        calls["text_len"] = kwargs["text_len"]
        calls["prompt_length"] = kwargs["prompt_length"]
        return torch.zeros_like(query)

    actual = SparseEasyAnimateAttnProcessor(
        attn_fn=attn_fn,
        layer_idx=0,
        step_tracker=_StepTracker(),
    )(attn, hidden_states, encoder_hidden_states)

    assert calls == {"query_len": 10, "text_len": 3, "prompt_length": 3}
    assert actual[0].shape == hidden_states.shape
    assert actual[1].shape == encoder_hidden_states.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svg2_sparse_attention_accepts_cfg_batch(monkeypatch):
    import sparsevideo.kernels.flashinfer_block_sparse as flashinfer_block_sparse
    from sparsevideo.methods.svg2.method import _svg2_attention

    monkeypatch.setattr(flashinfer_block_sparse, "HAS_FLASHINFER", False)

    torch.manual_seed(9)
    query = torch.randn(2, 32, 2, 16, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
    }

    actual = _svg2_attention(
        query,
        key,
        value,
        top_p_kmeans=0.9,
        min_kc_ratio=0.0,
        num_q_centroids=4,
        num_k_centroids=8,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        allow_triton_fallback=True,
        model_type="cogvideox",
    )

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svg2_sparse_attention_accepts_allegro_head_dim(monkeypatch):
    import sparsevideo.kernels.flashinfer_block_sparse as flashinfer_block_sparse
    from sparsevideo.methods.svg2.method import _svg2_attention

    monkeypatch.setattr(flashinfer_block_sparse, "HAS_FLASHINFER", False)

    torch.manual_seed(20)
    query = torch.randn(1, 32, 2, 96, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
    }

    actual = _svg2_attention(
        query,
        key,
        value,
        top_p_kmeans=0.9,
        min_kc_ratio=0.0,
        num_q_centroids=4,
        num_k_centroids=8,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        allow_triton_fallback=True,
        model_type="allegro",
    )

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_sparse_attention_accepts_cfg_batch():
    from sparsevideo.methods.svoo import config as svoo_config
    from sparsevideo.methods.svoo.ops import svoo_attention

    torch.manual_seed(10)
    query = torch.randn(2, 32, 2, 16, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cfg = dict(svoo_config.CONFIG_DEFAULTS)
    cfg.update(
        {
            "sparse_backend": "triton",
            "num_q_centroids": 4,
            "num_k_centroids": 8,
            "kmeans_iter_init": 1,
            "kmeans_iter_step": 1,
            "use_dynamic_min_kc_ratio": False,
            "use_global_constraints": False,
            "use_svoo": True,
        }
    )
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
        "prev_q_profile_centroids": None,
        "cached_clustering": None,
        "sparsity_lookup": None,
        "sparsity_lookup_path": None,
        "last_logged_sparsity_step": None,
    }

    actual = svoo_attention(
        query,
        key,
        value,
        cfg,
        state,
        current_step=0,
        layer_idx=0,
        text_len=2,
        prompt_length=2,
        model_type="cogvideox",
    )

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


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
