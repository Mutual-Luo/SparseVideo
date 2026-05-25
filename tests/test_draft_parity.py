from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparsevideo.methods.draft import DraftMethod
from sparsevideo.methods.draft.method import (
    _attention_percentile_mask_headwise,
    _draft_attention,
    _draft_cu_seqlens,
    _draft_dense_attention,
    _draft_mit_path,
    _draft_triton_path,
    _crop_draft_video_canvas,
    _fixed_block_sizes,
    _generate_reorg_restore_indices,
    _pad_draft_video_canvas,
    _sample_qk_attention_2d,
    _validate_upstream_draft_layout,
)


def _load_upstream_draft_module(monkeypatch, relative_path, module_name, block_sparse_func):
    repo = Path(__file__).resolve().parents[1]
    block_sparse_module = types.ModuleType("block_sparse_attn")
    block_sparse_module.block_sparse_attn_func = block_sparse_func
    monkeypatch.setitem(sys.modules, "block_sparse_attn", block_sparse_module)

    path = repo / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draft_capture_kernel(calls):
    def fake_block_sparse_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        **kwargs,
    ):
        calls.append(
            {
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "cu_seqlens_q": cu_seqlens_q.detach().clone(),
                "cu_seqlens_k": cu_seqlens_k.detach().clone(),
                "head_mask_type": head_mask_type.detach().clone(),
                "streaming_info": streaming_info,
                "base_blockmask": base_blockmask.detach().clone(),
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "p_dropout": p_dropout,
                "kwargs": dict(kwargs),
            }
        )
        return q

    return fake_block_sparse_attn_func


def _assert_draft_kernel_calls_match(local_call, upstream_call):
    for key in ("q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "head_mask_type", "base_blockmask"):
        assert torch.equal(local_call[key], upstream_call[key]), key
    for key in ("streaming_info", "max_seqlen_q", "max_seqlen_k", "p_dropout"):
        assert local_call[key] == upstream_call[key]
    for key in ("deterministic", "softmax_scale", "is_causal", "exact_streaming", "return_attn_probs"):
        assert local_call["kwargs"].get(key) == upstream_call["kwargs"].get(key)


def test_draft_dense_gate_uses_upstream_flash_attn_varlen(monkeypatch):
    import sparsevideo.methods.draft.method as draft_method

    calls = {}

    def fake_flash_attn(q, k, v, cu_q, cu_k, max_q, max_k):
        calls["q_shape"] = q.shape
        calls["k_shape"] = k.shape
        calls["cu_q"] = cu_q.tolist()
        calls["cu_k"] = cu_k.tolist()
        calls["max_q"] = max_q
        calls["max_k"] = max_k
        return q

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(draft_method, "_load_flash_attn_varlen_func", lambda: fake_flash_attn)

    query = torch.randn(1, 6, 2, 3)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    out = _draft_dense_attention(
        query,
        query,
        query,
        attention_mask=attention_mask,
        model_type="hunyuan_video",
        text_len=2,
    )

    assert calls == {
        "q_shape": (6, 2, 3),
        "k_shape": (6, 2, 3),
        "cu_q": [0, 4, 6],
        "cu_k": [0, 4, 6],
        "max_q": 6,
        "max_k": 6,
    }
    assert out.shape == query.shape


def test_draft_dense_cross_attention_keeps_separate_q_and_kv_lengths(monkeypatch):
    import sparsevideo.methods.draft.method as draft_method

    calls = {}

    def fake_flash_attn(q, k, v, cu_q, cu_k, max_q, max_k):
        calls["q_shape"] = q.shape
        calls["k_shape"] = k.shape
        calls["cu_q"] = cu_q.tolist()
        calls["cu_k"] = cu_k.tolist()
        calls["max_q"] = max_q
        calls["max_k"] = max_k
        return q

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(draft_method, "_load_flash_attn_varlen_func", lambda: fake_flash_attn)

    query = torch.randn(1, 4, 2, 3)
    key = torch.randn(1, 7, 2, 3)
    out = _draft_dense_attention(query, key, key, model_type="wan")

    assert calls == {
        "q_shape": (4, 2, 3),
        "k_shape": (7, 2, 3),
        "cu_q": [0, 4],
        "cu_k": [0, 7],
        "max_q": 4,
        "max_k": 7,
    }
    assert out.shape == query.shape


def test_draft_reorg_restore_indices_match_upstream_pattern():
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=2,
        pool_w=2,
        latent_h=4,
        latent_w=6,
        visual_len=24,
        text_len=3,
        device=torch.device("cpu"),
    )

    assert reorg_idx.tolist() == [
        0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11,
        12, 13, 18, 19, 14, 15, 20, 21, 16, 17, 22, 23,
        24, 25, 26,
    ]

    original = torch.arange(27)
    reordered = original.index_select(0, reorg_idx)
    restored = reordered.index_select(0, restore_idx)
    assert torch.equal(restored, original)


def test_draft_percentile_mask_is_head_global_like_upstream():
    attn = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    mask = _attention_percentile_mask_headwise(attn, keep_ratio=0.5)

    assert mask.tolist() == [[[[False, True], [True, True]]]]


def test_draft_dense_warmup_ratio_is_only_step_gate(monkeypatch):
    calls = []

    def fake_dense(query, key, value, **kwargs):
        calls.append("dense")
        return query

    def fake_sparse(query, key, value, **kwargs):
        calls.append("sparse")
        return query

    monkeypatch.setattr("sparsevideo.methods.draft.method._draft_dense_attention", fake_dense)
    monkeypatch.setattr("sparsevideo.methods.draft.method._draft_attention", fake_sparse)

    method = DraftMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="wan", model_key=None),
    )
    processor = method.create_processor(
        layer_idx=5,
        total_layers=30,
        original_processor=None,
        step_tracker=SimpleNamespace(step=20, timestep=999),
    )
    query = torch.randn(1, 128, 2, 64)

    processor.attn_fn(query, query, query, None)

    assert calls == ["sparse"]


def test_draft_runtime_layout_gate_rejects_non_upstream_hunyuan_shapes():
    _validate_upstream_draft_layout(
        33 * 48 * 80,
        frame_h=48,
        frame_w=80,
        pool_h=8,
        pool_w=16,
        model_type="hunyuan_video",
        text_len=256,
    )

    with pytest.raises(ValueError, match="129-frame 1280x768"):
        _validate_upstream_draft_layout(
            21 * 48 * 80,
            frame_h=48,
            frame_w=80,
            pool_h=8,
            pool_w=16,
            model_type="hunyuan_video",
            text_len=256,
        )

    with pytest.raises(ValueError, match="256-token text tail"):
        _validate_upstream_draft_layout(
            33 * 48 * 80,
            frame_h=48,
            frame_w=80,
            pool_h=8,
            pool_w=16,
            model_type="hunyuan_video",
            text_len=128,
        )


def test_draft_fixed_block_sizes_cover_text_tail():
    sizes = _fixed_block_sizes(
        total_len=27,
        block_dim=12,
        block_num=3,
        batch_heads=2,
        device=torch.device("cpu"),
    )

    assert sizes.dtype == torch.int32
    assert sizes.tolist() == [[12, 12, 3], [12, 12, 3]]


def test_draft_hunyuan_sparse_path_accepts_attention_mask_like_upstream(monkeypatch):
    calls = []

    def fake_draft_attention(query, key, value, **kwargs):
        calls.append(kwargs)
        return torch.empty_like(query)

    monkeypatch.setattr("sparsevideo.methods.draft.method._draft_attention", fake_draft_attention)

    method = DraftMethod(
        config={},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=2,
        total_layers=40,
        original_processor=None,
        step_tracker=SimpleNamespace(timestep=900),
    )
    query = torch.randn(1, 10, 2, 4)
    attention_mask = torch.ones(1, 256)

    out = processor.attn_fn(query, query, query, attention_mask, text_len=256)

    assert out.shape == query.shape
    assert calls[0]["model_type"] == "hunyuan_video"
    assert calls[0]["text_len"] == 256


def test_draft_wan_sparse_path_rejects_attention_mask(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.draft.method._draft_attention",
        lambda query, key, value, **kwargs: torch.empty_like(query),
    )
    method = DraftMethod(
        config={},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=1,
        total_layers=40,
        original_processor=None,
        step_tracker=SimpleNamespace(timestep=900),
    )
    query = torch.randn(1, 10, 2, 4)

    with pytest.raises(RuntimeError, match="without an attention mask"):
        processor.attn_fn(query, query, query, torch.ones(1, 10))


def test_draft_rejects_configured_layout_mismatches_before_sparse_kernel():
    query = torch.randn(1, 24, 1, 4)

    with pytest.raises(RuntimeError, match="visual_len config mismatch"):
        _draft_attention(
            query,
            query,
            query,
            sparsity_ratio=0.75,
            pool_h=2,
            pool_w=2,
            model_type="wan",
            visual_len=25,
        )

    with pytest.raises(RuntimeError, match="batch_size config mismatch"):
        _draft_attention(
            query,
            query,
            query,
            sparsity_ratio=0.75,
            pool_h=2,
            pool_w=2,
            model_type="wan",
            batch_size=2,
        )


def test_draft_runtime_uses_owned_triton_block_sparse_not_flashinfer(monkeypatch):
    calls = []

    def fake_block_sparse_attention(q_sorted, k_sorted, v_sorted, q_sizes, k_sizes, dynamic_map, scale):
        calls.append(
            {
                "q_shape": tuple(q_sorted.shape),
                "q_sizes": q_sizes.clone(),
                "k_sizes": k_sizes.clone(),
                "dynamic_map": dynamic_map.clone(),
                "scale": scale,
            }
        )
        return q_sorted

    monkeypatch.setattr(
        "sparsevideo.kernels.block_sparse_attn.block_sparse_attention",
        fake_block_sparse_attention,
    )

    query = torch.arange(1 * 24 * 1 * 16, dtype=torch.float32).reshape(1, 24, 1, 16)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=2,
        pool_w=2,
        latent_h=4,
        latent_w=6,
        visual_len=24,
        text_len=0,
        device=torch.device("cpu"),
    )

    out = _draft_triton_path(
        query,
        query,
        query,
        B=1,
        N=24,
        H=1,
        D=16,
        scale=16 ** -0.5,
        context_len=0,
        video_end=24,
        T=1,
        frame_h=4,
        frame_w=6,
        frame_size=24,
        sparsity_ratio=0.75,
        pool_h=2,
        pool_w=2,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert calls
    assert calls[0]["q_shape"] == (1, 24, 16)
    assert calls[0]["q_sizes"].dtype == torch.long
    assert calls[0]["k_sizes"].dtype == torch.long
    assert calls[0]["dynamic_map"].dtype == torch.bool
    assert torch.equal(out, query)


def test_draft_mit_path_calls_upstream_block_sparse_interface(monkeypatch):
    calls = []

    def fake_load_block_sparse_attn_func():
        def fake_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            max_seqlen_q,
            max_seqlen_k,
            p_dropout,
            **kwargs,
        ):
            calls.append(
                {
                    "q_shape": tuple(q.shape),
                    "cu_seqlens_q": cu_seqlens_q.clone(),
                    "head_mask_type": head_mask_type.clone(),
                    "streaming_info": streaming_info,
                    "base_blockmask": base_blockmask.clone(),
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "p_dropout": p_dropout,
                    "kwargs": kwargs,
                }
            )
            return q

        return fake_func

    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        fake_load_block_sparse_attn_func,
    )

    query = torch.arange(2 * 128 * 1 * 128, dtype=torch.float32).reshape(2, 128, 1, 128)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=8,
        pool_w=16,
        latent_h=8,
        latent_w=16,
        visual_len=128,
        text_len=0,
        device=torch.device("cpu"),
    )

    out = _draft_mit_path(
        query,
        query,
        query,
        B=2,
        N=128,
        H=1,
        D=128,
        context_len=0,
        video_end=128,
        T=1,
        frame_h=8,
        frame_w=16,
        frame_size=128,
        sparsity_ratio=0.75,
        pool_h=8,
        pool_w=16,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert calls
    call = calls[0]
    assert call["q_shape"] == (256, 1, 128)
    assert call["cu_seqlens_q"].tolist() == [0, 128, 256]
    assert call["head_mask_type"].tolist() == [1]
    assert call["streaming_info"] is None
    assert call["base_blockmask"].shape == (2, 1, 1, 1)
    assert call["base_blockmask"].dtype == torch.bool
    assert call["max_seqlen_q"] == 128
    assert call["max_seqlen_k"] == 128
    assert call["p_dropout"] == 0.0
    assert call["kwargs"]["is_causal"] is False
    assert call["kwargs"]["return_attn_probs"] is False
    assert torch.equal(out, query)


def test_draft_mit_path_pads_sub_128_head_dim_preserving_scale(monkeypatch):
    calls = []

    def fake_load_block_sparse_attn_func():
        def fake_func(q, k, v, *args, **kwargs):
            calls.append({"q_shape": tuple(q.shape), "kwargs": kwargs})
            return q

        return fake_func

    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        fake_load_block_sparse_attn_func,
    )

    query = torch.arange(1 * 128 * 1 * 96, dtype=torch.float32).reshape(1, 128, 1, 96)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=8,
        pool_w=16,
        latent_h=8,
        latent_w=16,
        visual_len=128,
        text_len=0,
        device=torch.device("cpu"),
    )

    out = _draft_mit_path(
        query,
        query,
        query,
        B=1,
        N=128,
        H=1,
        D=96,
        context_len=0,
        video_end=128,
        T=1,
        frame_h=8,
        frame_w=16,
        frame_size=128,
        sparsity_ratio=0.75,
        pool_h=8,
        pool_w=16,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert calls[0]["q_shape"] == (128, 1, 128)
    assert calls[0]["kwargs"]["softmax_scale"] == pytest.approx(96 ** -0.5)
    assert out.shape == query.shape
    assert torch.equal(out, query)


def test_draft_canvas_padding_round_trips_video_and_text_tail():
    query = torch.arange(1 * 14 * 1 * 2, dtype=torch.float32).reshape(1, 14, 1, 2)
    key = query + 1000
    value = query + 2000

    padded_query, padded_key, padded_value = _pad_draft_video_canvas(
        query,
        key,
        value,
        context_len=0,
        video_end=12,
        tail_len=2,
        T=1,
        frame_h=3,
        frame_w=4,
        canvas_h=4,
        canvas_w=8,
    )

    assert padded_query.shape == (1, 34, 1, 2)

    for padded, original in (
        (padded_query, query),
        (padded_key, key),
        (padded_value, value),
    ):
        cropped = _crop_draft_video_canvas(
            padded,
            context_len=0,
            tail_len=2,
            T=1,
            frame_h=3,
            frame_w=4,
            canvas_h=4,
            canvas_w=8,
        )
        assert torch.equal(cropped, original)


def test_draft_mit_path_matches_upstream_forward_kernel_inputs(monkeypatch):
    upstream_calls = []
    upstream = _load_upstream_draft_module(
        monkeypatch,
        "training_free/draft-attention/draft_attention.py",
        "sparsevideo_test_upstream_draft_attention",
        _draft_capture_kernel(upstream_calls),
    )
    local_calls = []
    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        lambda: _draft_capture_kernel(local_calls),
    )

    torch.manual_seed(0)
    B, T, frame_h, frame_w, H, D = 1, 1, 16, 32, 2, 128
    N = T * frame_h * frame_w
    pool_h, pool_w = 8, 16
    query = torch.randn(B, N, H, D)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, N], dtype=torch.int32)

    upstream_attention = upstream.Draft_Attention(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=N,
        text_len=0,
        sparsity_ratio=0.75,
    )
    upstream_out = upstream_attention(
        query.reshape(B * N, H, D),
        key.reshape(B * N, H, D),
        value.reshape(B * N, H, D),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=N,
        max_seqlen_kv=N,
        batch_size=B,
    )

    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=N,
        text_len=0,
        device=torch.device("cpu"),
    )
    local_out = _draft_mit_path(
        query,
        key,
        value,
        B=B,
        N=N,
        H=H,
        D=D,
        context_len=0,
        video_end=N,
        T=T,
        frame_h=frame_h,
        frame_w=frame_w,
        frame_size=frame_h * frame_w,
        sparsity_ratio=0.75,
        pool_h=pool_h,
        pool_w=pool_w,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert torch.equal(local_out, upstream_out)
    assert len(local_calls) == len(upstream_calls) == 1
    _assert_draft_kernel_calls_match(local_calls[0], upstream_calls[0])


def test_draft_mit_path_matches_upstream_classifier_free_guidance_kernel_inputs(monkeypatch):
    upstream_calls = []
    upstream = _load_upstream_draft_module(
        monkeypatch,
        "training_free/draft-attention/draft_attention_classifier_free_guidance.py",
        "sparsevideo_test_upstream_draft_attention_cfg",
        _draft_capture_kernel(upstream_calls),
    )
    local_calls = []
    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        lambda: _draft_capture_kernel(local_calls),
    )

    torch.manual_seed(1)
    B, T, frame_h, frame_w, H, D = 2, 1, 16, 32, 2, 128
    visual_len = T * frame_h * frame_w
    text_len = 4
    N = visual_len + text_len
    pool_h, pool_w = 8, 16
    query = torch.randn(B, N, H, D)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    cu_seqlens = _draft_cu_seqlens(
        attention_mask=attention_mask,
        batch_size=B,
        total_len=N,
        video_len=visual_len,
        text_len=text_len,
        device=torch.device("cpu"),
    )

    upstream_attention = upstream.Draft_Attention(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=visual_len,
        text_len=text_len,
        sparsity_ratio=0.9,
        batch_size=B,
    )
    upstream_out = upstream_attention(
        query.reshape(B * N, H, D),
        key.reshape(B * N, H, D),
        value.reshape(B * N, H, D),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=N,
        max_seqlen_kv=N,
        batch_size=B,
    )

    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=visual_len,
        text_len=text_len,
        device=torch.device("cpu"),
    )
    local_out = _draft_mit_path(
        query,
        key,
        value,
        B=B,
        N=N,
        H=H,
        D=D,
        context_len=0,
        video_end=visual_len,
        T=T,
        frame_h=frame_h,
        frame_w=frame_w,
        frame_size=frame_h * frame_w,
        sparsity_ratio=0.9,
        pool_h=pool_h,
        pool_w=pool_w,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
        attention_mask=attention_mask,
    )

    assert torch.equal(local_out, upstream_out)
    assert len(local_calls) == len(upstream_calls) == 1
    _assert_draft_kernel_calls_match(local_calls[0], upstream_calls[0])
    assert upstream_calls[0]["kwargs"]["m_block_dim"] == 128
    assert upstream_calls[0]["kwargs"]["n_block_dim"] == 128


def test_draft_block_sparse_build_defaults_to_full_upstream_extension():
    repo = Path(__file__).resolve().parents[1]
    native_root = repo / "src/sparsevideo/kernels/native/draft_block_sparse"

    assert 'os.getenv("BLOCK_SPARSE_ATTN_BUILD_MODE", "full")' in (
        native_root / "setup.py"
    ).read_text(encoding="utf-8")
    assert 'BLOCK_SPARSE_ATTN_BUILD_MODE="${BLOCK_SPARSE_ATTN_BUILD_MODE:-full}"' in (
        native_root / "setup.sh"
    ).read_text(encoding="utf-8")
    assert "BLOCK_SPARSE_ATTN_BUILD_MODE=full" in (
        native_root / "README.md"
    ).read_text(encoding="utf-8")


def test_draft_mit_path_uses_hunyuan_text_mask_cu_seqlens_like_upstream(monkeypatch):
    calls = []

    def fake_load_block_sparse_attn_func():
        def fake_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            max_seqlen_q,
            max_seqlen_k,
            p_dropout,
            **kwargs,
        ):
            calls.append(
                {
                    "cu_seqlens_q": cu_seqlens_q.clone(),
                    "cu_seqlens_k": cu_seqlens_k.clone(),
                    "base_blockmask": base_blockmask.clone(),
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                }
            )
            return q

        return fake_func

    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        fake_load_block_sparse_attn_func,
    )

    query = torch.arange(1 * 132 * 1 * 128, dtype=torch.float32).reshape(1, 132, 1, 128)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=8,
        pool_w=16,
        latent_h=8,
        latent_w=16,
        visual_len=128,
        text_len=4,
        device=torch.device("cpu"),
    )
    attention_mask = torch.tensor([[1] * 128 + [1, 1, 0, 0]])

    out = _draft_mit_path(
        query,
        query,
        query,
        B=1,
        N=132,
        H=1,
        D=128,
        context_len=0,
        video_end=128,
        T=1,
        frame_h=8,
        frame_w=16,
        frame_size=128,
        sparsity_ratio=0.9,
        pool_h=8,
        pool_w=16,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
        attention_mask=attention_mask,
    )

    assert calls
    call = calls[0]
    assert call["cu_seqlens_q"].tolist() == [0, 130, 132]
    assert torch.equal(call["cu_seqlens_k"], call["cu_seqlens_q"])
    assert call["base_blockmask"].shape == (2, 1, 2, 2)
    assert call["max_seqlen_q"] == 132
    assert call["max_seqlen_k"] == 132
    assert torch.equal(out, query)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_mit_native_extension_executes_wan_upstream_layout_cuda():
    from sparsevideo.kernels.draft_block_sparse_runtime import load_block_sparse_attn_func

    block_sparse_attn_func = load_block_sparse_attn_func()
    assert block_sparse_attn_func.__module__ == "block_sparse_attn.block_sparse_attn_interface"

    B, T, frame_h, frame_w, H, D = 1, 21, 32, 48, 1, 128
    N = T * frame_h * frame_w
    pool_h, pool_w = 8, 16
    query = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=N,
        text_len=0,
        device=query.device,
    )

    out = _draft_mit_path(
        query,
        key,
        value,
        B=B,
        N=N,
        H=H,
        D=D,
        context_len=0,
        video_end=N,
        T=T,
        frame_h=frame_h,
        frame_w=frame_w,
        frame_size=frame_h * frame_w,
        sparsity_ratio=0.75,
        pool_h=pool_h,
        pool_w=pool_w,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )
    torch.cuda.synchronize()

    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.isfinite(out).all()


def test_draft_mit_path_rejects_non_128_block_layout(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.kernels.draft_block_sparse_runtime.load_block_sparse_attn_func",
        lambda: (lambda *args, **kwargs: pytest.fail("kernel should not be called for a non-128 block layout")),
    )

    query = torch.randn(1, 24, 1, 128)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=2,
        pool_w=2,
        latent_h=4,
        latent_w=6,
        visual_len=24,
        text_len=0,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="128x128 blocks"):
        _draft_mit_path(
            query,
            query,
            query,
            B=1,
            N=24,
            H=1,
            D=128,
            context_len=0,
            video_end=24,
            T=1,
            frame_h=4,
            frame_w=6,
            frame_size=24,
            sparsity_ratio=0.75,
            pool_h=2,
            pool_w=2,
            reorg_idx=reorg_idx,
            restore_idx=restore_idx,
        )


def test_draft_block_sparse_layout_matches_upstream_with_text_tail(monkeypatch):
    calls = []

    def fake_block_sparse_attention(q_sorted, k_sorted, v_sorted, q_sizes, k_sizes, dynamic_map, scale):
        calls.append(
            {
                "q_sizes": q_sizes.clone(),
                "k_sizes": k_sizes.clone(),
                "dynamic_map": dynamic_map.clone(),
            }
        )
        return q_sorted

    monkeypatch.setattr(
        "sparsevideo.kernels.block_sparse_attn.block_sparse_attention",
        fake_block_sparse_attention,
    )

    query = torch.arange(1 * 27 * 1 * 16, dtype=torch.float32).reshape(1, 27, 1, 16)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=2,
        pool_w=2,
        latent_h=4,
        latent_w=6,
        visual_len=24,
        text_len=3,
        device=torch.device("cpu"),
    )

    out = _draft_triton_path(
        query,
        query,
        query,
        B=1,
        N=27,
        H=1,
        D=16,
        scale=16 ** -0.5,
        context_len=0,
        video_end=24,
        T=1,
        frame_h=4,
        frame_w=6,
        frame_size=24,
        sparsity_ratio=0.75,
        pool_h=2,
        pool_w=2,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert calls
    call = calls[0]
    assert call["q_sizes"].tolist() == [[4, 4, 4, 4, 4, 4, 3]]
    assert call["k_sizes"].tolist() == [[4, 4, 4, 4, 4, 4, 3]]
    assert call["dynamic_map"].shape == (1, 7, 7)

    sampled = _sample_qk_attention_2d(
        query[:, :24],
        query[:, :24],
        frame_h=4,
        frame_w=6,
        pool_h=2,
        pool_w=2,
    )
    expected_visual = _attention_percentile_mask_headwise(sampled, keep_ratio=0.25).reshape(1, 6, 6)

    assert torch.equal(call["dynamic_map"][:, :6, :6], expected_visual)
    assert call["dynamic_map"][:, 6, :].all()
    assert call["dynamic_map"][:, :, 6].all()
    assert torch.equal(out, query)


def test_draft_batch_masks_match_classifier_free_guidance_semantics(monkeypatch):
    calls = []

    def fake_block_sparse_attention(q_sorted, k_sorted, v_sorted, q_sizes, k_sizes, dynamic_map, scale):
        calls.append(
            {
                "q_shape": tuple(q_sorted.shape),
                "q_sizes": q_sizes.clone(),
                "dynamic_map": dynamic_map.clone(),
            }
        )
        return q_sorted

    monkeypatch.setattr(
        "sparsevideo.kernels.block_sparse_attn.block_sparse_attention",
        fake_block_sparse_attention,
    )

    torch.manual_seed(0)
    query = torch.randn(2, 24, 1, 16)
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=2,
        pool_w=2,
        latent_h=4,
        latent_w=6,
        visual_len=24,
        text_len=0,
        device=torch.device("cpu"),
    )

    out = _draft_triton_path(
        query,
        query,
        query,
        B=2,
        N=24,
        H=1,
        D=16,
        scale=16 ** -0.5,
        context_len=0,
        video_end=24,
        T=1,
        frame_h=4,
        frame_w=6,
        frame_size=24,
        sparsity_ratio=0.75,
        pool_h=2,
        pool_w=2,
        reorg_idx=reorg_idx,
        restore_idx=restore_idx,
    )

    assert calls
    call = calls[0]
    assert call["q_shape"] == (2, 24, 16)
    assert call["q_sizes"].tolist() == [[4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4]]

    sampled = _sample_qk_attention_2d(
        query,
        query,
        frame_h=4,
        frame_w=6,
        pool_h=2,
        pool_w=2,
    )
    expected = _attention_percentile_mask_headwise(sampled, keep_ratio=0.25).reshape(2, 6, 6)

    assert torch.equal(call["dynamic_map"], expected)
    assert not torch.equal(call["dynamic_map"][0], call["dynamic_map"][1])
    assert torch.equal(out, query)
