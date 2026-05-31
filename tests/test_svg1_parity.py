from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.svg1.method import (
    _build_svg_block_mask,
    _profile_masks,
    _svg1_dense_attention,
    _place_svg_heads,
    _restore_svg_heads,
    _round_svg_window_width,
    _sparsity_to_width,
    _svg_attention,
    _svg_common_mask,
    _svg_kv_blocks,
    _svg_profile_mask_rows,
    _svg_window_width,
)
from sparsevideo.methods._schedule import configured_dense_warmup_requires_dense


def test_svg1_wan_full_attention_uses_upstream_sdpa_layout(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method

    calls = {}

    def fake_sdpa(q, k, v, **kwargs):
        calls["shape"] = q.shape
        calls["kwargs"] = kwargs
        return q

    monkeypatch.setattr(svg1_method.F, "scaled_dot_product_attention", fake_sdpa)

    query = torch.randn(1, 4, 2, 3)
    out = _svg1_dense_attention(query, query, query, None, model_type="wan")

    assert calls["shape"] == (1, 2, 4, 3)
    assert calls["kwargs"] == {"dropout_p": 0.0, "is_causal": False}
    assert out.shape == query.shape


def test_svg1_hunyuan_full_attention_uses_upstream_flash_attn_varlen(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method

    calls = {}

    def fake_flash_attn(q, k, v, cu_q, cu_k, max_q, max_k):
        calls["q_shape"] = q.shape
        calls["cu_q"] = cu_q.tolist()
        calls["cu_k"] = cu_k.tolist()
        calls["max_q"] = max_q
        calls["max_k"] = max_k
        return q

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg1_method, "_load_flash_attn_varlen_func", lambda: fake_flash_attn)

    query = torch.randn(1, 6, 2, 3)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    out = _svg1_dense_attention(query, query, query, attention_mask, model_type="hunyuan_video")

    assert calls == {
        "q_shape": (6, 2, 3),
        "cu_q": [0, 4, 6],
        "cu_k": [0, 4, 6],
        "max_q": 6,
        "max_k": 6,
    }
    assert out.shape == query.shape


def _upstream_svg1_profile_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, model_type):
    video_len = num_frame * frame_size
    seq_len = context_length + video_len
    attention_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    if model_type == "hunyuan_video":
        video = torch.zeros((video_len, video_len), dtype=torch.bool)
        block_thres = frame_size * 1.5
    else:
        video = torch.zeros_like(attention_mask, dtype=torch.bool)
        video[:, :frame_size] = True
        block_thres = frame_size * 2

    block_size = 128
    num_block = (video_len + block_size - 1) // block_size
    for i in range(num_block):
        for j in range(num_block):
            if abs(i - j) < block_thres // block_size:
                video[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = True

    if mask_name == "temporal":
        video = video.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(video_len, video_len)
    elif mask_name != "spatial":
        raise ValueError(mask_name)

    if model_type == "hunyuan_video":
        attention_mask[:-context_length, :-context_length] = video
        attention_mask[-context_length:, :] = True
        attention_mask[:, -context_length:] = True
    else:
        attention_mask = video

    return attention_mask[:sample_mse_max_row]


def _upstream_svg1_common_mask_matrix(model_type, video_len, frame_size, num_frames, window_width, prompt_length=0):
    q_idx = torch.arange(video_len + (256 if model_type == "hunyuan_video" else 0)).unsqueeze(1)
    kv_idx = torch.arange(q_idx.shape[0]).unsqueeze(0)
    if model_type == "hunyuan_video":
        real_length = video_len + prompt_length
        real_mask = (kv_idx < real_length) & (q_idx < real_length)
        fake_mask = (kv_idx >= real_length) & (q_idx >= real_length)
        temporal_head_mask = torch.abs(q_idx - kv_idx) < window_width
        text_column_mask = (video_len <= kv_idx) & (kv_idx < real_length)
        text_row_mask = (video_len <= q_idx) & (q_idx < real_length)
        return (real_mask & (temporal_head_mask | text_column_mask | text_row_mask)) | fake_mask
    temporal_head_mask = torch.abs(q_idx - kv_idx) <= window_width
    first_frame_mask = kv_idx < frame_size
    return first_frame_mask | temporal_head_mask


def test_svg1_flex_attention_uses_upstream_compile_settings(monkeypatch):
    import sparsevideo.methods.svg1.method as svg1_method

    calls = []

    def fake_compile(fn, *, dynamic, mode=None):
        calls.append({"dynamic": dynamic, "mode": mode})

        def compiled(query, key, value, *, block_mask):
            calls.append({"block_mask": block_mask})
            return query

        return compiled

    monkeypatch.setattr(svg1_method, "_SVG_FLEX_ATTENTION", {})
    monkeypatch.setattr(torch, "compile", fake_compile)

    query = torch.zeros((1, 1, 2, 4))
    block_mask = object()
    out = svg1_method._svg_flex_attention(query, query, query, block_mask, model_type="wan")

    assert out is query
    assert calls == [
        {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        {"block_mask": block_mask},
    ]


def test_svg1_hunyuan_flex_attention_uses_upstream_compile_settings(monkeypatch):
    import sparsevideo.methods.svg1.method as svg1_method

    calls = []

    def fake_compile(fn, *, dynamic, mode=None):
        calls.append({"dynamic": dynamic, "mode": mode})

        def compiled(query, key, value, *, block_mask):
            calls.append({"block_mask": block_mask})
            return query

        return compiled

    monkeypatch.setattr(svg1_method, "_SVG_FLEX_ATTENTION", {})
    monkeypatch.setattr(torch, "compile", fake_compile)

    query = torch.zeros((1, 1, 2, 4))
    block_mask = object()
    out = svg1_method._svg_flex_attention(query, query, query, block_mask, model_type="hunyuan_video")

    assert out is query
    assert calls == [
        {"dynamic": False, "mode": None},
        {"block_mask": block_mask},
    ]


def test_svg1_flex_attention_pads_non_power_of_two_head_dim(monkeypatch):
    import sparsevideo.methods.svg1.method as svg1_method

    calls = []

    def fake_compile(fn, *, dynamic, mode=None):
        calls.append({"dynamic": dynamic, "mode": mode})

        def compiled(query, key, value, *, block_mask, scale=None):
            calls.append({"shape": tuple(query.shape), "block_mask": block_mask, "scale": scale})
            return query

        return compiled

    monkeypatch.setattr(svg1_method, "_SVG_FLEX_ATTENTION", {})
    monkeypatch.setattr(torch, "compile", fake_compile)

    query = torch.zeros((1, 1, 2, 96))
    block_mask = object()
    out = svg1_method._svg_flex_attention(query, query, query, block_mask, model_type="allegro")

    assert out.shape == query.shape
    assert calls == [
        {"dynamic": False, "mode": "max-autotune-no-cudagraphs"},
        {"shape": (1, 1, 2, 128), "block_mask": block_mask, "scale": 96 ** -0.5},
    ]


def test_svg1_temporal_profile_sink_is_token_major_like_upstream_wan():
    frame_size = 256
    num_frames = 3
    video_len = frame_size * num_frames
    q_idx = torch.tensor([video_len - 1])
    all_idx = torch.arange(video_len)

    mask = _svg_profile_mask_rows(
        "temporal",
        q_idx,
        all_idx,
        context_len=0,
        video_end=video_len,
        frame_size=frame_size,
        num_frames=num_frames,
        model_type="wan",
    )

    assert mask[0, frame_size]


def test_svg1_profile_masks_match_upstream_wan_reference():
    frame_size = 256
    num_frames = 3
    video_len = frame_size * num_frames
    all_idx = torch.arange(video_len)
    q_idx = torch.arange(video_len)

    for mask_name in ("spatial", "temporal"):
        actual = _svg_profile_mask_rows(
            mask_name,
            q_idx,
            all_idx,
            context_len=0,
            video_end=video_len,
            frame_size=frame_size,
            num_frames=num_frames,
            model_type="wan",
        )
        expected = _upstream_svg1_profile_mask(
            mask_name,
            sample_mse_max_row=video_len,
            context_length=0,
            num_frame=num_frames,
            frame_size=frame_size,
            model_type="wan",
        )
        assert torch.equal(actual.cpu(), expected)


def test_svg1_profile_masks_match_upstream_hunyuan_reference():
    frame_size = 256
    num_frames = 3
    context_length = 4
    video_len = frame_size * num_frames
    seq_len = video_len + context_length
    all_idx = torch.arange(seq_len)
    q_idx = torch.arange(seq_len)

    for mask_name in ("spatial", "temporal"):
        actual = _svg_profile_mask_rows(
            mask_name,
            q_idx,
            all_idx,
            context_len=0,
            video_end=video_len,
            frame_size=frame_size,
            num_frames=num_frames,
            model_type="hunyuan_video",
        )
        expected = _upstream_svg1_profile_mask(
            mask_name,
            sample_mse_max_row=seq_len,
            context_length=context_length,
            num_frame=num_frames,
            frame_size=frame_size,
            model_type="hunyuan_video",
        )
        assert torch.equal(actual.cpu(), expected)


def test_svg1_common_mask_matches_upstream_wan_reference():
    frame_size = 256
    num_frames = 3
    video_len = frame_size * num_frames
    window_width = 512
    q_idx = torch.arange(video_len).unsqueeze(1)
    kv_idx = torch.arange(video_len).unsqueeze(0)

    actual = _svg_common_mask(q_idx, kv_idx, video_len, frame_size, window_width, model_type="wan")
    expected = _upstream_svg1_common_mask_matrix("wan", video_len, frame_size, num_frames, window_width)

    assert torch.equal(actual.cpu(), expected)


def test_svg1_common_mask_supports_longcat_kv_offset_condition_prefix():
    q_idx = torch.tensor([[0], [4]])
    kv_idx = torch.arange(10).unsqueeze(0)

    mask = _svg_common_mask(
        q_idx,
        kv_idx,
        video_len=6,
        frame_size=2,
        window_width=1,
        model_type="wan",
        q_kv_offset=4,
    )

    assert mask.tolist() == [
        [True, True, True, True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False, True, True, True],
    ]


def test_svg1_common_mask_matches_upstream_hunyuan_reference():
    frame_size = 256
    num_frames = 3
    video_len = frame_size * num_frames
    prompt_length = 7
    window_width = 256
    seq_len = video_len + 256
    q_idx = torch.arange(seq_len).unsqueeze(1)
    kv_idx = torch.arange(seq_len).unsqueeze(0)

    actual = _svg_common_mask(
        q_idx,
        kv_idx,
        video_len,
        frame_size,
        window_width,
        model_type="hunyuan_video",
        prompt_length=prompt_length,
    )
    expected = _upstream_svg1_common_mask_matrix(
        "hunyuan_video",
        video_len,
        frame_size,
        num_frames,
        window_width,
        prompt_length=prompt_length,
    )

    assert torch.equal(actual.cpu(), expected)


def _candidate_block_matrix_from_indices(kv_num_blocks, kv_indices, num_kv_blocks):
    out = torch.zeros(kv_num_blocks.shape[-1], num_kv_blocks, dtype=torch.bool)
    for q_block in range(kv_num_blocks.shape[-1]):
        count = int(kv_num_blocks[0, 0, q_block])
        out[q_block, kv_indices[0, 0, q_block, :count].cpu().long()] = True
    return out


def _block_mask_matrix(block_mask):
    return block_mask.to_dense()[0, 0].cpu()


def _expected_svg_candidate_blocks(N, video_len, frame_size, window_width, model_type="wan", prompt_length=0):
    block_size = 128
    num_q_blocks = (N + block_size - 1) // block_size
    num_kv_blocks = num_q_blocks
    expected = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool)
    for q_block in range(num_q_blocks):
        q_idx = torch.arange(q_block * block_size, min(N, (q_block + 1) * block_size)).unsqueeze(1)
        for kv_block in range(num_kv_blocks):
            kv_idx = torch.arange(kv_block * block_size, min(N, (kv_block + 1) * block_size)).unsqueeze(0)
            expected[q_block, kv_block] = bool(
                _svg_common_mask(
                    q_idx,
                    kv_idx,
                    video_len,
                    frame_size,
                    window_width,
                    model_type=model_type,
                    prompt_length=prompt_length,
                ).any()
            )
    return expected


def test_svg1_block_mask_candidates_match_common_mask_wan_without_dense_n2_mask():
    N = 3 * 256
    kv_num_blocks, kv_indices = _svg_kv_blocks(
        N,
        video_len=N,
        frame_size=256,
        window_width=256,
        model_type="wan",
        device=torch.device("cpu"),
    )

    actual = _candidate_block_matrix_from_indices(kv_num_blocks, kv_indices, (N + 127) // 128)
    expected = _expected_svg_candidate_blocks(N, N, 256, 256, model_type="wan")

    assert torch.equal(actual, expected)


def test_svg1_block_mask_candidates_match_common_mask_hunyuan_without_dense_n2_mask():
    video_len = 3 * 256
    prompt_length = 7
    N = video_len + 256
    kv_num_blocks, kv_indices = _svg_kv_blocks(
        N,
        video_len=video_len,
        frame_size=256,
        window_width=256,
        model_type="hunyuan_video",
        prompt_length=prompt_length,
        device=torch.device("cpu"),
    )

    actual = _candidate_block_matrix_from_indices(kv_num_blocks, kv_indices, (N + 127) // 128)
    expected = _expected_svg_candidate_blocks(
        N, video_len, 256, 256, model_type="hunyuan_video", prompt_length=prompt_length,
    )

    assert torch.equal(actual, expected)


def test_svg1_block_mask_partitions_match_common_mask_wan_without_dense_n2_mask():
    N = 3 * 256
    block_mask = _build_svg_block_mask(
        N,
        video_len=N,
        frame_size=256,
        num_frames=3,
        window_width=256,
        device=torch.device("cpu"),
        model_type="wan",
    )

    expected = _expected_svg_candidate_blocks(N, N, 256, 256, model_type="wan")
    partial = _candidate_block_matrix_from_indices(
        block_mask.kv_num_blocks, block_mask.kv_indices, (N + 127) // 128,
    )
    full = _candidate_block_matrix_from_indices(
        block_mask.full_kv_num_blocks, block_mask.full_kv_indices, (N + 127) // 128,
    )

    assert torch.equal(partial | full, expected)
    assert not (partial & full).any()
    assert torch.equal(_block_mask_matrix(block_mask), expected)


def test_svg1_rectangular_block_mask_matches_offset_common_mask():
    q_len = 256
    kv_len = 384
    video_len = q_len
    frame_size = 128
    num_frames = 2
    q_kv_offset = 128
    window_width = 128

    block_mask = _build_svg_block_mask(
        q_len,
        video_len,
        frame_size,
        num_frames,
        window_width,
        torch.device("cpu"),
        model_type="wan",
        kv_len=kv_len,
        q_kv_offset=q_kv_offset,
    )
    expected = torch.zeros(2, 3, dtype=torch.bool)
    for q_block in range(2):
        q_idx = torch.arange(q_block * 128, min(q_len, (q_block + 1) * 128)).unsqueeze(1)
        for kv_block in range(3):
            kv_idx = torch.arange(kv_block * 128, min(kv_len, (kv_block + 1) * 128)).unsqueeze(0)
            expected[q_block, kv_block] = bool(
                _svg_common_mask(
                    q_idx,
                    kv_idx,
                    video_len,
                    frame_size,
                    window_width,
                    model_type="wan",
                    q_kv_offset=q_kv_offset,
                ).any()
            )

    assert block_mask.shape == (1, 1, q_len, kv_len)
    assert torch.equal(_block_mask_matrix(block_mask).bool(), expected)


def test_svg1_block_mask_partitions_match_common_mask_hunyuan_without_dense_n2_mask():
    video_len = 3 * 256
    prompt_length = 7
    N = video_len + 256
    block_mask = _build_svg_block_mask(
        N,
        video_len=video_len,
        frame_size=256,
        num_frames=3,
        window_width=256,
        device=torch.device("cpu"),
        model_type="hunyuan_video",
        prompt_length=prompt_length,
    )

    expected = _expected_svg_candidate_blocks(
        N, video_len, 256, 256, model_type="hunyuan_video", prompt_length=prompt_length,
    )
    partial = _candidate_block_matrix_from_indices(
        block_mask.kv_num_blocks, block_mask.kv_indices, (N + 127) // 128,
    )
    full = _candidate_block_matrix_from_indices(
        block_mask.full_kv_num_blocks, block_mask.full_kv_indices, (N + 127) // 128,
    )

    assert torch.equal(partial | full, expected)
    assert not (partial & full).any()
    assert torch.equal(_block_mask_matrix(block_mask), expected)


def test_svg1_wan_720p_block_mask_metadata_stays_block_sparse():
    video_len = 21 * 45 * 80
    frame_size = 45 * 80
    window_width = _round_svg_window_width(
        _sparsity_to_width(0.3, 0, 21, frame_size) * frame_size,
        model_type="wan",
    )

    kv_num_blocks, kv_indices = _svg_kv_blocks(
        video_len,
        video_len=video_len,
        frame_size=frame_size,
        window_width=window_width,
        model_type="wan",
        device=torch.device("cpu"),
    )

    num_blocks = (video_len + 127) // 128
    assert kv_num_blocks.shape == (1, 1, num_blocks)
    assert kv_indices.shape[-1] == num_blocks
    assert int(kv_num_blocks.max()) < num_blocks


def test_svg1_wan_720p_block_mask_build_does_not_evaluate_dense_mask(monkeypatch):
    import sparsevideo.methods.svg1.method as svg1_method

    def fail_if_called(*args, **kwargs):
        raise AssertionError("svg1 block mask construction must not evaluate the elementwise mask")

    video_len = 21 * 45 * 80
    frame_size = 45 * 80
    window_width = _round_svg_window_width(
        _sparsity_to_width(0.3, 0, 21, frame_size) * frame_size,
        model_type="wan",
    )

    monkeypatch.setattr(svg1_method, "_svg_common_mask", fail_if_called)
    block_mask = svg1_method._build_svg_block_mask(
        video_len,
        video_len=video_len,
        frame_size=frame_size,
        num_frames=21,
        window_width=window_width,
        device=torch.device("cpu"),
        model_type="wan",
    )

    num_blocks = (video_len + 127) // 128
    assert block_mask.shape == (1, 1, video_len, video_len)
    assert block_mask.kv_indices.shape[-1] == num_blocks
    assert block_mask.full_kv_indices.shape[-1] == num_blocks
    assert int((block_mask.kv_num_blocks + block_mask.full_kv_num_blocks).max()) < num_blocks


def test_svg1_width_rounding_matches_upstream_wan_and_hunyuan():
    frame_size = 3600
    num_frames = 21
    sparsity = 0.3
    raw_width = _sparsity_to_width(sparsity, 0, num_frames, frame_size) * frame_size

    assert _round_svg_window_width(raw_width, model_type="wan") == 128 * ((raw_width + 127) // 128)
    assert _round_svg_window_width(raw_width, model_type="hunyuan_video") == 128 * int(raw_width // 128)


def test_svg1_hunyuan_window_width_uses_upstream_context_tail_length():
    frame_size = 45 * 80
    num_frames = 33
    context_length = 256
    sparsity = 0.25

    raw_width = _sparsity_to_width(sparsity, context_length, num_frames, frame_size) * frame_size

    assert _svg_window_width(sparsity, "hunyuan_video", context_length, num_frames, frame_size) == (
        128 * int(raw_width // 128)
    )
    assert _svg_window_width(sparsity, "hunyuan_video", context_length, num_frames, frame_size) != (
        _round_svg_window_width(
            _sparsity_to_width(sparsity, 0, num_frames, frame_size) * frame_size,
            model_type="hunyuan_video",
        )
    )


def test_svg1_warmup_step_gate_uses_dense_warmup_ratio():
    config = {"dense_warmup_step_ratio": 0.2}

    dense_steps = [
        configured_dense_warmup_requires_dense(config, 50, step=step)
        for step in range(1, 13)
    ]

    assert dense_steps == [True] * 10 + [False, False]


def test_svg1_method_uses_dense_warmup_step_ratio(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method
    from sparsevideo.methods.svg1.method import SVG1Method

    calls = []

    def fake_dense(query, key, value, attention_mask, *, model_type):
        calls.append("dense")
        return query

    def fake_sparse(*args, **kwargs):
        calls.append("sparse")
        return args[0]

    monkeypatch.setattr(svg1_method, "_svg1_dense_attention", fake_dense)
    monkeypatch.setattr(svg1_method, "_svg_attention", fake_sparse)

    method = SVG1Method(
        {
            "dense_warmup_step_ratio": 0.5,
            "dense_warmup_layer_ratio": 0.0,
            "num_inference_steps": 50,
        },
        SimpleNamespace(model_type="wan", model_key=None),
    )
    step_tracker = SimpleNamespace(step=20, timestep=0)
    processor = method.create_processor(0, 2, None, step_tracker)
    query = torch.zeros(1, 8, 1, 4)

    processor.attn_fn(query, query, query, None, timestep=torch.tensor([0]))

    assert calls == ["dense"]


def test_svg1_method_uses_tracker_step_for_dense_warmup(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method
    from sparsevideo.methods.svg1.method import SVG1Method

    calls = []

    def fake_dense(query, key, value, attention_mask, *, model_type):
        calls.append("dense")
        return query

    def fake_sparse(*args, **kwargs):
        calls.append("sparse")
        return args[0]

    monkeypatch.setattr(svg1_method, "_svg1_dense_attention", fake_dense)
    monkeypatch.setattr(svg1_method, "_svg_attention", fake_sparse)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    method = SVG1Method(
        {
            "dense_warmup_step_ratio": 0.5,
            "dense_warmup_layer_ratio": 0.0,
            "num_inference_steps": 50,
        },
        SimpleNamespace(model_type="wan", model_key=None),
    )
    step_tracker = SimpleNamespace(step=26, timestep=926.0)
    processor = method.create_processor(0, 2, None, step_tracker)
    query = torch.zeros(1, 8, 1, 4)

    processor.attn_fn(query, query, query, None, timestep=torch.tensor([0]))

    assert calls == ["sparse"]


def test_svg1_hunyuan_uses_config_prompt_length_fallback_like_upstream(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method
    from sparsevideo.methods.svg1.method import SVG1Method

    captured = {}

    def fake_sparse(query, key, value, **kwargs):
        captured["prompt_length"] = kwargs["prompt_length"]
        captured["context_length"] = kwargs["context_length"]
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg1_method, "_svg_attention", fake_sparse)

    method = SVG1Method(
        {
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
            "context_length": 4,
            "prompt_length": 3,
        },
        SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    step_tracker = SimpleNamespace(step=1, timestep=0)
    processor = method.create_processor(0, 2, None, step_tracker)
    query = torch.zeros(1, 12, 1, 4)

    processor.attn_fn(query, query, query, None, text_len=4)

    assert captured == {"prompt_length": 3, "context_length": 4}


def test_svg1_hunyuan_runtime_prompt_length_overrides_config_fallback(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method
    from sparsevideo.methods.svg1.method import SVG1Method

    captured = {}

    def fake_sparse(query, key, value, **kwargs):
        captured["prompt_length"] = kwargs["prompt_length"]
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg1_method, "_svg_attention", fake_sparse)

    method = SVG1Method(
        {
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
            "context_length": 4,
            "prompt_length": 4,
        },
        SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    step_tracker = SimpleNamespace(step=1, timestep=0)
    processor = method.create_processor(0, 2, None, step_tracker)
    query = torch.zeros(1, 12, 1, 4)

    processor.attn_fn(query, query, query, None, text_len=4, prompt_length=2)

    assert captured == {"prompt_length": 2}


def test_svg1_hunyuan_rejects_context_length_mismatch_like_upstream_assertion():
    query = torch.zeros(1, 12, 1, 4)

    with pytest.raises(RuntimeError, match="context_length must match"):
        _svg_attention(
            query,
            query,
            query,
            sparsity=0.25,
            num_sampled_rows=4,
            sample_mse_max_row=8,
            state={},
            step_tracker_step=1,
            model_type="hunyuan_video",
            text_len=4,
            prompt_length=3,
            context_length=5,
        )


def test_svg1_wan_sparse_attention_accepts_rectangular_longcat_qkv(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method

    calls = {}

    def fake_profile(query, key, value, scale, context_len, video_end,
                     frame_size, num_frames, num_sampled_rows,
                     sample_mse_max_row, **kwargs):
        calls["profile"] = {
            "query_len": query.shape[1],
            "key_len": key.shape[1],
            "video_end": video_end,
            "frame_size": frame_size,
            "num_frames": num_frames,
            "kv_video_end": kwargs["kv_video_end"],
            "kv_num_frames": kwargs["kv_num_frames"],
            "q_kv_offset": kwargs["q_kv_offset"],
        }
        return torch.zeros(query.shape[0], query.shape[2], dtype=torch.long, device=query.device)

    def fake_flex(query, key, value, block_mask, model_type="wan"):
        calls["flex"] = {
            "query_shape": tuple(query.shape),
            "key_shape": tuple(key.shape),
            "value_shape": tuple(value.shape),
            "block_mask_shape": block_mask.shape,
        }
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg1_method, "_svg_placement_triton_supported", lambda tensor: False)
    monkeypatch.setattr(svg1_method, "_profile_masks", fake_profile)
    monkeypatch.setattr(svg1_method, "_svg_flex_attention", fake_flex)

    query = torch.randn(1, 6, 2, 4)
    key = torch.randn(1, 12, 2, 4)

    out = _svg_attention(
        query,
        key,
        key,
        sparsity=0.25,
        num_sampled_rows=2,
        sample_mse_max_row=6,
        state={},
        step_tracker_step=1,
        model_type="wan",
    )

    assert out.shape == query.shape
    assert calls["profile"] == {
        "query_len": 6,
        "key_len": 12,
        "video_end": 6,
        "frame_size": 3,
        "num_frames": 2,
        "kv_video_end": 12,
        "kv_num_frames": 4,
        "q_kv_offset": 6,
    }
    assert calls["flex"] == {
        "query_shape": (1, 2, 6, 4),
        "key_shape": (1, 2, 12, 4),
        "value_shape": (1, 2, 12, 4),
        "block_mask_shape": (1, 1, 6, 12),
    }


def test_svg1_warmup_ratio_gate_still_covers_first_ratio_steps_without_scheduler_threshold():
    dense = [
        configured_dense_warmup_requires_dense({"dense_warmup_step_ratio": 0.2}, 50, step=step)
        for step in range(1, 13)
    ]

    assert dense[:10] == [True] * 10
    assert dense[10:] == [False, False]


def test_svg1_profile_sampling_uses_upstream_cpu_rng(monkeypatch):
    calls = []
    original_randint = torch.randint

    def wrapped_randint(*args, **kwargs):
        calls.append(kwargs.get("device"))
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "randint", wrapped_randint)
    query = torch.randn(1, 768, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    _profile_masks(
        query,
        key,
        value,
        scale=4 ** -0.5,
        context_len=0,
        video_end=768,
        frame_size=256,
        num_frames=3,
        num_sampled_rows=4,
        sample_mse_max_row=16,
        model_type="wan",
    )

    assert calls[0] == "cpu"


def test_svg1_head_placement_round_trips_temporal_heads():
    video_len = 12
    num_frames = 3
    frame_size = 4
    tensor = torch.arange(1 * 2 * video_len * 1).reshape(1, 2, video_len, 1)
    head_choices = torch.tensor([[0, 1]])

    placed, _, _ = _place_svg_heads(tensor, tensor, tensor, head_choices, video_len, num_frames, frame_size)
    restored = _restore_svg_heads(placed, head_choices, video_len, num_frames, frame_size)

    assert torch.equal(placed[:, 0], tensor[:, 0])
    assert placed[0, 1, :, 0].tolist() == [12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]
    assert torch.equal(restored, tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SVG1 Triton placement")
def test_svg1_triton_head_placement_matches_reference():
    video_len = 12
    num_frames = 3
    frame_size = 4
    tensor = torch.arange(1 * 2 * video_len * 4, device="cuda", dtype=torch.float32).reshape(1, 2, video_len, 4)
    head_choices = torch.tensor([[0, 1]], device="cuda")
    token_major = (torch.arange(video_len, device="cuda") % frame_size) * num_frames + (
        torch.arange(video_len, device="cuda") // frame_size
    )

    placed, _, _ = _place_svg_heads(tensor, tensor, tensor, head_choices, video_len, num_frames, frame_size)
    restored = _restore_svg_heads(placed, head_choices, video_len, num_frames, frame_size)

    expected = tensor.clone()
    expected[:, 1, token_major, :] = tensor[:, 1, :video_len, :]
    assert torch.equal(placed, expected)
    assert torch.equal(restored, tensor)


def test_svg1_hunyuan_common_mask_keeps_fake_text_isolated_like_upstream():
    video_len = 12
    frame_size = 4
    prompt_length = 2
    window_width = 8

    assert _svg_common_mask(
        torch.tensor(video_len + prompt_length),
        torch.tensor(0),
        video_len,
        frame_size,
        window_width,
        model_type="hunyuan_video",
        prompt_length=prompt_length,
    ).item() is False
    assert _svg_common_mask(
        torch.tensor(video_len + prompt_length),
        torch.tensor(video_len + prompt_length + 1),
        video_len,
        frame_size,
        window_width,
        model_type="hunyuan_video",
        prompt_length=prompt_length,
    ).item() is True
