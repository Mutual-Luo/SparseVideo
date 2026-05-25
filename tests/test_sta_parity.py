from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparsevideo.methods.sta import STAMethod
from sparsevideo.methods.sta.config import _HUNYUAN_MASK_STRATEGY, _WAN_MASK_STRATEGY
from sparsevideo.methods.sta.method import (
    _is_supported_fastvideo_shape,
    _load_mask_strategy,
    _sta_attention,
    _sta_backend_name,
    _sta_pad_video_canvas,
    _sta_padded_border_indices,
    _sta_sparsevideo_fastvideo_path,
    _sta_tile_bhsd,
    _sta_untile_bhsd,
    _sta_window_sizes,
)
from sparsevideo.methods.sta.search import (
    model_search_defaults,
    model_strategy_shape,
    summarize_strategy,
    tune_search_results,
)
from sparsevideo.methods.sta.ops import (
    STA_SUPPORTED_SEQ_SHAPES,
    STA_TILE_SIZE,
    _can_use_a100_sta,
    _sta_triton_head_dim,
    _can_use_h100_sta,
    _sta_a100_block_mask_cpu,
    _sta_a100_image_valid_mask_cpu,
    _owned_fastvideo_sta_triton,
    _patch_a100_triton_autotune,
    _sliding_tile_attention_triton,
    sliding_tile_attention,
)
from sparsevideo.methods.sta import ops as sta_ops
from sparsevideo.kernels.native import sta_h100


def _canonical_strategy_sha256(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def test_sta_supported_shapes_match_fastvideo_kernel_api():
    assert set(STA_SUPPORTED_SEQ_SHAPES) == {"18x48x80", "30x48x80", "36x48x48"}
    assert _is_supported_fastvideo_shape((18, 48, 80))
    assert _is_supported_fastvideo_shape((30, 48, 80))
    assert _is_supported_fastvideo_shape((36, 48, 48))
    assert not _is_supported_fastvideo_shape((21, 45, 80))


def test_sta_h100_owned_source_import_is_optional():
    assert hasattr(sta_h100, "sta_fwd")


def test_sta_triton_fallback_loads_owned_fastvideo_source():
    fn = _owned_fastvideo_sta_triton()
    path = fn.__code__.co_filename

    assert "training_free" not in path
    assert "src/sparsevideo/kernels/native/sta_h100/python/fastvideo_kernel/triton_kernels/st_attn_triton.py" in path


def _source_files(root):
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_sta_owned_fastvideo_runtime_sources_match_upstream_references():
    repo = Path(__file__).resolve().parents[1]
    owned_root = repo / "src/sparsevideo/kernels/native/sta_h100"
    upstream_root = repo / "training_free/FastVideo/fastvideo-kernel"
    generalized_triton = Path("python/fastvideo_kernel/triton_kernels/st_attn_triton.py")
    generalized_triton_under_python = Path("fastvideo_kernel/triton_kernels/st_attn_triton.py")

    for rel in ["LICENSE", "MANIFEST.in", "pyproject.toml"]:
        assert (owned_root / rel).read_bytes() == (upstream_root / rel).read_bytes()

    for rel in [
        "csrc",
        "python",
        "include/cutlass/include",
        "include/tk/include",
        "include/tk/prototype",
    ]:
        owned = owned_root / rel
        upstream = upstream_root / rel
        owned_files = _source_files(owned)
        assert owned_files == _source_files(upstream)
        for source in owned_files:
            if rel == "python" and source == generalized_triton_under_python:
                continue
            assert (owned / source).read_bytes() == (upstream / source).read_bytes()

    owned_triton = (owned_root / generalized_triton).read_text(encoding="utf-8")
    upstream_triton = (upstream_root / generalized_triton).read_text(encoding="utf-8")
    assert owned_triton != upstream_triton
    for token in [
        "def sliding_tile_attention_triton",
        "if dit_seq_shape == '30x48x80'",
        "elif dit_seq_shape == '36x48x48'",
        "elif dit_seq_shape == '18x48x80'",
        "assert q.shape[1] == len(window_size)",
    ]:
        assert token in owned_triton
    for token in [
        "dit_seq_shape='30x48x80', sm_scale=None",
        'parts = str(dit_seq_shape).lower().split("x")',
        "img_seq_len = canvas_t * canvas_h * canvas_w",
    ]:
        assert token in owned_triton


def test_sta_dispatches_to_a100_block_sparse_when_h100_is_unavailable(monkeypatch):
    calls = {}

    def fake_validate(q, k, v, window_size, has_text, seq_shape):
        return "18x48x80"

    def fake_a100(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape=None):
        calls["path"] = "a100_block_sparse"
        calls["seq_shape"] = seq_shape
        calls["source_seq_shape"] = source_seq_shape
        return q

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(sta_ops, "_validate_fastvideo_sta_inputs", fake_validate)
    monkeypatch.setattr(sta_ops, "_can_use_h100_sta", lambda q: False)
    monkeypatch.setattr(sta_ops, "_can_use_a100_sta", lambda q: True)
    monkeypatch.setattr(sta_ops, "_sliding_tile_attention_a100", fake_a100)

    q = torch.randn(1, 1, 8, 4)

    out = sliding_tile_attention(
        q,
        q,
        q,
        window_size=[(3, 3, 5)],
        text_length=0,
        has_text=False,
        seq_shape="18x48x80",
    )

    assert out is q
    assert calls == {"path": "a100_block_sparse", "seq_shape": "18x48x80", "source_seq_shape": None}


def test_sta_backend_name_marks_a100_block_sparse_cuda_path(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(sta_ops, "sta_fwd", object())
    monkeypatch.setattr(sta_ops, "_can_use_a100_sta", lambda q: True)
    monkeypatch.delenv("SPARSEVIDEO_STA_TRITON_AUTOTUNE", raising=False)

    assert _sta_backend_name(torch.empty(1)) == "fastvideo_sta_a100_block_sparse_cuda"


def test_sta_backend_name_does_not_use_triton_autotune_env(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(sta_ops, "sta_fwd", object())
    monkeypatch.setattr(sta_ops, "_can_use_a100_sta", lambda q: True)
    monkeypatch.setenv("SPARSEVIDEO_STA_TRITON_AUTOTUNE", "full")

    assert _sta_backend_name(torch.empty(1)) == "fastvideo_sta_a100_block_sparse_cuda"


def test_sta_wan_generalized_shape_reaches_owned_path(monkeypatch):
    calls = {}

    def fake_path(
        query, key, value, batch, tokens, heads, dim,
        vid_start, video_len, text_len, context_len,
        frames, spatial_h, spatial_w,
        frames_pad, height_pad, width_pad,
        tile_size, kernel_size, model_type, seq_shape, has_text,
        layer_idx, step_idx, mask_strategy, prompt_length,
    ):
        calls["shape"] = (frames, spatial_h, spatial_w)
        calls["padded_shape"] = (frames_pad, height_pad, width_pad)
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr("sparsevideo.methods.sta.method._sta_sparsevideo_fastvideo_path", fake_path)

    query = torch.zeros(1, 21 * 45 * 80, 2, 4)
    out = _sta_attention(
        query,
        query,
        query,
        tile_size=(6, 8, 8),
        kernel_size=(3, 6, 10),
        model_type="wan",
        has_text=False,
    )

    assert out is query
    assert calls == {"shape": (21, 45, 80), "padded_shape": (24, 48, 80)}


def test_sta_generalized_canvas_padding_repeats_edges():
    x = torch.arange(1 * 1 * 2 * 2 * 2 * 1).reshape(1, 1, 8, 1)

    padded = _sta_pad_video_canvas(x, (2, 2, 2), (3, 3, 3)).view(1, 1, 3, 3, 3, 1)
    source = x.view(1, 1, 2, 2, 2, 1)

    assert torch.equal(padded[:, :, :2, :2, :2], source)
    assert torch.equal(padded[:, :, 2, :2, :2], source[:, :, 1])
    assert torch.equal(padded[:, :, :, 2, :2], padded[:, :, :, 1, :2])
    assert torch.equal(padded[:, :, :, :, 2], padded[:, :, :, :, 1])


def test_sta_padded_border_indices_cover_only_incomplete_tiles():
    indices = _sta_padded_border_indices((3, 5, 4), (4, 6, 4), (2, 3, 4))
    coords = {tuple(item.tolist()) for item in torch.stack(torch.unravel_index(indices, (3, 5, 4)), dim=1)}

    assert (2, 0, 0) in coords
    assert (0, 3, 0) in coords
    assert (0, 0, 0) not in coords
    assert all(t == 2 or h >= 3 for t, h, _ in coords)


def test_sta_a100_autotune_patch_is_wrapper_owned(monkeypatch):
    class DummyActive:
        @staticmethod
        def get_current_target():
            return SimpleNamespace(backend="cuda", arch="80")

    class DummyTriton:
        runtime = SimpleNamespace(driver=SimpleNamespace(active=DummyActive()))

        @staticmethod
        def Config(values, *, num_stages, num_warps):
            return {"values": values, "num_stages": num_stages, "num_warps": num_warps}

    module = SimpleNamespace(
        triton=DummyTriton,
        triton_sta_kernel=SimpleNamespace(configs=["full-search"]),
    )

    monkeypatch.delenv("SPARSEVIDEO_STA_TRITON_AUTOTUNE", raising=False)
    _patch_a100_triton_autotune(module)

    assert module.triton_sta_kernel.configs == [
        {"values": {"BLOCK_Q": 128, "BLOCK_KV": 32}, "num_stages": 3, "num_warps": 4}
    ]


def test_sta_full_window_uses_dense_equivalent_without_loading_triton(monkeypatch):
    def fail_load():
        raise AssertionError("full-window STA should not load the Triton tile kernel")

    monkeypatch.setattr(sta_ops, "_owned_fastvideo_sta_triton", fail_load)
    monkeypatch.setattr(sta_ops, "_canvas_shape", lambda seq_shape: (1, 2, 8))
    monkeypatch.setattr(sta_ops, "STA_TILE_SIZE", (1, 2, 4))
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = _sliding_tile_attention_triton(
        q,
        k,
        v,
        window_size=[(1, 1, 2), (1, 1, 2)],
        text_length=0,
        has_text=False,
        seq_shape="18x48x80",
    )
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    torch.testing.assert_close(actual, expected)


def test_sta_text_full_window_requires_exact_text_length_for_dense_fast_path(monkeypatch):
    calls = {}

    def fake_load():
        def fake_triton(q, k, v, window_size, text_length, has_text, seq_shape, sm_scale=None):
            calls["used"] = True
            return q

        return fake_triton

    monkeypatch.setattr(sta_ops, "_owned_fastvideo_sta_triton", fake_load)
    q = torch.randn(1, 1, 8, 4)

    out = _sliding_tile_attention_triton(
        q,
        q,
        q,
        window_size=[(5, 6, 10)],
        text_length=4,
        has_text=True,
        seq_shape="30x48x80",
    )

    assert out is q
    assert calls == {"used": True}


def test_sta_triton_pads_non_power_of_two_head_dim(monkeypatch):
    calls = {}

    def fake_load():
        def fake_triton(q, k, v, window_size, text_length, has_text, seq_shape, sm_scale=None):
            calls["q_shape"] = tuple(q.shape)
            calls["sm_scale"] = sm_scale
            return torch.zeros_like(q)

        return fake_triton

    monkeypatch.setattr(sta_ops, "_owned_fastvideo_sta_triton", fake_load)
    q = torch.randn(1, 2, 768, 96)

    out = _sliding_tile_attention_triton(
        q,
        q,
        q,
        window_size=[(1, 1, 1), (1, 1, 1)],
        text_length=0,
        has_text=False,
        seq_shape="6x8x16",
    )

    assert _sta_triton_head_dim(96) == 128
    assert calls == {
        "q_shape": (1, 2, 768, 128),
        "sm_scale": pytest.approx(96 ** -0.5),
    }
    assert out.shape == q.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="STA device backend check requires CUDA")
def test_sta_current_a100_does_not_claim_h100_native_path():
    major, _minor = torch.cuda.get_device_capability()
    if major >= 9:
        pytest.skip("current CUDA device is Hopper or newer")

    q = torch.empty(1, device="cuda")

    assert sta_h100.sta_fwd is not None
    assert _can_use_h100_sta(q) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="STA A100 CUDA correctness requires CUDA")
def test_sta_a100_block_sparse_matches_masked_dense_small_shape():
    major, _minor = torch.cuda.get_device_capability()
    if major != 8:
        pytest.skip("STA A100 block-sparse CUDA path is for Ampere/A100")
    if not _can_use_a100_sta(torch.empty(1, device="cuda")):
        pytest.skip("SparseVideo-owned block-sparse CUDA backend is not available")

    def explicit_mask(shape, window, text_length):
        canvas_t, canvas_h, canvas_w = shape
        tile_t, tile_h, tile_w = STA_TILE_SIZE
        kernel_t, kernel_h, kernel_w = window
        total_tile = math.prod(STA_TILE_SIZE)
        img_len = canvas_t * canvas_h * canvas_w
        seq_len = img_len + text_length
        tile_h_count = canvas_h // tile_h
        tile_w_count = canvas_w // tile_w
        tile_t_count = canvas_t // tile_t
        q_idx = torch.arange(seq_len, device="cuda")[:, None]
        kv_idx = torch.arange(seq_len, device="cuda")[None, :]

        def coords(idx):
            tile_id = idx // total_tile
            return (
                tile_id // (tile_h_count * tile_w_count),
                (tile_id % (tile_h_count * tile_w_count)) // tile_w_count,
                tile_id % tile_w_count,
            )

        q_t, q_h, q_w = coords(q_idx.clamp(max=img_len - 1))
        kv_t, kv_h, kv_w = coords(kv_idx.clamp(max=img_len - 1))
        center_t = q_t.clamp(kernel_t // 2, (tile_t_count - 1) - kernel_t // 2)
        center_h = q_h.clamp(kernel_h // 2, (tile_h_count - 1) - kernel_h // 2)
        center_w = q_w.clamp(kernel_w // 2, (tile_w_count - 1) - kernel_w // 2)
        image = (
            (q_idx < img_len)
            & (kv_idx < img_len)
            & ((center_t - kv_t).abs() <= kernel_t // 2)
            & ((center_h - kv_h).abs() <= kernel_h // 2)
            & ((center_w - kv_w).abs() <= kernel_w // 2)
        )
        image_to_text = (q_idx < img_len) & (kv_idx >= img_len) & (kv_idx < img_len + text_length)
        text_to_all = (q_idx >= img_len) & (kv_idx < img_len + text_length)
        return image | image_to_text | text_to_all

    batch, heads, head_dim = 1, 2, 32
    shape = (6, 8, 16)
    text_length = 17
    seq_len = math.prod(shape) + text_length
    windows = [(1, 1, 1), (1, 1, 2)]
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = sliding_tile_attention(q, k, v, windows, text_length, True, "6x8x16")
    expected = []
    for head_idx, window in enumerate(windows):
        mask = explicit_mask(shape, window, text_length)
        scores = (q[:, head_idx].float() @ k[:, head_idx].float().transpose(-1, -2)) * (head_dim ** -0.5)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        expected.append((torch.softmax(scores, dim=-1) @ v[:, head_idx].float()).to(actual.dtype))
    expected = torch.stack(expected, dim=1)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2.5e-2, rtol=2.5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="STA A100 CUDA correctness requires CUDA")
def test_sta_a100_block_sparse_masks_partial_border_keys():
    major, _minor = torch.cuda.get_device_capability()
    if major != 8:
        pytest.skip("STA A100 block-sparse CUDA path is for Ampere/A100")
    if not _can_use_a100_sta(torch.empty(1, device="cuda")):
        pytest.skip("SparseVideo-owned block-sparse CUDA backend is not available")

    def explicit_mask(shape, window):
        canvas_t, canvas_h, canvas_w = shape
        tile_t, tile_h, tile_w = STA_TILE_SIZE
        kernel_t, kernel_h, kernel_w = window
        total_tile = math.prod(STA_TILE_SIZE)
        img_len = canvas_t * canvas_h * canvas_w
        tile_h_count = canvas_h // tile_h
        tile_w_count = canvas_w // tile_w
        tile_t_count = canvas_t // tile_t
        q_idx = torch.arange(img_len, device="cuda")[:, None]
        kv_idx = torch.arange(img_len, device="cuda")[None, :]

        def coords(idx):
            tile_id = idx // total_tile
            return (
                tile_id // (tile_h_count * tile_w_count),
                (tile_id % (tile_h_count * tile_w_count)) // tile_w_count,
                tile_id % tile_w_count,
            )

        q_t, q_h, q_w = coords(q_idx)
        kv_t, kv_h, kv_w = coords(kv_idx)
        center_t = q_t.clamp(kernel_t // 2, (tile_t_count - 1) - kernel_t // 2)
        center_h = q_h.clamp(kernel_h // 2, (tile_h_count - 1) - kernel_h // 2)
        center_w = q_w.clamp(kernel_w // 2, (tile_w_count - 1) - kernel_w // 2)
        return (
            ((center_t - kv_t).abs() <= kernel_t // 2)
            & ((center_h - kv_h).abs() <= kernel_h // 2)
            & ((center_w - kv_w).abs() <= kernel_w // 2)
        )

    batch, heads, head_dim = 1, 2, 32
    padded_shape = (6, 8, 16)
    source_shape = (5, 7, 15)
    seq_len = math.prod(padded_shape)
    windows = [(1, 1, 1), (1, 1, 2)]
    valid = _sta_a100_image_valid_mask_cpu("6x8x16", "5x7x15").bool().to("cuda")
    torch.manual_seed(1)
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = (0.1 * torch.randn_like(q.float())).to(torch.bfloat16)
    v = torch.randn_like(q)
    k[:, :, ~valid, :] = 8.0
    v[:, :, ~valid, :] = 32.0

    actual = sliding_tile_attention(
        q,
        k,
        v,
        windows,
        text_length=0,
        has_text=False,
        seq_shape="6x8x16",
        source_seq_shape="5x7x15",
    )
    expected = []
    for head_idx, window in enumerate(windows):
        mask = explicit_mask(padded_shape, window) & valid[None, :]
        scores = (q[:, head_idx].float() @ k[:, head_idx].float().transpose(-1, -2)) * (head_dim ** -0.5)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        expected.append((torch.softmax(scores, dim=-1) @ v[:, head_idx].float()).to(actual.dtype))
    expected = torch.stack(expected, dim=1)

    torch.testing.assert_close(
        actual[:, :, valid, :].float(),
        expected[:, :, valid, :].float(),
        atol=3.0e-2,
        rtol=3.0e-2,
    )


def test_sta_partial_border_path_passes_source_shape_and_skips_repair(monkeypatch):
    calls = []

    def fake_sliding_tile_attention(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape=None):
        calls.append(
            {
                "shape": tuple(q.shape),
                "seq_shape": seq_shape,
                "source_seq_shape": source_seq_shape,
            }
        )
        return q

    def fail_repair(*args, **kwargs):
        raise AssertionError("partial-border A100 path should not call dense repair")

    monkeypatch.setattr(sta_ops, "sliding_tile_attention", fake_sliding_tile_attention)
    monkeypatch.setattr("sparsevideo.methods.sta.method._sta_repair_padded_border_outputs", fail_repair)

    query = torch.arange(1 * (2 * 3 * 3) * 1 * 2, dtype=torch.float32).reshape(1, 18, 1, 2)
    out = _sta_sparsevideo_fastvideo_path(
        query,
        query,
        query,
        B=1,
        N=18,
        H=1,
        D=2,
        vid_start=0,
        video_len=18,
        text_len=0,
        context_len=0,
        T=2,
        spatial_h=3,
        spatial_w=3,
        T_pad=2,
        H_pad=4,
        W_pad=4,
        tile_size=(1, 2, 2),
        kernel_size=(1, 1, 1),
        model_type="wan",
        seq_shape_override="2x3x3",
        has_text_config=False,
        layer_idx=0,
        step_idx=0,
        mask_strategy=None,
    )

    assert calls == [{"shape": (1, 1, 32, 2), "seq_shape": "2x4x4", "source_seq_shape": "2x3x3"}]
    assert torch.equal(out, query)


def test_sta_wan_mask_strategy_matches_archive_branch_shape():
    strategy = _load_mask_strategy(_WAN_MASK_STRATEGY)

    assert len(strategy) == 50 * 40 * 40
    assert strategy["0_0_0"] == (3, 6, 10)
    assert strategy["49_39_39"] == (3, 1, 10)
    assert (
        _canonical_strategy_sha256(_WAN_MASK_STRATEGY)
        == "83eb66c3a3ff27fa076c329cd84a6a71a9d69eafb60d8c28720730d7f3745df5"
    )


def test_sta_hunyuan_mask_strategy_matches_archive_branch_shape():
    strategy = _load_mask_strategy(_HUNYUAN_MASK_STRATEGY)

    assert len(strategy) == 50 * 60 * 24
    assert strategy["0_0_0"] == (5, 6, 10)
    assert strategy["49_59_23"] == (1, 6, 10)
    assert (
        _canonical_strategy_sha256(_HUNYUAN_MASK_STRATEGY)
        == "1d04eb7e84c894f2c517c78e24608126bc8be3118e1f9347747ec16f536b4df0"
    )


def test_sta_runtime_rejects_training_free_mask_strategy_path():
    repo = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="training_free path"):
        _load_mask_strategy(repo / "training_free/FastVideo/docs/attention/sta/index.md")


def test_sta_window_sizes_are_selected_by_step_layer_head():
    strategy = {
        "0_0_0": (3, 6, 10),
        "0_0_1": (1, 5, 7),
    }

    assert _sta_window_sizes(strategy, step_idx=0, layer_idx=0, num_heads=3, default_window=(3, 3, 3)) == [
        (3, 6, 10),
        (1, 5, 7),
        (3, 3, 3),
    ]


def test_sta_search_tuning_writes_wan13_sized_strategy(tmp_path):
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    with (search_dir / "mask_search_prompt0.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(2):
            for layer in range(2):
                handle.write(json.dumps({
                    "step": step,
                    "layer": layer,
                    "L2_loss": {
                        "3,1,10": [0.2, 0.1],
                        "1,5,7": [0.1, 0.3],
                    },
                    "L1_loss": {
                        "3,1,10": [0.2, 0.1],
                        "1,5,7": [0.1, 0.3],
                    },
                }) + "\n")
    (search_dir / "metrics_gpu0.jsonl").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")

    output_file = tmp_path / "mask_strategy_wan13.json"
    summary = tune_search_results(
        search_dir,
        output_file,
        candidates=[(3, 1, 10), (1, 5, 7)],
        full_window=(3, 6, 10),
        skip_time_steps=1,
        timesteps=2,
        layers=2,
        heads=2,
    )
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert summary["entries"] == 8
    assert data["0_0_0"] == [3, 6, 10]
    assert data["1_0_0"] == [1, 5, 7]
    assert data["1_0_1"] == [3, 1, 10]
    assert summarize_strategy(output_file)["strategy_counts"]["3,6,10"] == 4


def test_sta_search_tuning_infers_shape_from_records(tmp_path):
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    with (search_dir / "mask_search_prompt0.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(3):
            for layer in range(2):
                handle.write(json.dumps({
                    "step": step,
                    "layer": layer,
                    "L2_loss": {
                        "3,1,10": [0.2, 0.1, 0.4],
                        "1,5,7": [0.1, 0.3, 0.2],
                    },
                }) + "\n")

    output_file = tmp_path / "mask_strategy_inferred.json"
    summary = tune_search_results(
        search_dir,
        output_file,
        candidates=[(3, 1, 10), (1, 5, 7)],
        skip_time_steps=1,
    )

    assert summary["timesteps"] == 3
    assert summary["layers"] == 2
    assert summary["heads"] == 3
    assert summary["entries"] == 18


def test_sta_search_tuning_uses_model_shape_defaults(tmp_path):
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / "mask_search_prompt0.jsonl").write_text(
        json.dumps({
            "step": 12,
            "layer": 0,
            "L2_loss": {
                "3,1,10": [0.2],
                "1,5,7": [0.1],
            },
        }) + "\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "mask_strategy_wan13.json"
    summary = tune_search_results(search_dir, output_file, model="wan1.3b")

    assert summary["timesteps"] == 50
    assert summary["layers"] == 30
    assert summary["heads"] == 12
    assert summary["entries"] == 50 * 30 * 12


def test_sta_search_model_defaults_cover_supported_backbones():
    assert model_strategy_shape("wan1.3b") == (50, 30, 12)
    assert model_strategy_shape("wan22") == (40, 40, 40)
    assert model_strategy_shape("hunyuan-i2v") == (50, 60, 24)
    assert model_strategy_shape("cogvideox") == (50, 42, 48)
    assert model_strategy_shape("ltx-i2v") == (50, 28, 32)
    assert model_strategy_shape("allegro") == (100, 32, 24)
    assert model_strategy_shape("mochi") == (64, 48, 24)
    assert model_strategy_shape("easyanimate") == (50, 48, 48)


def test_sta_search_uses_hunyuan_tuning_defaults(tmp_path):
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    with (search_dir / "mask_search_prompt0.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "step": 15,
            "layer": 0,
            "L2_loss": {
                "5,3,3": [0.2],
                "1,6,10": [0.1],
            },
        }) + "\n")

    output_file = tmp_path / "mask_strategy_hunyuan_small.json"
    summary = tune_search_results(
        search_dir,
        output_file,
        model="hunyuan",
        timesteps=16,
        layers=1,
        heads=1,
    )
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert model_search_defaults("hunyuan")["skip_time_steps"] == 15
    assert summary["skip_time_steps"] == 15
    assert data["14_0_0"] == [5, 6, 10]
    assert data["15_0_0"] == [1, 6, 10]


def test_sta_searching_mode_records_candidate_losses(monkeypatch, tmp_path):
    def fake_sta_attention(query, key, value, *, kernel_size, **kwargs):
        return torch.zeros_like(query) + float(kernel_size[0])

    monkeypatch.setattr("sparsevideo.methods.sta.method._sta_attention", fake_sta_attention)

    method = STAMethod(
        config={
            "STA_mode": "STA_searching",
            "mask_candidates": [[1, 1, 1], [2, 1, 1]],
            "mask_search_output_dir": str(tmp_path),
            "mask_search_prompt_id": "unit",
        },
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b"),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.zeros(1, 4, 2, 3)

    out = processor.attn_fn(query, query, query, None)
    method._mask_search_recorder.close()
    records = list(tmp_path.glob("mask_search_unit_*.jsonl"))
    payload = json.loads(records[0].read_text(encoding="utf-8").strip())

    assert torch.equal(out, torch.zeros_like(query) + 3)
    assert payload["step"] == 0
    assert payload["layer"] == 0
    assert set(payload["L2_loss"]) == {"1,1,1", "2,1,1"}
    assert payload["L2_loss"]["1,1,1"] == [4.0, 4.0]
    assert payload["L2_loss"]["2,1,1"] == [1.0, 1.0]


def test_sta_hunyuan_processor_passes_prompt_length_to_kernel_path(monkeypatch):
    calls = []

    def fake_sta_attention(query, key, value, **kwargs):
        calls.append(kwargs)
        return torch.empty_like(query)

    monkeypatch.setattr("sparsevideo.methods.sta.method._sta_attention", fake_sta_attention)

    method = STAMethod(
        config={},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=2,
        total_layers=60,
        original_processor=None,
        step_tracker=SimpleNamespace(step=10),
    )
    query = torch.randn(1, 10, 2, 4)

    processor.attn_fn(query, query, query, None, text_len=256, prompt_length=17)

    assert calls
    assert calls[0]["text_len"] == 256
    assert calls[0]["prompt_length"] == 17


def test_sta_hunyuan_kernel_uses_prompt_length_not_text_tail_length(monkeypatch):
    calls = []

    def fake_sliding_tile_attention(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape=None):
        calls.append(
            {
                "shape": tuple(q.shape),
                "text_length": text_length,
                "has_text": has_text,
                "seq_shape": seq_shape,
                "source_seq_shape": source_seq_shape,
            }
        )
        return q

    monkeypatch.setattr(sta_ops, "sliding_tile_attention", fake_sliding_tile_attention)

    query = torch.arange(1 * 12 * 1 * 2, dtype=torch.float32).reshape(1, 12, 1, 2)
    out = _sta_sparsevideo_fastvideo_path(
        query,
        query,
        query,
        B=1,
        N=12,
        H=1,
        D=2,
        vid_start=0,
        video_len=8,
        text_len=4,
        context_len=0,
        T=1,
        spatial_h=2,
        spatial_w=4,
        T_pad=1,
        H_pad=2,
        W_pad=4,
        tile_size=(1, 2, 2),
        kernel_size=(1, 1, 1),
        model_type="hunyuan_video",
        seq_shape_override="1x2x4",
        has_text_config=True,
        layer_idx=0,
        step_idx=0,
        mask_strategy=None,
        prompt_length=2,
    )

    assert calls == [
        {
            "shape": (1, 1, 12, 2),
            "text_length": 2,
            "has_text": True,
            "seq_shape": "1x2x4",
            "source_seq_shape": "1x2x4",
        }
    ]
    assert torch.equal(out, query)


def test_sta_hunyuan_text_tail_over_kernel_capacity_uses_dense_tail(monkeypatch):
    calls = []

    def fake_sliding_tile_attention(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape=None):
        calls.append(
            {
                "shape": tuple(q.shape),
                "text_length": text_length,
                "has_text": has_text,
                "seq_shape": seq_shape,
                "source_seq_shape": source_seq_shape,
            }
        )
        return q

    monkeypatch.setattr(sta_ops, "sliding_tile_attention", fake_sliding_tile_attention)

    query = torch.arange(1 * 12 * 1 * 2, dtype=torch.float32).reshape(1, 12, 1, 2)
    out = _sta_sparsevideo_fastvideo_path(
        query,
        query,
        query,
        B=1,
        N=12,
        H=1,
        D=2,
        vid_start=0,
        video_len=8,
        text_len=4,
        context_len=0,
        T=1,
        spatial_h=2,
        spatial_w=4,
        T_pad=1,
        H_pad=2,
        W_pad=4,
        tile_size=(1, 1, 1),
        kernel_size=(1, 1, 1),
        model_type="hunyuan_video",
        seq_shape_override="1x2x4",
        has_text_config=True,
        layer_idx=0,
        step_idx=0,
        mask_strategy=None,
        prompt_length=2,
    )

    assert calls == [
        {
            "shape": (1, 1, 9, 2),
            "text_length": 1,
            "has_text": True,
            "seq_shape": "1x2x4",
            "source_seq_shape": "1x2x4",
        }
    ]
    assert out.shape == query.shape


def test_sta_tile_untile_roundtrip_matches_fastvideo_layout():
    x = torch.arange(1 * 2 * (6 * 8 * 8) * 1).view(1, 2, 6 * 8 * 8, 1)

    tiled = _sta_tile_bhsd(x, (6, 8, 8), (3, 4, 4))
    restored = _sta_untile_bhsd(tiled, (6, 8, 8), (3, 4, 4))

    assert torch.equal(restored, x)
    assert not torch.equal(tiled, x)


def test_sta_text_path_keeps_upstream_hunyuan_only_boundary():
    q = torch.randn(1, 1, 8, 4)

    with pytest.raises(ValueError, match="text path is only defined"):
        sliding_tile_attention(
            q,
            q,
            q,
            window_size=[(3, 3, 5)],
            text_length=1,
            has_text=True,
            seq_shape="18x48x80",
        )


def test_sta_video_only_shape_boundary_is_strict():
    q = torch.randn(1, 1, 8, 4)

    with pytest.raises(ValueError, match="expected exactly 69120 image tokens"):
        sliding_tile_attention(
            q,
            q,
            q,
            window_size=[(3, 3, 5)],
            text_length=0,
            has_text=False,
            seq_shape="18x48x80",
        )
