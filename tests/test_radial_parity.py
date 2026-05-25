from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import torch

from sparsevideo.methods.radial import RadialMethod
from sparsevideo.methods.radial.method import (
    _estimate_frame_size,
    _expand_attention_mask,
    _radial_attention,
    _radial_block_sizes,
    _radial_flashinfer_attention,
    _radial_sage_attention,
    _radial_bsr_mask,
    _radial_window_width,
    _sparge_mask_convert,
    _sparge_sage_qk_block_sizes,
)
from sparsevideo.kernels.flashinfer_block_sparse import build_bsr_from_mask
from sparsevideo.kernels.sageattention_runtime import _candidate_sageattention_roots


def _install_radial_upstream_attention_stubs(monkeypatch):
    def stub_module(name):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    sparse_sageattn = stub_module("sparse_sageattn")
    sparse_sageattn.sparse_sageattn = lambda *args, **kwargs: pytest.fail(
        "sparse_sageattn should not be used by FlashInfer parity path"
    )
    sageattention = stub_module("sageattention")
    sageattention.sageattn = lambda *args, **kwargs: pytest.fail(
        "sageattention should not be used by FlashInfer parity path"
    )
    spas_sage_attn = stub_module("spas_sage_attn")
    spas_sage_attn.block_sparse_sage2_attn_cuda = lambda *args, **kwargs: pytest.fail(
        "spas_sage_attn should not be used by FlashInfer parity path"
    )

    monkeypatch.setitem(sys.modules, "sparse_sageattn", sparse_sageattn)
    monkeypatch.setitem(sys.modules, "sageattention", sageattention)
    monkeypatch.setitem(sys.modules, "spas_sage_attn", spas_sage_attn)
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "training_free" / "radial-attention"))


def test_sageattention_env_root_cannot_select_external_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSEVIDEO_SAGEATTENTION_ROOT", str(tmp_path))

    with pytest.raises(ImportError, match="outside the SparseVideo-owned runtime root"):
        _candidate_sageattention_roots()


def test_sageattention_env_root_rejects_training_free_runtime(monkeypatch, tmp_path):
    upstream_root = tmp_path / "training_free" / "SageAttention"
    monkeypatch.setenv("SPARSEVIDEO_SAGEATTENTION_ROOT", str(upstream_root))

    with pytest.raises(ImportError, match="inside training_free"):
        _candidate_sageattention_roots()


def _reference_shrink_mask_strict(mask, block_size=128):
    seqlen = mask.shape[0]
    block_num = seqlen // block_size
    mask = mask[:block_num * block_size, :block_num * block_size].view(
        block_num, block_size, block_num, block_size,
    )
    col_densities = mask.sum(dim=1) / block_size
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (
        non_zero_densities.sum(dim=-1) + 1e-9
    )
    block_mask = frac_high_density_cols > 0.6
    block_mask[0:0] = True
    block_mask[-1:-1] = True
    return block_mask


def _reference_radial_bsr_mask(vid_len, block_size, frame_size, num_frames, decay_factor, model_type):
    final_log_mask = torch.zeros((vid_len // block_size, vid_len // block_size), dtype=torch.bool)
    token_per_frame = frame_size
    video_text_border = vid_len // block_size
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True

    col_indices = torch.arange(0, token_per_frame).view(1, -1)
    row_indices = torch.arange(0, token_per_frame).view(-1, 1)
    for i in range(num_frames):
        for j in range(num_frames):
            if j == 0 and model_type == "wan":
                local_mask = torch.ones((token_per_frame, token_per_frame), dtype=torch.bool)
            else:
                dist = abs(i - j)
                if model_type == "wan" and dist <= 1:
                    window_width = token_per_frame
                elif model_type == "hunyuan" and dist <= 1:
                    window_width = token_per_frame
                else:
                    group = dist.bit_length()
                    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
                    window_width = decay_length if decay_length >= block_size else block_size

                local_mask = torch.abs(col_indices - row_indices) <= window_width

                group = dist.bit_length()
                decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
                if decay_length < 128:
                    split_factor = int(128 / decay_length)
                    if dist % split_factor != 0:
                        local_mask = torch.zeros_like(local_mask)

            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            padded_local_mask = torch.zeros((all_length_row, all_length_col), dtype=torch.bool)
            padded_local_mask[
                remainder_row:remainder_row + token_per_frame,
                remainder_col:remainder_col + token_per_frame,
            ] = local_mask
            block_mask = _reference_shrink_mask_strict(padded_local_mask, block_size=block_size)
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] |= block_mask

    return final_log_mask


def test_radial_frame_size_includes_upstream_18_and_30_frame_layouts():
    assert _estimate_frame_size(18 * 48 * 80) == 48 * 80
    assert _estimate_frame_size(30 * 48 * 80) == 48 * 80


def test_radial_wan_first_frame_attention_sink_matches_upstream():
    mask = _radial_bsr_mask(
        vid_len=4 * 128,
        block_size=64,
        frame_size=128,
        num_frames=4,
        decay_factor=1.0,
        model_type="wan",
    )

    # For Wan, every query frame attends to key frame 0 as an attention sink.
    assert mask[6:8, 0:2].all()


def test_radial_hunyuan_does_not_use_wan_first_frame_sink():
    mask = _radial_bsr_mask(
        vid_len=4 * 128,
        block_size=64,
        frame_size=128,
        num_frames=4,
        decay_factor=1.0,
        model_type="hunyuan_video",
    )

    # Hunyuan keeps adjacent frames dense, but frame 3 -> frame 0 is governed by
    # radial split/window rules and is fully dropped for this small upstream case.
    assert not mask[6:8, 0:2].any()


def test_radial_window_width_matches_upstream_near_frame_policy():
    assert _radial_window_width(3, 2, 128, 1.0, 64, "wan") == 128
    assert _radial_window_width(3, 2, 128, 1.0, 64, "hunyuan") == 128
    assert _radial_window_width(3, 0, 128, 1.0, 64, "hunyuan") == 64


def test_radial_bsr_mask_matches_upstream_shrinked_reference():
    mask = _radial_bsr_mask(
        vid_len=6 * 160,
        block_size=64,
        frame_size=160,
        num_frames=6,
        decay_factor=0.95,
        model_type="hunyuan_video",
    )
    reference = _reference_radial_bsr_mask(
        vid_len=6 * 160,
        block_size=64,
        frame_size=160,
        num_frames=6,
        decay_factor=0.95,
        model_type="hunyuan",
    )

    assert torch.equal(mask, reference)


def test_radial_bsr_mask_keeps_partial_final_block():
    mask = _radial_bsr_mask(
        vid_len=3 * 130,
        block_size=128,
        frame_size=130,
        num_frames=3,
        decay_factor=0.95,
        model_type="hunyuan_video",
    )

    assert mask.shape == (4, 4)
    assert mask[-1].any()
    assert mask[:, -1].any()


def test_radial_block_sizes_use_partial_final_block():
    sizes = _radial_block_sizes(3 * 128 + 6, 128, torch.device("cpu"))

    assert sizes.tolist() == [128, 128, 128, 6]


def test_radial_bsr_conversion_keeps_indptr_and_indices_on_requested_device():
    mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ],
        dtype=torch.bool,
    )

    indptr, indices = build_bsr_from_mask(mask, torch.device("cpu"))

    assert indptr.device.type == "cpu"
    assert indices.device.type == "cpu"
    assert indptr.tolist() == [0, 2, 3]
    assert indices.tolist() == [0, 2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radial_bsr_conversion_accepts_cuda_masks_without_device_mixing():
    mask = torch.tensor(
        [
            [True, False],
            [True, True],
        ],
        dtype=torch.bool,
        device="cuda",
    )

    indptr, indices = build_bsr_from_mask(mask, torch.device("cuda"))

    assert indptr.device.type == "cuda"
    assert indices.device.type == "cuda"
    assert indptr.cpu().tolist() == [0, 1, 3]
    assert indices.cpu().tolist() == [0, 0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_radial_flashinfer_attention_matches_upstream_backend_cuda(monkeypatch):
    pytest.importorskip("flashinfer")

    from sparsevideo._runtime import _cuda_toolkit_status
    from sparsevideo.kernels.flashinfer_block_sparse import _ensure_cuda_home_for_flashinfer_jit

    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    _ensure_cuda_home_for_flashinfer_jit()
    _install_radial_upstream_attention_stubs(monkeypatch)
    from radial_attn.attn_mask import MaskMap, RadialAttention

    torch.manual_seed(0)
    batch, video_len, heads, head_dim = 1, 512, 2, 64
    block_size = 128
    query = torch.randn(batch, video_len, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    MaskMap._log_mask = None
    mask_map = MaskMap(video_token_num=video_len, num_frame=4)
    video_mask = mask_map.queryLogMask(
        query,
        "radial",
        block_size=block_size,
        decay_factor=0.5,
        model_type="hunyuan",
    )
    expected = RadialAttention(
        query,
        key,
        value,
        mask_map=mask_map,
        sparsity_type="radial",
        block_size=block_size,
        decay_factor=0.5,
        model_type="hunyuan",
        pre_defined_mask=None,
        use_sage_attention=False,
    ).reshape(batch, video_len, heads, head_dim)

    actual = _radial_flashinfer_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        0,
        block_size,
        pre_defined_mask=None,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_radial_flashinfer_attention_supports_partial_final_block_cuda():
    from sparsevideo._runtime import _cuda_toolkit_status
    from sparsevideo.kernels.flashinfer_block_sparse import HAS_FLASHINFER, _ensure_cuda_home_for_flashinfer_jit

    if not HAS_FLASHINFER:
        pytest.skip("flashinfer.sparse is not available")
    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    _ensure_cuda_home_for_flashinfer_jit()
    torch.manual_seed(2)
    batch, video_len, heads, head_dim = 1, 130, 2, 64
    block_size = 64
    query = torch.randn(batch, video_len, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    num_blocks = _radial_block_sizes(video_len, block_size, query.device).numel()
    video_mask = torch.ones(num_blocks, num_blocks, device="cuda", dtype=torch.bool)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    actual = _radial_flashinfer_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        0,
        block_size,
        pre_defined_mask=None,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_radial_flashinfer_attention_supports_partial_final_block_with_tail_cuda():
    from sparsevideo._runtime import _cuda_toolkit_status
    from sparsevideo.kernels.flashinfer_block_sparse import HAS_FLASHINFER, _ensure_cuda_home_for_flashinfer_jit

    if not HAS_FLASHINFER:
        pytest.skip("flashinfer.sparse is not available")
    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    _ensure_cuda_home_for_flashinfer_jit()
    torch.manual_seed(3)
    batch, video_len, tail_len, heads, head_dim = 1, 130, 5, 2, 64
    seq_len = video_len + tail_len
    block_size = 64
    query = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    num_blocks = _radial_block_sizes(video_len, block_size, query.device).numel()
    video_mask = torch.ones(num_blocks, num_blocks, device="cuda", dtype=torch.bool)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    actual = _radial_flashinfer_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        tail_len,
        block_size,
        pre_defined_mask=None,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_radial_hunyuan_attention_mask_expands_like_upstream_predefined_mask():
    attention_mask = torch.tensor([[[[True, True, False, True]]]])
    mask = _expand_attention_mask(attention_mask, sequence_length=4, device=torch.device("cpu"))

    assert mask.shape == (4, 4)
    assert mask.tolist() == [
        [True, True, False, True],
        [True, True, False, True],
        [True, True, False, True],
        [True, True, False, True],
    ]


def test_radial_sparge_mask_convert_matches_upstream_arch_layouts():
    mask = torch.tensor(
        [
            [True, False, False, True],
            [False, False, True, False],
            [True, True, False, False],
            [False, True, True, True],
        ]
    )

    torch.testing.assert_close(
        _sparge_mask_convert(mask, block_size=128, arch="sm80"),
        torch.repeat_interleave(mask, 2, dim=1),
    )
    torch.testing.assert_close(
        _sparge_mask_convert(mask, block_size=128, arch="sm90"),
        torch.repeat_interleave(mask, 2, dim=0),
    )
    torch.testing.assert_close(
        _sparge_mask_convert(mask, block_size=64, arch="sm80"),
        torch.max(mask.view(2, 2, 4), dim=1).values,
    )
    torch.testing.assert_close(
        _sparge_mask_convert(mask, block_size=64, arch="sm90"),
        torch.max(mask.view(4, 2, 2), dim=2).values,
    )


def test_radial_sparge_mask_convert_pads_odd_64_token_blocks():
    mask = torch.eye(3, dtype=torch.bool)

    sm80 = _sparge_mask_convert(mask, block_size=64, arch="sm80")
    sm90 = _sparge_mask_convert(mask, block_size=64, arch="sm90")

    assert sm80.shape == (2, 3)
    assert sm80[1, 2]
    assert sm90.shape == (3, 2)
    assert sm90[2, 1]


def test_radial_sage_kernel_block_shape_matches_owned_spas_runtime():
    assert _sparge_sage_qk_block_sizes("sm80") == (128, 64)
    assert _sparge_sage_qk_block_sizes("sm89") == (128, 64)
    assert _sparge_sage_qk_block_sizes("sm90") == (64, 128)


def test_radial_use_sage_attention_loads_owned_block_sparse_runtime(monkeypatch):
    sparse_sentinel = object()
    dense_sentinel = object()
    monkeypatch.setattr(
        "sparsevideo.methods.radial.method.load_block_sparse_sage2_attn_function",
        lambda: sparse_sentinel,
    )
    monkeypatch.setattr(
        "sparsevideo.methods.radial.method.load_sageattn_function",
        lambda: dense_sentinel,
    )

    method = RadialMethod(
        config={"use_sage_attention": True},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )

    assert method._block_sparse_sage2_attn_fn is sparse_sentinel
    assert method._sageattn_fn is dense_sentinel


def test_radial_use_sage_attention_accepts_partial_final_block(monkeypatch):
    captured = {}

    def fake_block_sparse_sage2_attn_cuda(query, key, value, mask_id, tensor_layout):
        captured["mask_id_shape"] = tuple(mask_id.shape)
        captured["tensor_layout"] = tensor_layout
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr("sparsevideo.methods.radial.method._cuda_arch", lambda device: "sm80")

    video_len = 21 * 45 * 80
    query = torch.randn(1, video_len, 2, 4)
    output = _radial_attention(
        query,
        query,
        query,
        decay_factor=0.2,
        block_mask_cache={},
        block_size=128,
        model_type="wan",
        use_sage_attention=True,
        block_sparse_sage2_attn_fn=fake_block_sparse_sage2_attn_cuda,
        sageattn_fn=object(),
    )

    assert output.shape == query.shape
    assert captured == {
        "mask_id_shape": (1, 2, 591, 1182),
        "tensor_layout": "HND",
    }


def test_radial_dense_warmup_ratio_is_only_step_gate(monkeypatch):
    calls = []

    def fake_radial(query, key, value, **kwargs):
        calls.append("dense" if kwargs.get("force_dense") else "sparse")
        return torch.empty_like(query)

    monkeypatch.setattr("sparsevideo.methods.radial.method._radial_attention", fake_radial)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    method = RadialMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    query = torch.randn(1, 128, 2, 64)

    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=50, timestep=11),
    )
    processor.attn_fn(query, query, query, None)

    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1, timestep=999),
    )
    processor.attn_fn(query, query, query, None)

    assert calls == ["sparse", "sparse"]


def test_radial_dense_layer_gate_uses_common_warmup_ratio(monkeypatch):
    calls = []

    def fake_radial(query, key, value, **kwargs):
        calls.append("dense" if kwargs.get("force_dense") else "sparse")
        return torch.empty_like(query)

    monkeypatch.setattr("sparsevideo.methods.radial.method._radial_attention", fake_radial)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    method = RadialMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.5},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    query = torch.randn(1, 128, 2, 64)

    processor = method.create_processor(
        layer_idx=0,
        total_layers=2,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    processor.attn_fn(query, query, query, None)

    processor = method.create_processor(
        layer_idx=1,
        total_layers=2,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    processor.attn_fn(query, query, query, None)

    assert calls == ["dense", "sparse"]


def test_radial_use_sage_dense_warmup_uses_owned_sageattention_nhd(monkeypatch):
    from sparsevideo.methods.radial.method import _radial_attention

    calls = {}

    def fake_sageattn(q, k, v, tensor_layout):
        calls["q_shape"] = tuple(q.shape)
        calls["k_shape"] = tuple(k.shape)
        calls["tensor_layout"] = tensor_layout
        return q + 1

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    query = torch.randn(1, 21 * 128, 2, 64)
    out = _radial_attention(
        query,
        query,
        query,
        decay_factor=0.2,
        block_mask_cache={},
        block_size=128,
        model_type="wan",
        use_sage_attention=True,
        block_sparse_sage2_attn_fn=object(),
        sageattn_fn=fake_sageattn,
        force_dense=True,
    )

    assert calls == {
        "q_shape": (1, 21 * 128, 2, 64),
        "k_shape": (1, 21 * 128, 2, 64),
        "tensor_layout": "NHD",
    }
    torch.testing.assert_close(out, query + 1)


def test_radial_sage_hunyuan_mask_keeps_text_kv_columns_like_upstream(monkeypatch):
    import sys
    import types

    captured = {}

    def fake_block_sparse_sage2_attn_cuda(query, key, value, mask_id, tensor_layout):
        captured["mask_id"] = mask_id.detach().clone()
        return torch.zeros_like(query)

    def fake_single_prefill_with_kv_cache(q, k, v, causal=False, return_lse=False, custom_mask=None):
        return torch.zeros_like(q)

    monkeypatch.setattr("sparsevideo.methods.radial.method._cuda_arch", lambda device: "sm80")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        types.SimpleNamespace(single_prefill_with_kv_cache=fake_single_prefill_with_kv_cache),
    )

    batch_size, video_len, tail_len, heads, dim = 1, 256, 128, 2, 4
    seq_len = video_len + tail_len
    query = torch.randn(batch_size, seq_len, heads, dim)
    key = torch.randn(batch_size, seq_len, heads, dim)
    value = torch.randn(batch_size, seq_len, heads, dim)
    video_mask = torch.eye(video_len // 128, dtype=torch.bool)
    pre_defined_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    _radial_sage_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        tail_len,
        128,
        pre_defined_mask,
        fake_block_sparse_sage2_attn_cuda,
    )

    # Upstream builds a full sequence block mask before sparge_mask_convert.
    # On sm80 with block_size=128 this repeats KV columns into 64-token blocks:
    # 384 valid tokens => 6 KV blocks, including the final two text blocks.
    assert captured["mask_id"].shape == (batch_size, heads, 2, 6)
    assert captured["mask_id"][:, :, :, -2:].all()


def test_radial_sage_hunyuan_partial_video_tail_crops_kv_blocks(monkeypatch):
    import sys

    captured = {}

    def fake_block_sparse_sage2_attn_cuda(query, key, value, mask_id, tensor_layout):
        captured["mask_id_shape"] = tuple(mask_id.shape)
        return torch.zeros_like(query)

    def fake_single_prefill_with_kv_cache(q, k, v, causal=False, return_lse=False, custom_mask=None):
        return torch.zeros_like(q)

    monkeypatch.setattr("sparsevideo.methods.radial.method._cuda_arch", lambda device: "sm80")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        types.SimpleNamespace(single_prefill_with_kv_cache=fake_single_prefill_with_kv_cache),
    )

    batch_size, video_len, tail_len, heads, dim = 1, 130, 5, 2, 4
    seq_len = video_len + tail_len
    query = torch.randn(batch_size, seq_len, heads, dim)
    key = torch.randn(batch_size, seq_len, heads, dim)
    value = torch.randn(batch_size, seq_len, heads, dim)
    video_mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
    pre_defined_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    output = _radial_sage_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        tail_len,
        128,
        pre_defined_mask,
        fake_block_sparse_sage2_attn_cuda,
    )

    assert output.shape == query.shape
    assert captured["mask_id_shape"] == (batch_size, heads, 2, 3)
