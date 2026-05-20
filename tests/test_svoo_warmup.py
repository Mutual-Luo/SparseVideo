from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.svoo.warmup import warmup_dimensions, warmup_svoo_kernels_from_pipeline


def _pipe(config, *, vae_temporal=4, vae_spatial=8, block_hidden_dim=1536):
    norm1 = SimpleNamespace(weight=torch.empty(block_hidden_dim))
    block = SimpleNamespace(norm1=norm1)
    transformer = SimpleNamespace(config=config, blocks=[block])
    return SimpleNamespace(
        transformer=transformer,
        vae_scale_factor_temporal=vae_temporal,
        vae_scale_factor_spatial=vae_spatial,
    )


def test_svoo_warmup_dimensions_match_wan_upstream_layout():
    pipe = _pipe(
        SimpleNamespace(num_attention_heads=12, attention_head_dim=128, patch_size=(1, 2, 2)),
        block_hidden_dim=1536,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="wan",
        height=720,
        width=1280,
        num_frames=81,
        config={},
    )

    assert dims["model_name"] == "Wan"
    assert dims["num_heads"] == 12
    assert dims["head_dim"] == 128
    assert dims["block_hidden_dim"] == 1536
    assert dims["norm_hidden_dim"] == 128
    assert dims["seq_len"] == 21 * 45 * 80
    assert dims["inverse_seq_len"] is None
    assert dims["include_wan_block_kernels"] is True
    assert dims["include_rmsnorm"] is True


def test_svoo_warmup_dimensions_match_hunyuan_upstream_layout():
    pipe = _pipe(
        SimpleNamespace(num_attention_heads=24, attention_head_dim=128),
        block_hidden_dim=3072,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="hunyuan_video",
        height=720,
        width=1280,
        num_frames=129,
        config={"context_length": 256},
    )

    assert dims["model_name"] == "HunyuanVideo 1.0"
    assert dims["num_heads"] == 24
    assert dims["head_dim"] == 128
    assert dims["seq_len"] == 33 * 3600
    assert dims["inverse_seq_len"] == 256 + 33 * 3600
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_dimensions_match_cogvideox_layout():
    pipe = _pipe(
        SimpleNamespace(
            num_attention_heads=48,
            attention_head_dim=64,
            patch_size=2,
            patch_size_t=None,
            max_text_seq_length=226,
        ),
        block_hidden_dim=3072,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="cogvideox",
        height=480,
        width=720,
        num_frames=13,
        config={},
    )

    assert dims["model_name"] == "CogVideoX"
    assert dims["num_heads"] == 48
    assert dims["head_dim"] == 64
    assert dims["seq_len"] == 4 * 30 * 45
    assert dims["inverse_seq_len"] == 226 + 4 * 30 * 45
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_dimensions_match_ltx_video_layout():
    pipe = _pipe(
        SimpleNamespace(
            num_attention_heads=32,
            attention_head_dim=64,
            patch_size=1,
            patch_size_t=1,
        ),
        vae_temporal=8,
        vae_spatial=32,
        block_hidden_dim=2048,
    )
    pipe.vae_temporal_compression_ratio = 8
    pipe.vae_spatial_compression_ratio = 32

    dims = warmup_dimensions(
        pipe,
        model_type="ltx_video",
        height=512,
        width=704,
        num_frames=161,
        config={},
    )

    assert dims["model_name"] == "LTX Video"
    assert dims["num_heads"] == 32
    assert dims["head_dim"] == 64
    assert dims["seq_len"] == 21 * 16 * 22
    assert dims["inverse_seq_len"] is None
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_dimensions_match_allegro_layout():
    pipe = _pipe(
        SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=96,
            patch_size=2,
            patch_size_t=1,
        ),
        vae_temporal=4,
        vae_spatial=8,
        block_hidden_dim=2304,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="allegro",
        height=720,
        width=1280,
        num_frames=88,
        config={},
    )

    assert dims["model_name"] == "Allegro"
    assert dims["num_heads"] == 24
    assert dims["head_dim"] == 96
    assert dims["seq_len"] == 22 * 45 * 80
    assert dims["inverse_seq_len"] is None
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_dimensions_match_mochi_layout():
    pipe = _pipe(
        SimpleNamespace(
            num_attention_heads=24,
            attention_head_dim=128,
            patch_size=2,
            max_sequence_length=256,
        ),
        vae_temporal=6,
        vae_spatial=8,
        block_hidden_dim=3072,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="mochi",
        height=480,
        width=848,
        num_frames=19,
        config={},
    )

    assert dims["model_name"] == "Mochi"
    assert dims["num_heads"] == 24
    assert dims["head_dim"] == 128
    assert dims["seq_len"] == 4 * 30 * 53
    assert dims["inverse_seq_len"] == 256 + 4 * 30 * 53
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_dimensions_match_easyanimate_layout():
    pipe = _pipe(
        SimpleNamespace(
            num_attention_heads=48,
            attention_head_dim=64,
            patch_size=2,
            max_text_seq_length=256,
        ),
        vae_temporal=4,
        vae_spatial=8,
        block_hidden_dim=3072,
    )

    dims = warmup_dimensions(
        pipe,
        model_type="easyanimate",
        height=512,
        width=512,
        num_frames=49,
        config={},
    )

    assert dims["model_name"] == "EasyAnimate"
    assert dims["num_heads"] == 48
    assert dims["head_dim"] == 64
    assert dims["seq_len"] == 13 * 32 * 32
    assert dims["inverse_seq_len"] == 256 + 13 * 32 * 32
    assert dims["include_wan_block_kernels"] is False
    assert dims["include_rmsnorm"] is False


def test_svoo_warmup_can_be_disabled_before_touching_cuda(monkeypatch):
    monkeypatch.setenv("SVOO_TRITON_WARMUP", "0")

    status = warmup_svoo_kernels_from_pipeline(
        None,
        model_type="wan",
        height=720,
        width=1280,
        num_frames=81,
        config={},
        dtype=torch.bfloat16,
        device="cuda",
    )

    assert status["enabled"] is False
    assert status["ran"] is False
    assert status["reason"] == "disabled"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_warmup_runs_owned_triton_kernels_on_cuda(monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("SVOO_TRITON_WARMUP", "1")
    monkeypatch.setenv("SVOO_TRITON_WARMUP_MODE", "compile")
    monkeypatch.setenv("SVOO_TRITON_WARMUP_SEQ_LEN", "16")
    monkeypatch.setenv("SVOO_TRITON_WARMUP_STRICT", "1")
    monkeypatch.setenv("SVOO_FLASHINFER_WARMUP", "0")

    pipe = _pipe(
        SimpleNamespace(num_attention_heads=2, attention_head_dim=16, patch_size=(1, 2, 2)),
        block_hidden_dim=32,
    )

    status = warmup_svoo_kernels_from_pipeline(
        pipe,
        model_type="wan",
        height=64,
        width=64,
        num_frames=1,
        config={
            "sparse_backend": "triton",
            "num_q_centroids": 1,
            "num_k_centroids": 1,
        },
        dtype=torch.float16,
        device="cuda",
    )
    torch.cuda.synchronize()

    assert status["enabled"] is True
    assert status["ran"] is True
    assert status["error"] is None
    assert status["seq_len"] == 16
    assert status["warmup_seq_len"] == 16
    assert status["sparse_backend"] == "triton"
