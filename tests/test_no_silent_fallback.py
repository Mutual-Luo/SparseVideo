from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.sta.method import _sta_attention
from sparsevideo.methods.flashomni.method import _flashomni_attention, _flashomni_import
from sparsevideo.methods.radial.method import _radial_attention
from sparsevideo.methods.svg1.method import _svg_attention


def test_sta_sparse_stage_does_not_silently_run_cpu_fallback():
    query = torch.randn(1, 80, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        _sta_attention(
            query,
            key,
            value,
            tile_size=(1, 2, 2),
            kernel_size=(1, 1, 1),
            model_type="wan",
            text_len=0,
        )


def test_flashomni_upstream_does_not_silently_use_flex_on_cpu():
    query = torch.randn(1, 128, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        _flashomni_attention(
            query,
            key,
            value,
            sparse_kv_budget=0.5,
            sparse_block_size_for_q=64,
            sparse_block_size_for_kv=64,
            implementation="upstream",
            backend="auto",
            workspace_bytes=1024,
        )


def test_flashomni_missing_package_is_explicit(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "flashomni" else object())

    with pytest.raises(ImportError, match="requires the FlashOmni package"):
        _flashomni_import()


def test_svg1_sparse_stage_does_not_fallback_to_dense_without_video_tokens():
    query = torch.randn(1, 0, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="could not find video tokens"):
        _svg_attention(
            query,
            key,
            value,
            sparsity=0.25,
            num_sampled_rows=4,
            sample_mse_max_row=16,
            state={"block_mask": None, "profiled_step": -1},
            step_tracker_step=1,
            model_type="wan",
            text_len=0,
        )


def test_radial_sparse_stage_does_not_fallback_to_dense_without_video_tokens():
    query = torch.randn(1, 0, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="could not find video tokens"):
        _radial_attention(
            query,
            key,
            value,
            decay_factor=1,
            block_mask_cache={},
            model_type="wan",
            text_len=0,
        )
