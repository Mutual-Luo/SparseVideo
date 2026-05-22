from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.adacluster import AdaClusterMethod
from sparsevideo.methods.draft import DraftMethod
from sparsevideo.methods.sta.method import _sta_attention
from sparsevideo.methods.flashomni import FlashOmniMethod
from sparsevideo.methods.flashomni.method import _flashomni_attention, _flashomni_import
from sparsevideo.methods.radial.method import _radial_attention
from sparsevideo.methods.spargeattn import SpargeAttnMethod
from sparsevideo.methods.svoo import SVOOMethod
from sparsevideo.methods.svg1.method import _svg_attention
from sparsevideo.methods.svg2.method import _svg2_attention


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


def test_sta_sparse_stage_rejects_non_fastvideo_tile_size(monkeypatch):
    query = torch.randn(1, 80, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    with pytest.raises(RuntimeError, match="requires FastVideo STA tile_size"):
        _sta_attention(
            query,
            key,
            value,
            tile_size=(1, 2, 2),
            kernel_size=(1, 1, 1),
            model_type="wan",
            text_len=0,
            seq_shape="1x8x10",
        )


def test_sta_sparse_stage_rejects_invalid_seq_shape(monkeypatch):
    query = torch.randn(1, 80, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    with pytest.raises(RuntimeError, match="could not infer video shape"):
        _sta_attention(
            query,
            key,
            value,
            tile_size=(6, 8, 8),
            kernel_size=(1, 1, 1),
            model_type="wan",
            text_len=0,
            seq_shape="not-a-shape",
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


def test_flashomni_global_random_stage_requires_cuda_before_native_dispatch():
    method = FlashOmniMethod(
        config={
            "sparse_pattern": "global_random",
            "dense_warmup_step_ratio": 0.0,
            "dense_warmup_layer_ratio": 0.0,
        },
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 8)

    with pytest.raises(RuntimeError, match="global_random sparse path requires CUDA"):
        processor.attn_fn(query, query, query, None)


def test_flashomni_missing_package_is_explicit(monkeypatch):
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._candidate_flashomni_roots", lambda: [])
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "flashomni" else object())

    with pytest.raises(ImportError, match="requires the SparseVideo-owned FlashOmni package"):
        _flashomni_import()


def test_draft_sparse_stage_does_not_silently_run_dense_on_cpu():
    method = DraftMethod(
        config={},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=1,
        total_layers=2,
        original_processor=None,
        step_tracker=SimpleNamespace(timestep=925),
    )
    query = torch.randn(1, 21 * 32 * 48, 1, 4)

    with pytest.raises(RuntimeError, match="draft sparse path requires CUDA"):
        processor.attn_fn(query, query, query, None)


def test_adacluster_sparse_stage_does_not_silently_run_dense_on_cpu():
    method = AdaClusterMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 8)

    with pytest.raises(RuntimeError, match="adacluster sparse path requires CUDA"):
        processor.attn_fn(query, query, query, None)


def test_svoo_sparse_stage_does_not_silently_run_dense_on_cpu():
    method = SVOOMethod(
        config={
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
            "use_dynamic_min_kc_ratio": False,
        },
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1, timestep=0),
    )
    query = torch.randn(1, 128, 2, 8)

    with pytest.raises(RuntimeError, match="svoo sparse path requires CUDA"):
        processor.attn_fn(query, query, query, None)


def test_spargeattn_sparse_mode_does_not_silently_run_dense(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf sparse path should not run")),
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("topk sparse path should not run")),
        ),
    )
    method = SpargeAttnMethod(
        config={"mode": "topk", "dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 64)

    with pytest.raises(RuntimeError, match="Use mode=full for the dense baseline"):
        processor.attn_fn(query, query, query, None)


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


def test_svg2_sparse_stage_does_not_silently_use_triton_fallback(monkeypatch):
    import sparsevideo.kernels.flashinfer_block_sparse as flashinfer_block_sparse
    import sparsevideo.methods.svg2.kmeans as kmeans_module
    from sparsevideo.methods.svg2 import method as svg2_method

    def fake_kmeans(x, n_clusters, max_iters, init_centroids=None, **kwargs):
        batch_heads, tokens, dim = x.shape
        labels = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch_heads, -1) % n_clusters
        centroids = torch.zeros(batch_heads, n_clusters, dim, device=x.device, dtype=x.dtype)
        sizes = torch.zeros(batch_heads, n_clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.long))
        return labels, centroids, sizes

    def fake_dynamic_map(query_centroids, key_centroids, q_sizes, k_sizes, p, min_kc_ratio):
        return torch.ones(
            query_centroids.shape[0],
            query_centroids.shape[1],
            key_centroids.shape[1],
            dtype=torch.bool,
            device=query_centroids.device,
        )

    monkeypatch.setattr(kmeans_module, "triton_kmeans", fake_kmeans)
    monkeypatch.setattr(svg2_method, "identify_dynamic_map", fake_dynamic_map)
    monkeypatch.setattr(flashinfer_block_sparse, "HAS_FLASHINFER", False)

    query = torch.randn(1, 8, 1, 4)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    with pytest.raises(RuntimeError, match="allow_triton_fallback"):
        _svg2_attention(
            query,
            query,
            query,
            top_p_kmeans=0.9,
            min_kc_ratio=0.1,
            num_q_centroids=2,
            num_k_centroids=3,
            kmeans_iter_init=1,
            kmeans_iter_step=1,
            state=state,
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


def test_radial_sparse_stage_does_not_silently_use_flex_fallback(monkeypatch):
    import sparsevideo.kernels.flashinfer_block_sparse as flashinfer_block_sparse

    query = torch.randn(1, 21 * 45 * 80, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(flashinfer_block_sparse, "HAS_FLASHINFER", False)

    with pytest.raises(RuntimeError, match="allow_flex_fallback"):
        _radial_attention(
            query,
            key,
            value,
            decay_factor=1,
            block_mask_cache={},
            model_type="wan",
            text_len=0,
        )
