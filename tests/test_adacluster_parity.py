from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.adacluster.method import (
    AdaClusterMethod,
    _adacluster_attention,
    _adacluster_dense_attention,
    _adacluster_flashinfer_cluster_sparse_attn,
    _adacluster_hunyuan_kv_length,
    _adacluster_kernel_head_dim,
    _adacluster_reuse_policy,
    _adacluster_thresholded_kmeans_count,
    _adacluster_topk_from_qkv_minmax,
    _adacluster_trim_hunyuan_kv,
)


def test_wan_adacluster_dense_fallback_uses_upstream_sdpa_layout(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    calls = {}

    def fake_sdpa(q, k, v, **kwargs):
        calls["shape"] = q.shape
        calls["kwargs"] = kwargs
        return q

    monkeypatch.setattr(adacluster_method.F, "scaled_dot_product_attention", fake_sdpa)

    query = torch.randn(1, 4, 2, 3)
    out = _adacluster_dense_attention(query, query, query, model_type="wan")

    assert calls["shape"] == (1, 2, 4, 3)
    assert calls["kwargs"] == {"dropout_p": 0.0, "is_causal": False}
    assert out.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for upstream Triton AdaCluster kernels")
def test_wan_adacluster_full_cluster_mask_matches_dense_attention_cuda():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    batch, tokens, heads, head_dim = 1, 64, 2, 64
    query = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
        "q_kernel_num": None,
        "kv_kernel_num": None,
    }

    actual = _adacluster_attention(
        query,
        key,
        value,
        topk_num=tokens,
        q_kernel_num=tokens,
        kv_kernel_num=tokens,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        topk_policy="cluster_attn",
        reuse_prev_centroids=False,
        model_type="wan",
    )
    expected = _adacluster_dense_attention(query, key, value, model_type="wan")

    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_hunyuan_adacluster_dense_gate_uses_upstream_flash_attn_func(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    calls = {}

    def fake_flash_attn(q, k, v, **kwargs):
        calls["q_shape"] = q.shape
        calls["k_shape"] = k.shape
        calls["kwargs"] = kwargs
        return q

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(adacluster_method, "_load_flash_attn_func", lambda: fake_flash_attn)

    query = torch.randn(1, 5, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    out = _adacluster_dense_attention(query, key, key, model_type="hunyuan_video")

    assert calls == {
        "q_shape": (1, 5, 2, 4),
        "k_shape": (1, 3, 2, 4),
        "kwargs": {"causal": False, "softmax_scale": 0.5},
    }
    assert out.shape == query.shape


def _triton_kernel_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("triton_kernel."):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in imported:
                calls.add(node.func.id)
    return calls


def test_hunyuan_adacluster_dense_gate_is_only_common_warmup(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    calls = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(adacluster_method, "_adacluster_attention", lambda *args, **kwargs: calls.append("sparse") or args[0])

    method = AdaClusterMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=40,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 8, 2, 4)
    mask = torch.ones(1, 8)

    processor.attn_fn(query, query, query, mask)

    assert calls == ["sparse"]


def test_wan_adacluster_default_matches_runwan_fixed_cluster_path():
    from sparsevideo.methods.adacluster.config import default_config

    wan = default_config(model_key="wan21-t2v-1.3b")
    hunyuan = default_config(model_key="hunyuan-t2v")

    assert wan["topk_num"] == 128
    assert wan["q_kernel_num"] == 100
    assert wan["kv_kernel_num"] == 500
    assert wan["use_thresholded_kmeans_loop"] is False
    assert wan["initial_q_kernel_num"] == 50
    assert wan["initial_kv_kernel_num"] == 200
    assert wan["q_distance_threshold"] == 9.0
    assert wan["kv_distance_threshold"] == 5.5
    assert wan["thresholded_kmeans_iter_time"] == 3
    assert wan["thresholded_kmeans_max_iterations"] == 10
    assert hunyuan["use_thresholded_kmeans_loop"] is False


def test_runwan_reference_uses_fixed_cluster_defaults_not_thresholded_loop():
    repo = Path(__file__).resolve().parents[1]
    wan_path = repo / "training_free/Adacluster/runwan/wan/modules/model.py"
    tree = ast.parse(wan_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WanSelfAttention":
            forward = next(
                item for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "forward"
            )
            defaults = dict(zip(
                [arg.arg for arg in forward.args.args[-len(forward.args.defaults):]],
                forward.args.defaults,
            ))
            assert ast.literal_eval(defaults["topk_num"]) == 128
            assert ast.literal_eval(defaults["q_kernel_num"]) == 100
            assert ast.literal_eval(defaults["kv_kernel_num"]) == 500
            break
    else:
        raise AssertionError("WanSelfAttention.forward not found in Adacluster runwan model.py")

    assert "thresholded_kmeans_loop" not in wan_path.read_text(encoding="utf-8")


def test_adacluster_centroid_reuse_policies_match_upstream_models():
    assert _adacluster_reuse_policy(True) == (True, True)
    assert _adacluster_reuse_policy("both") == (True, True)
    assert _adacluster_reuse_policy("key") == (False, True)
    assert _adacluster_reuse_policy(False) == (False, False)


def test_hunyuan_adacluster_processor_does_not_reuse_centroids_like_upstream(monkeypatch):
    calls = []

    def fake_adacluster_attention(query, key, value, **kwargs):
        calls.append(
            {
                "topk_policy": kwargs["topk_policy"],
                "reuse_prev_centroids": kwargs["reuse_prev_centroids"],
            }
        )
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        "sparsevideo.methods.adacluster.method._adacluster_attention",
        fake_adacluster_attention,
    )

    method = AdaClusterMethod(
        config={},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=18,
        total_layers=40,
        original_processor=None,
        step_tracker=SimpleNamespace(step=9),
    )
    query = torch.randn(1, 32, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == [
        {
            "topk_policy": "minmax",
            "reuse_prev_centroids": False,
        }
    ]


def test_wan_adacluster_processor_default_uses_runwan_fixed_cluster_path(monkeypatch):
    calls = []

    def fake_adacluster_attention(query, key, value, **kwargs):
        calls.append(
            {
                "topk_num": kwargs["topk_num"],
                "q_kernel_num": kwargs["q_kernel_num"],
                "kv_kernel_num": kwargs["kv_kernel_num"],
                "topk_policy": kwargs["topk_policy"],
                "reuse_prev_centroids": kwargs["reuse_prev_centroids"],
                "thresholded_kmeans_config": kwargs["thresholded_kmeans_config"],
            }
        )
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        "sparsevideo.methods.adacluster.method._adacluster_attention",
        fake_adacluster_attention,
    )

    method = AdaClusterMethod(
        config={},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=2,
        total_layers=30,
        original_processor=None,
        step_tracker=SimpleNamespace(step=10),
    )
    query = torch.randn(1, 32, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == [
        {
            "topk_num": 128,
            "q_kernel_num": 100,
            "kv_kernel_num": 500,
            "topk_policy": "cluster_attn",
            "reuse_prev_centroids": "both",
            "thresholded_kmeans_config": None,
        }
    ]


def test_adacluster_owned_kernel_sources_match_upstream_references():
    repo = Path(__file__).resolve().parents[1]
    owned_kmeans = repo / "src/sparsevideo/kernels/native/adacluster/fast_kmeans_single.py"
    upstream_kmeans = repo / "training_free/Adacluster/triton_kernel/fast_kmeans_single.py"
    owned_attn = repo / "src/sparsevideo/kernels/native/adacluster/triton_cluster_sparse_attn.py"
    upstream_attn = repo / "training_free/Adacluster/triton_kernel/triton_cluster_sparse_attn.py"

    assert "training_free" not in owned_kmeans.parts
    assert "training_free" not in owned_attn.parts
    assert owned_kmeans.read_text(encoding="utf-8") == upstream_kmeans.read_text(encoding="utf-8")

    owned_text = owned_attn.read_text(encoding="utf-8")
    upstream_text = upstream_attn.read_text(encoding="utf-8")
    assert "SPARSEVIDEO_ADACLUSTER_PROFILE" in owned_text
    assert _normalize_adacluster_profile_guard(owned_text) == upstream_text


def _normalize_adacluster_profile_guard(text: str) -> str:
    text = text.replace("import os\n\nimport torch", "import torch")
    text = text.replace(
        '    profile = os.environ.get("SPARSEVIDEO_ADACLUSTER_PROFILE") == "1"\n'
        "    if profile:\n"
        "        start_time = torch.cuda.Event(enable_timing=True)\n"
        "        end_time = torch.cuda.Event(enable_timing=True)\n"
        "        start_time.record()\n",
        "    # 添加计时和统计\n"
        "    start_time = torch.cuda.Event(enable_timing=True)\n"
        "    end_time = torch.cuda.Event(enable_timing=True)\n"
        "    \n"
        "    start_time.record()\n",
    )
    text = text.replace(
        "    if profile:\n"
        "        end_time.record()\n"
        "        torch.cuda.synchronize()\n"
        '        print(f"Execution time: {start_time.elapsed_time(end_time)}ms")\n'
        '        print(f"Q counts distribution: {q_counts.cpu().numpy()}")\n',
        "    end_time.record()\n"
        "    torch.cuda.synchronize()\n"
        "    \n"
        '    print(f"Execution time: {start_time.elapsed_time(end_time)}ms")\n'
        '    print(f"Q counts distribution: {q_counts.cpu().numpy()}")\n',
    )
    return text


def test_adacluster_owned_kernel_set_matches_upstream_runtime_call_sites():
    repo = Path(__file__).resolve().parents[1]
    wan_path = repo / "training_free/Adacluster/runwan/wan/modules/model.py"
    hunyuan_path = (
        repo
        / "training_free/Adacluster/runhunyuan/modify_hunyuan_video"
        / "hunyuan_video_attn_processor_kvclus_withrightclusmaxclus.py"
    )

    assert _triton_kernel_calls(wan_path) == {
        "flash_kmeans_single",
        "triton_cluster_sparse_attn",
    }
    assert _triton_kernel_calls(hunyuan_path) == {
        "flash_kmeans_single",
        "triton_cluster_sparse_attn",
    }

    owned_sources = {
        path.name
        for path in (repo / "src/sparsevideo/kernels/native/adacluster").glob("*.py")
        if path.name != "__init__.py"
    }
    assert owned_sources == {
        "fast_kmeans_single.py",
        "triton_cluster_sparse_attn.py",
        "triton_cluster_sparse_attn_topk.py",
    }


def test_hunyuan_adacluster_minmax_topk_policy_matches_reference_formula():
    query = torch.tensor([[[1.0, -2.0], [-1.0, 3.0]]])
    key = torch.tensor([[[2.0, -1.0], [-3.0, 4.0], [0.5, 0.5]]])

    topk = _adacluster_topk_from_qkv_minmax(query, key, topk=1)

    q_pos = torch.clamp(query, min=0.0)
    q_neg = torch.clamp(query, max=0.0)
    k_pos = torch.clamp(key, min=0.0)
    k_neg = torch.clamp(key, max=0.0)
    expected_score = torch.matmul(q_pos, k_pos.transpose(-2, -1)) + torch.matmul(
        q_neg, k_neg.transpose(-2, -1)
    )
    expected = expected_score.topk(k=1, dim=-1).indices

    assert torch.equal(topk, expected)


def test_hunyuan_adacluster_attention_mask_trims_kv_like_upstream():
    key = torch.arange(1 * 6 * 1 * 2).reshape(1, 6, 1, 2)
    value = key + 100
    attention_mask = torch.tensor([[1, 1, 0, 1, 0, 0]])

    trimmed_key, trimmed_value = _adacluster_trim_hunyuan_kv(key, value, attention_mask)

    assert _adacluster_hunyuan_kv_length(attention_mask, max_length=6) == 3
    assert torch.equal(trimmed_key, key[:, :3])
    assert torch.equal(trimmed_value, value[:, :3])


def test_wan_adacluster_uses_thresholded_kernel_selection_on_first_call(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    threshold_calls = []
    kmeans_calls = []

    def fake_threshold(data, *, initial_clusters, iter_time, distance_threshold, max_iterations, num_heads):
        threshold_calls.append(
            {
                "initial_clusters": initial_clusters,
                "iter_time": iter_time,
                "distance_threshold": distance_threshold,
                "max_iterations": max_iterations,
                "num_heads": num_heads,
                "shape": tuple(data.shape),
            }
        )
        if initial_clusters == 200:
            return 4
        if initial_clusters == 50:
            return 3
        raise AssertionError(initial_clusters)

    def fake_flash_kmeans_single(kernel, data, iter_time):
        n_clusters = kernel.shape[2]
        kmeans_calls.append((n_clusters, iter_time, tuple(kernel.shape)))
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        centroids = torch.zeros_like(kernel)
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return centroids, sizes, labels

    def fake_cluster_sparse_attn(query, key, value, compressed_attn_mask, q_counts, kv_counts, sm_scale, selected_kv_indices=None):
        return query

    monkeypatch.setattr(adacluster_method, "_adacluster_thresholded_kmeans_count", fake_threshold)
    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    query = torch.randn(1, 8, 2, 4)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
        "q_kernel_num": None,
        "kv_kernel_num": None,
    }
    cfg = {
        "initial_q_kernel_num": 50,
        "initial_kv_kernel_num": 200,
        "q_distance_threshold": 9.0,
        "kv_distance_threshold": 5.5,
        "thresholded_kmeans_iter_time": 3,
        "thresholded_kmeans_max_iterations": 10,
    }

    _adacluster_attention(
        query,
        query,
        query,
        topk_num=2,
        q_kernel_num=100,
        kv_kernel_num=500,
        kmeans_iter_init=3,
        kmeans_iter_step=1,
        state=state,
        thresholded_kmeans_config=cfg,
    )

    assert threshold_calls == [
        {
            "initial_clusters": 200,
            "iter_time": 3,
            "distance_threshold": 5.5,
            "max_iterations": 10,
            "num_heads": 2,
            "shape": (2, 8, _adacluster_kernel_head_dim(4)),
        },
        {
            "initial_clusters": 50,
            "iter_time": 3,
            "distance_threshold": 9.0,
            "max_iterations": 10,
            "num_heads": 2,
            "shape": (2, 8, _adacluster_kernel_head_dim(4)),
        },
    ]
    assert state["q_kernel_num"] == 3
    assert state["kv_kernel_num"] == 4
    assert [call[0] for call in kmeans_calls] == [3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AdaCluster Triton kernel smoke requires CUDA")
def test_adacluster_owned_upstream_triton_path_executes_cuda():
    torch.manual_seed(0)
    query = torch.randn(1, 32, 1, 64, device="cuda", dtype=torch.float16)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
    }

    out = _adacluster_attention(
        query,
        query,
        query,
        topk_num=1,
        q_kernel_num=4,
        kv_kernel_num=4,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        topk_policy="cluster_attn",
        reuse_prev_centroids=True,
    )

    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.isfinite(out).all()
    assert state["centroids_init"] is True
    assert state["prev_q_centroids"].shape == (1, 1, 4, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AdaCluster Triton kernel comparison requires CUDA")
def test_adacluster_topk_kernel_matches_upstream_mask_kernel_cuda():
    from sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn import triton_cluster_sparse_attn
    from sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn_topk import triton_cluster_sparse_attn_topk

    torch.manual_seed(7)
    query = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    value = torch.randn(1, 1, 8, 16, device="cuda", dtype=torch.float16)
    q_counts = torch.tensor([[[4, 8]]], device="cuda", dtype=torch.int32)
    kv_counts = torch.tensor([[[3, 6, 8]]], device="cuda", dtype=torch.int32)
    selected = torch.tensor([[[[0, 2], [1, 2]]]], device="cuda", dtype=torch.int64)
    compressed = torch.zeros(1, 1, 2, 3, device="cuda", dtype=torch.bool)
    compressed.scatter_(dim=-1, index=selected, value=True)

    upstream = triton_cluster_sparse_attn(
        query=query,
        key=key,
        value=value,
        compressed_attn_mask=compressed,
        q_counts=q_counts,
        kv_counts=kv_counts,
        sm_scale=16 ** -0.5,
    )
    optimized = triton_cluster_sparse_attn_topk(
        query=query,
        key=key,
        value=value,
        selected_kv_indices=selected,
        q_counts=q_counts,
        kv_counts=kv_counts,
        sm_scale=16 ** -0.5,
    )

    assert torch.allclose(optimized, upstream, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AdaCluster FlashInfer comparison requires CUDA")
def test_adacluster_flashinfer_kernel_matches_topk_kernel_cuda():
    from sparsevideo.kernels.flashinfer_block_sparse import HAS_FLASHINFER
    from sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn_topk import triton_cluster_sparse_attn_topk

    if not HAS_FLASHINFER:
        pytest.skip("flashinfer is not installed")

    torch.manual_seed(11)
    query = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    value = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    q_sizes = torch.tensor([[[[32], [32]]]], device="cuda", dtype=torch.int32)
    kv_sizes = torch.tensor([[[[20], [20], [24]]]], device="cuda", dtype=torch.int32)
    q_counts = q_sizes.squeeze(-1).cumsum(dim=-1).contiguous()
    kv_counts = kv_sizes.squeeze(-1).cumsum(dim=-1).contiguous()
    selected = torch.tensor([[[[0, 2], [1, 2]]]], device="cuda", dtype=torch.int64)
    compressed = torch.zeros(1, 1, 2, 3, device="cuda", dtype=torch.bool)
    compressed.scatter_(dim=-1, index=selected, value=True)
    sm_scale = 0.1234

    expected = triton_cluster_sparse_attn_topk(
        query=query,
        key=key,
        value=value,
        selected_kv_indices=selected,
        q_counts=q_counts,
        kv_counts=kv_counts,
        sm_scale=sm_scale,
    )
    actual = _adacluster_flashinfer_cluster_sparse_attn(
        query=query,
        key=key,
        value=value,
        compressed_attn_mask=compressed,
        q_sizes=q_sizes,
        kv_sizes=kv_sizes,
        sm_scale=sm_scale,
    )

    assert actual is not None
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)


def test_adacluster_sparse_path_pads_non_kernel_head_dim(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    kmeans_calls = []
    attn_calls = {}

    def fake_flash_kmeans_single(kernel, data, iter_time):
        kmeans_calls.append((tuple(kernel.shape), tuple(data.shape), iter_time))
        n_clusters = kernel.shape[2]
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        centroids = torch.zeros_like(kernel)
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return centroids, sizes, labels

    def fake_cluster_sparse_attn(query, key, value, compressed_attn_mask, q_counts, kv_counts, sm_scale, selected_kv_indices=None):
        attn_calls["query_shape"] = tuple(query.shape)
        attn_calls["scale"] = sm_scale
        attn_calls["selected_shape"] = tuple(selected_kv_indices.shape)
        return query

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    query = torch.randn(1, 8, 2, 96)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
    }

    out = _adacluster_attention(
        query,
        query,
        query,
        topk_num=2,
        q_kernel_num=4,
        kv_kernel_num=4,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
    )

    assert _adacluster_kernel_head_dim(96) == 128
    assert kmeans_calls[0][0] == (1, 2, 4, 128)
    assert kmeans_calls[0][1] == (1, 2, 8, 128)
    assert attn_calls == {
        "query_shape": (1, 2, 8, 128),
        "scale": 96 ** -0.5,
        "selected_shape": (1, 2, 4, 2),
    }
    assert out.shape == query.shape


def test_adacluster_attention_prefers_flashinfer_variable_block_backend(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    calls = {}

    def fake_flash_kmeans_single(kernel, data, iter_time):
        n_clusters = kernel.shape[2]
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        centroids = torch.zeros_like(kernel)
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return centroids, sizes, labels

    def fake_flashinfer(query, key, value, compressed_attn_mask, q_sizes, kv_sizes, sm_scale):
        calls["query_shape"] = tuple(query.shape)
        calls["mask_shape"] = tuple(compressed_attn_mask.shape)
        calls["q_sizes_shape"] = tuple(q_sizes.shape)
        calls["kv_sizes_shape"] = tuple(kv_sizes.shape)
        calls["scale"] = sm_scale
        return query

    def fake_cluster_sparse_attn(*args, **kwargs):
        raise AssertionError("topk Triton fallback should not run when FlashInfer returns an output")

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_flashinfer_cluster_sparse_attn", fake_flashinfer)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    query = torch.randn(1, 8, 2, 64)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    backend_trace = []

    out = _adacluster_attention(
        query,
        query,
        query,
        topk_num=2,
        q_kernel_num=4,
        kv_kernel_num=4,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        backend_trace=backend_trace,
    )

    assert calls == {
        "query_shape": (1, 2, 8, 64),
        "mask_shape": (1, 2, 4, 4),
        "q_sizes_shape": (1, 2, 4, 1),
        "kv_sizes_shape": (1, 2, 4, 1),
        "scale": 64 ** -0.5,
    }
    assert backend_trace == ["variable_block_sparse_attn"]
    assert out.shape == query.shape


def test_adacluster_attention_falls_back_to_topk_when_flashinfer_unavailable(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    attn_calls = {}

    def fake_flash_kmeans_single(kernel, data, iter_time):
        n_clusters = kernel.shape[2]
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        centroids = torch.zeros_like(kernel)
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return centroids, sizes, labels

    def fake_cluster_sparse_attn(query, key, value, compressed_attn_mask, q_counts, kv_counts, sm_scale, selected_kv_indices=None):
        attn_calls["selected_shape"] = tuple(selected_kv_indices.shape)
        attn_calls["q_counts_shape"] = tuple(q_counts.shape)
        attn_calls["kv_counts_shape"] = tuple(kv_counts.shape)
        return query

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_flashinfer_cluster_sparse_attn", lambda *args, **kwargs: None)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    query = torch.randn(1, 8, 2, 64)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    backend_trace = []

    out = _adacluster_attention(
        query,
        query,
        query,
        topk_num=2,
        q_kernel_num=4,
        kv_kernel_num=4,
        kmeans_iter_init=1,
        kmeans_iter_step=1,
        state=state,
        backend_trace=backend_trace,
    )

    assert attn_calls == {
        "selected_shape": (1, 2, 4, 2),
        "q_counts_shape": (1, 2, 4),
        "kv_counts_shape": (1, 2, 4),
    }
    assert backend_trace == ["triton_cluster_sparse_attn_topk"]
    assert out.shape == query.shape


def test_wan_adacluster_thresholded_selection_rejects_full_attention_fallback(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    monkeypatch.setattr(
        adacluster_method,
        "_adacluster_thresholded_kmeans_count",
        lambda *args, **kwargs: -1,
    )

    query = torch.randn(1, 4, 1, 4)
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
        "q_kernel_num": None,
        "kv_kernel_num": None,
    }

    with pytest.raises(RuntimeError, match="dense fallback is controlled only by the common dense warmup ratios"):
        _adacluster_attention(
            query,
            query,
            query,
            topk_num=2,
            q_kernel_num=100,
            kv_kernel_num=500,
            kmeans_iter_init=3,
            kmeans_iter_step=1,
            state=state,
            thresholded_kmeans_config={
                "initial_q_kernel_num": 50,
                "initial_kv_kernel_num": 200,
                "q_distance_threshold": 9.0,
                "kv_distance_threshold": 5.5,
                "thresholded_kmeans_iter_time": 3,
                "thresholded_kmeans_max_iterations": 10,
            },
        )


def test_thresholded_kmeans_loop_returns_full_attention_when_cluster_cap_exceeded(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    def fake_flash_kmeans_single(kernel, data, iter_time):
        labels = torch.zeros(data.shape[0], data.shape[1], data.shape[2], dtype=torch.int32, device=data.device)
        sizes = torch.ones(data.shape[0], data.shape[1], kernel.shape[2], 1, dtype=torch.int32, device=data.device)
        return kernel.clone(), sizes, labels

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    data = torch.arange(2 * 9 * 4, dtype=torch.float32).reshape(2, 9, 4)
    count = _adacluster_thresholded_kmeans_count(
        data,
        initial_clusters=2,
        iter_time=3,
        distance_threshold=-1.0,
        max_iterations=10,
        num_heads=2,
    )

    assert count == -1


def test_thresholded_kmeans_loop_clamps_invalid_labels_like_upstream(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    def fake_flash_kmeans_single(kernel, data, iter_time):
        labels = torch.full(
            (data.shape[0], data.shape[1], data.shape[2]),
            kernel.shape[2] + 3,
            dtype=torch.int32,
            device=data.device,
        )
        sizes = torch.ones(data.shape[0], data.shape[1], kernel.shape[2], 1, dtype=torch.int32, device=data.device)
        return kernel.clone(), sizes, labels

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    data = torch.zeros(2, 9, 4)
    count = _adacluster_thresholded_kmeans_count(
        data,
        initial_clusters=2,
        iter_time=3,
        distance_threshold=1.0,
        max_iterations=10,
        num_heads=2,
    )

    assert count == 2


def test_adacluster_reuses_centroids_for_wan_and_reinitializes_for_hunyuan(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    calls = []

    def fake_flash_kmeans_single(kernel, data, iter_time):
        calls.append(
            {
                "max_iters": iter_time,
                "init_min": float(kernel.min().item()),
                "init_max": float(kernel.max().item()),
            }
        )
        n_clusters = kernel.shape[2]
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return kernel + 1000, sizes, labels

    def fake_cluster_sparse_attn(query, key, value, compressed_attn_mask, q_counts, kv_counts, sm_scale, selected_kv_indices=None):
        return query

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n - 1, -1, -1, device=device))

    query = torch.arange(1 * 4 * 1 * 2, dtype=torch.float32).reshape(1, 4, 1, 2)
    key = query + 20
    value = key + 20

    wan_state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    _adacluster_attention(query, key, value, 1, 2, 2, 3, 1, wan_state, reuse_prev_centroids=True)
    _adacluster_attention(query, key, value, 1, 2, 2, 3, 1, wan_state, reuse_prev_centroids=True)
    assert [call["max_iters"] for call in calls[:4]] == [3, 3, 1, 1]
    assert calls[2]["init_min"] >= 1000
    assert calls[3]["init_min"] >= 1000

    calls.clear()
    hunyuan_state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    _adacluster_attention(query, key, value, 1, 2, 2, 3, 1, hunyuan_state, reuse_prev_centroids=False)
    _adacluster_attention(query, key, value, 1, 2, 2, 3, 1, hunyuan_state, reuse_prev_centroids=False)
    assert [call["max_iters"] for call in calls] == [3, 3, 1, 1]
    assert calls[2]["init_max"] < 1000
    assert calls[3]["init_max"] < 1000


def test_adacluster_sparse_attention_allows_hunyuan_q_kv_length_mismatch(monkeypatch):
    import sparsevideo.methods.adacluster.method as adacluster_method

    def fake_flash_kmeans_single(kernel, data, iter_time):
        n_clusters = kernel.shape[2]
        labels = torch.arange(data.shape[2], device=data.device).remainder(n_clusters)
        labels = labels.expand(data.shape[0], data.shape[1], -1).int()
        sizes = torch.zeros(data.shape[0], data.shape[1], n_clusters, 1, dtype=torch.int32, device=data.device)
        sizes.scatter_add_(2, labels.long().unsqueeze(-1), torch.ones_like(labels, dtype=torch.int32).unsqueeze(-1))
        return kernel, sizes, labels

    def fake_cluster_sparse_attn(query, key, value, compressed_attn_mask, q_counts, kv_counts, sm_scale, selected_kv_indices=None):
        assert query.shape[2] == 5
        assert key.shape[2] == 3
        assert value.shape[2] == 3
        assert selected_kv_indices.shape == (1, 1, 2, 1)
        return query

    monkeypatch.setattr(adacluster_method, "_adacluster_flash_kmeans_single", fake_flash_kmeans_single)
    monkeypatch.setattr(adacluster_method, "_adacluster_cluster_sparse_attn", fake_cluster_sparse_attn)
    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.arange(n, device=device))

    query = torch.arange(1 * 5 * 1 * 2, dtype=torch.float32).reshape(1, 5, 1, 2)
    key = torch.arange(1 * 3 * 1 * 2, dtype=torch.float32).reshape(1, 3, 1, 2)
    value = key + 10
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = _adacluster_attention(
        query, key, value,
        topk_num=1,
        q_kernel_num=2,
        kv_kernel_num=2,
        kmeans_iter_init=3,
        kmeans_iter_step=1,
        state=state,
        topk_policy="minmax",
        reuse_prev_centroids=False,
    )

    assert out.shape == query.shape
