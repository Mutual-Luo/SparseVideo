from __future__ import annotations

from pathlib import Path
import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _install_sparse_videogen_cuvs_stub(monkeypatch):
    cuvs = ModuleType("cuvs")
    cluster = ModuleType("cuvs.cluster")
    kmeans = ModuleType("cuvs.cluster.kmeans")

    class KMeansParams:
        def __init__(self, *args, **kwargs):
            pass

    def fit(*args, **kwargs):
        raise AssertionError("cuVS must not be used by the SVG2 Euclidean kmeans parity path")

    kmeans.KMeansParams = KMeansParams
    kmeans.fit = fit
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.cluster", cluster)
    monkeypatch.setitem(sys.modules, "cuvs.cluster.kmeans", kmeans)


def _patch_flashinfer_sparse_identity(monkeypatch, captured=None):
    import sparsevideo.kernels.flashinfer_block_sparse as flashinfer_module

    def fake_variable_block_sparse_attn(q, k, v, dynamic_map, q_sizes, k_sizes):
        if captured is not None:
            captured["q"] = q.detach().clone()
            captured["q_sizes"] = q_sizes.detach().clone()
            captured["k_sizes"] = k_sizes.detach().clone()
            captured["dynamic_map"] = dynamic_map.detach().clone()
        return q

    monkeypatch.setattr(flashinfer_module, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_module, "variable_block_sparse_attn", fake_variable_block_sparse_attn)


def test_svg2_uses_shared_sparse_videogen_dynamic_map():
    from sparsevideo.kernels.dynamic_map import identify_dynamic_map
    from sparsevideo.methods.svg2 import method as svg2_method

    assert svg2_method.identify_dynamic_map is identify_dynamic_map


def test_svg2_uses_method_owned_sparse_videogen_kmeans_runtime():
    from sparsevideo.methods.svg2 import kmeans as svg2_kmeans

    source = inspect.getsource(svg2_kmeans)

    assert "training_free" not in str(Path(svg2_kmeans.__file__).parts)
    assert "batch_kmeans_Euclid" in source
    assert "triton_centroid_update_sorted_euclid" in source
    assert "_euclid_assign_kernel" in source


def test_svg2_dynamic_map_matches_upstream_shifted_top_p_boundary():
    from sparsevideo.methods.svg2.method import identify_dynamic_map
    from sparsevideo.kernels.dynamic_map import weighted_softmax

    query_centroids = torch.tensor([[[1.0, 0.0]]])
    key_centroids = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]]])
    q_sizes = torch.tensor([[4]])
    k_sizes = torch.tensor([[1, 1, 1]])

    scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (2**0.5)
    probs = weighted_softmax(scores, k_sizes.unsqueeze(-2))
    sorted_probs, _ = probs.sort(dim=-1, descending=True)
    top_p_on_boundary = sorted_probs[..., :2].sum().item()

    dynamic_map = identify_dynamic_map(
        query_centroids,
        key_centroids,
        q_sizes,
        k_sizes,
        p=top_p_on_boundary,
        min_kc_ratio=0,
    )

    assert dynamic_map.tolist() == [[[True, True, True]]]


def test_svg2_folds_classifier_free_batch_into_batch_head_slots(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method
    from sparsevideo.methods.svg2 import kmeans as svg2_kmeans

    kmeans_calls = []

    def fake_triton_kmeans(x, n_clusters, iter_time, init_centroids=None, final_reassign=False):
        kmeans_calls.append(tuple(x.shape))
        labels = torch.arange(x.shape[1], device=x.device).remainder(n_clusters)
        labels = labels.expand(x.shape[0], -1).int()
        centroids = torch.zeros(x.shape[0], n_clusters, x.shape[2], dtype=x.dtype, device=x.device)
        sizes = torch.ones(x.shape[0], n_clusters, 1, dtype=torch.int32, device=x.device)
        return labels, centroids, sizes

    monkeypatch.setattr(svg2_kmeans, "triton_kmeans", fake_triton_kmeans)

    query = torch.zeros(2, 8, 1, 4)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = svg2_method._svg2_attention(
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
        initialize_only=True,
    )

    assert out is None
    assert kmeans_calls == [(2, 8, 4), (2, 8, 4)]
    assert state["centroids_init"] is True


def test_svg2_wan_keeps_centroid_state_per_layer_like_upstream(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method
    from sparsevideo.methods.svg2 import method as svg2_method

    seen_states = []
    seen_centroids_init = []

    def fake_svg2_attention(query, key, value, **kwargs):
        state = kwargs["state"]
        seen_states.append(state)
        seen_centroids_init.append(state["centroids_init"])
        state["centroids_init"] = True
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg2_method, "_svg2_attention", fake_svg2_attention)

    method = SVG2Method(
        {"dense_warmup_layer_ratio": 0.0, "dense_warmup_step_ratio": 0.0},
        SimpleNamespace(model_type="wan", model_key=None),
    )
    step_tracker = SimpleNamespace(step=1, timestep=0)
    first = method.create_processor(0, 2, None, step_tracker)
    second = method.create_processor(1, 2, None, step_tracker)

    query = torch.zeros(1, 8, 1, 4)
    first.attn_fn(query, query, query, None)
    second.attn_fn(query, query, query, None)

    assert seen_states[0] is not seen_states[1]
    assert seen_centroids_init == [False, False]


def test_svg2_hunyuan_keeps_centroid_state_per_layer_like_upstream(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method
    from sparsevideo.methods.svg2 import method as svg2_method

    seen_states = []
    seen_centroids_init = []

    def fake_svg2_attention(query, key, value, **kwargs):
        state = kwargs["state"]
        seen_states.append(state)
        seen_centroids_init.append(state["centroids_init"])
        state["centroids_init"] = True
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg2_method, "_svg2_attention", fake_svg2_attention)

    method = SVG2Method(
        {"dense_warmup_layer_ratio": 0.0, "dense_warmup_step_ratio": 0.0},
        SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    step_tracker = SimpleNamespace(step=1, timestep=0)
    first = method.create_processor(0, 2, None, step_tracker)
    second = method.create_processor(1, 2, None, step_tracker)

    query = torch.zeros(1, 8, 1, 4)
    first.attn_fn(query, query, query, None)
    second.attn_fn(query, query, query, None)

    assert seen_states[0] is not seen_states[1]
    assert seen_centroids_init == [False, False]


def test_svg2_hunyuan_uses_config_prompt_length_fallback_like_upstream(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method
    from sparsevideo.methods.svg2 import method as svg2_method

    captured = {}

    def fake_svg2_attention(query, key, value, **kwargs):
        captured["prompt_length"] = kwargs["prompt_length"]
        captured["context_length"] = kwargs["context_length"]
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg2_method, "_svg2_attention", fake_svg2_attention)

    method = SVG2Method(
        {
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
            "context_length": 4,
            "prompt_length": 3,
        },
        SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    processor = method.create_processor(0, 2, None, SimpleNamespace(step=1, timestep=0))
    query = torch.zeros(1, 12, 1, 4)

    processor.attn_fn(query, query, query, None, text_len=4)

    assert captured == {"prompt_length": 3, "context_length": 4}


def test_svg2_hunyuan_runtime_prompt_length_overrides_config_fallback(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method
    from sparsevideo.methods.svg2 import method as svg2_method

    captured = {}

    def fake_svg2_attention(query, key, value, **kwargs):
        captured["prompt_length"] = kwargs["prompt_length"]
        return query

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg2_method, "_svg2_attention", fake_svg2_attention)

    method = SVG2Method(
        {
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
            "context_length": 4,
            "prompt_length": 4,
        },
        SimpleNamespace(model_type="hunyuan_video", model_key=None),
    )
    processor = method.create_processor(0, 2, None, SimpleNamespace(step=1, timestep=0))
    query = torch.zeros(1, 12, 1, 4)

    processor.attn_fn(query, query, query, None, text_len=4, prompt_length=2)

    assert captured == {"prompt_length": 2}


def test_svg2_method_uses_dense_warmup_step_ratio(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method
    from sparsevideo.methods.svg2 import method as svg2_method

    calls = []

    def fake_dense(query, key, value, attention_mask, *, model_type):
        calls.append("dense")
        return query

    def fake_sparse(*args, **kwargs):
        calls.append("sparse")
        return args[0]

    monkeypatch.setattr(svg2_method, "_svg2_dense_attention", fake_dense)
    monkeypatch.setattr(svg2_method, "_svg2_attention", fake_sparse)

    method = SVG2Method(
        {
            "dense_warmup_step_ratio": 0.5,
            "dense_warmup_layer_ratio": 0.0,
        },
        SimpleNamespace(model_type="wan", model_key=None),
    )
    step_tracker = SimpleNamespace(step=20, timestep=0, num_inference_steps=lambda: 50)
    processor = method.create_processor(0, 2, None, step_tracker)
    query = torch.zeros(1, 8, 1, 4)

    processor.attn_fn(query, query, query, None, timestep=torch.tensor([0]))

    assert calls == ["dense"]


def test_svg2_wan_full_attention_uses_upstream_sdpa_layout(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method

    calls = {}

    def fake_sdpa(q, k, v, **kwargs):
        calls["shape"] = q.shape
        calls["kwargs"] = kwargs
        return q

    monkeypatch.setattr(svg2_method.F, "scaled_dot_product_attention", fake_sdpa)

    query = torch.randn(1, 4, 2, 3)
    out = svg2_method._svg2_dense_attention(query, query, query, None, model_type="wan")

    assert calls["shape"] == (1, 2, 4, 3)
    assert calls["kwargs"] == {"dropout_p": 0.0, "is_causal": False}
    assert out.shape == query.shape


def test_svg2_hunyuan_full_attention_uses_upstream_flashinfer_varlen(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method

    calls = {}

    def fake_flashinfer(query, key, value, attention_mask):
        calls["q_shape"] = query.shape
        calls["mask_shape"] = attention_mask.shape
        calls["valid_len"] = int(attention_mask.sum().item())
        return torch.empty_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svg2_method, "_svg2_hunyuan_flashinfer_varlen", fake_flashinfer)

    query = torch.randn(1, 6, 2, 3)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    out = svg2_method._svg2_dense_attention(
        query, query, query, attention_mask, model_type="hunyuan_video",
    )

    assert calls == {
        "q_shape": (1, 6, 2, 3),
        "mask_shape": (1, 1, 1, 6),
        "valid_len": 4,
    }
    assert out.shape == query.shape


def test_svg2_hunyuan_flashinfer_varlen_uses_upstream_two_segment_mask(monkeypatch):
    from sparsevideo.kernels import flashinfer_block_sparse
    from sparsevideo.methods.svg2 import method as svg2_method

    calls = {}

    def fake_flashinfer(q, k, v, *, valid_len):
        calls["q_shape"] = q.shape
        calls["valid_len"] = valid_len
        return q

    monkeypatch.setattr(flashinfer_block_sparse, "hunyuan_flashinfer_varlen_attn", fake_flashinfer)

    query = torch.randn(1, 6, 2, 3)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    out = svg2_method._svg2_hunyuan_flashinfer_varlen(query, query, query, attention_mask)

    assert calls == {
        "q_shape": (2, 6, 3),
        "valid_len": 4,
    }
    assert out.shape == query.shape


def test_svg2_hunyuan_appends_prompt_and_fake_text_clusters_like_upstream(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method
    import sparsevideo.methods.svg2.kmeans as kmeans_module

    captured = {}

    def fake_kmeans(x, n_clusters, max_iters, init_centroids=None, **kwargs):
        assert kwargs["final_reassign"] is False
        batch_heads, tokens, dim = x.shape
        labels = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch_heads, -1) % n_clusters
        centroids = torch.zeros(batch_heads, n_clusters, dim, device=x.device, dtype=x.dtype)
        sizes = torch.zeros(batch_heads, n_clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.long))
        return labels, centroids, sizes

    def fake_dynamic_map(query_centroids, key_centroids, q_sizes, k_sizes, p, min_kc_ratio):
        out = torch.zeros(
            query_centroids.shape[0],
            query_centroids.shape[1],
            key_centroids.shape[1],
            dtype=torch.bool,
            device=query_centroids.device,
        )
        out[..., 0] = True
        return out

    monkeypatch.setattr(kmeans_module, "triton_kmeans", fake_kmeans)
    monkeypatch.setattr(svg2_method, "identify_dynamic_map", fake_dynamic_map)
    _patch_flashinfer_sparse_identity(monkeypatch, captured)

    query = torch.arange(1 * 12 * 1 * 1, dtype=torch.float32).reshape(1, 12, 1, 1)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = svg2_method._svg2_attention(
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
        model_type="hunyuan_video",
        text_len=4,
        prompt_length=3,
    )

    assert out.shape == query.shape
    assert torch.equal(out, query)
    assert captured["q"].flatten().tolist()[-4:] == [8.0, 9.0, 10.0, 11.0]
    assert captured["q_sizes"].tolist() == [[4, 4, 3, 1]]
    assert captured["k_sizes"].tolist() == [[3, 3, 2, 3, 1]]

    dynamic_map = captured["dynamic_map"][0]
    assert dynamic_map.shape == (4, 5)
    assert dynamic_map[-2, :-1].all()
    assert dynamic_map[:-1, -2].all()
    assert dynamic_map[-1, -1]
    assert not dynamic_map[-1, :-1].any()
    assert not dynamic_map[:-1, -1].any()


def test_svg2_hunyuan_rejects_context_length_mismatch_like_upstream_assertion():
    from sparsevideo.methods.svg2 import method as svg2_method

    query = torch.zeros(1, 12, 1, 4)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    with pytest.raises(RuntimeError, match="context_length must match"):
        svg2_method._svg2_attention(
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
            model_type="hunyuan_video",
            text_len=4,
            prompt_length=3,
            context_length=5,
        )


def test_svg2_kmeans_uses_upstream_no_final_reassign(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method
    import sparsevideo.methods.svg2.kmeans as kmeans_module

    final_reassign_values = []

    def fake_kmeans(x, n_clusters, max_iters, init_centroids=None, final_reassign=True, **kwargs):
        final_reassign_values.append(final_reassign)
        batch_heads, tokens, dim = x.shape
        labels = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch_heads, -1) % n_clusters
        centroids = torch.zeros(batch_heads, n_clusters, dim, device=x.device, dtype=x.dtype)
        sizes = torch.zeros(batch_heads, n_clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.long))
        return labels, centroids, sizes

    monkeypatch.setattr(kmeans_module, "triton_kmeans", fake_kmeans)
    monkeypatch.setattr(
        svg2_method,
        "identify_dynamic_map",
        lambda query_centroids, key_centroids, q_sizes, k_sizes, p, min_kc_ratio: torch.ones(
            query_centroids.shape[0],
            query_centroids.shape[1],
            key_centroids.shape[1],
            dtype=torch.bool,
            device=query_centroids.device,
        ),
    )
    _patch_flashinfer_sparse_identity(monkeypatch)

    query = torch.arange(1 * 8 * 1 * 1, dtype=torch.float32).reshape(1, 8, 1, 1)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = svg2_method._svg2_attention(
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
        model_type="wan",
    )

    assert out.shape == query.shape
    assert final_reassign_values == [False, False]


def test_svg2_kmeans_inputs_are_contiguous_like_upstream(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method
    import sparsevideo.methods.svg2.kmeans as kmeans_module

    seen = []

    def fake_kmeans(x, n_clusters, max_iters, init_centroids=None, final_reassign=True, **kwargs):
        seen.append((x.is_contiguous(), x.stride()))
        batch_heads, tokens, dim = x.shape
        labels = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch_heads, -1) % n_clusters
        centroids = torch.zeros(batch_heads, n_clusters, dim, device=x.device, dtype=x.dtype)
        sizes = torch.zeros(batch_heads, n_clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.long))
        return labels, centroids, sizes

    monkeypatch.setattr(kmeans_module, "triton_kmeans", fake_kmeans)
    monkeypatch.setattr(
        svg2_method,
        "identify_dynamic_map",
        lambda query_centroids, key_centroids, q_sizes, k_sizes, p, min_kc_ratio: torch.ones(
            query_centroids.shape[0],
            query_centroids.shape[1],
            key_centroids.shape[1],
            dtype=torch.bool,
            device=query_centroids.device,
        ),
    )
    _patch_flashinfer_sparse_identity(monkeypatch)

    query = torch.arange(1 * 8 * 2 * 4, dtype=torch.float32).reshape(1, 8, 2, 4)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = svg2_method._svg2_attention(
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
        model_type="wan",
    )

    assert out.shape == query.shape
    assert seen == [(True, (32, 4, 1)), (True, (32, 4, 1))]


def test_svg2_cuda_path_dispatches_owned_triton_permutation(monkeypatch):
    from sparsevideo.methods.svg2 import method as svg2_method
    import sparsevideo.methods.svg2.kmeans as kmeans_module
    import sparsevideo.kernels.permute as permute_module

    calls = []

    def fake_kmeans(x, n_clusters, max_iters, init_centroids=None, final_reassign=True, **kwargs):
        batch_heads, tokens, dim = x.shape
        labels = torch.arange(tokens, device=x.device).unsqueeze(0).expand(batch_heads, -1) % n_clusters
        centroids = torch.zeros(batch_heads, n_clusters, dim, device=x.device, dtype=x.dtype)
        sizes = torch.zeros(batch_heads, n_clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.long))
        return labels, centroids, sizes

    def fake_permute(tensor, labels, dim, *, sorted_indices=None):
        calls.append("permute")
        batch, heads, seq_len, dim_size = tensor.shape
        flat = tensor.reshape(batch * heads, seq_len, dim_size)
        if sorted_indices is None:
            sorted_indices = labels.long().argsort(dim=-1).to(torch.int32)
        gathered = torch.gather(
            flat,
            1,
            sorted_indices.long().unsqueeze(-1).expand(-1, -1, dim_size),
        )
        return gathered.reshape_as(tensor), sorted_indices

    def fake_inverse(tensor, sorted_indices, dim):
        calls.append("inverse")
        batch, heads, seq_len, dim_size = tensor.shape
        flat = tensor.reshape(batch * heads, seq_len, dim_size)
        out = torch.empty_like(flat)
        out.scatter_(
            1,
            sorted_indices.long().unsqueeze(-1).expand(-1, -1, dim_size),
            flat,
        )
        return out.reshape_as(tensor)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(kmeans_module, "triton_kmeans", fake_kmeans)
    monkeypatch.setattr(
        svg2_method,
        "identify_dynamic_map",
        lambda query_centroids, key_centroids, q_sizes, k_sizes, p, min_kc_ratio: torch.ones(
            query_centroids.shape[0],
            query_centroids.shape[1],
            key_centroids.shape[1],
            dtype=torch.bool,
            device=query_centroids.device,
        ),
    )
    _patch_flashinfer_sparse_identity(monkeypatch)
    monkeypatch.setattr(permute_module, "permute_tensor_by_labels_triton", fake_permute)
    monkeypatch.setattr(permute_module, "apply_inverse_permutation_triton", fake_inverse)

    query = torch.arange(1 * 8 * 1 * 1, dtype=torch.float32).reshape(1, 8, 1, 1)
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}

    out = svg2_method._svg2_attention(
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
        model_type="wan",
    )

    assert out.shape == query.shape
    assert calls == ["permute", "permute", "permute", "inverse"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_svg2_wan_sparse_attention_matches_upstream_kmeans_permutation_cuda(monkeypatch):
    pytest.importorskip("flashinfer.sparse")

    from sparsevideo._runtime import _cuda_toolkit_status

    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    _install_sparse_videogen_cuvs_stub(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "training_free" / "Sparse-VideoGen"))

    from sparsevideo.kernels.flashinfer_block_sparse import variable_block_sparse_attn
    from sparsevideo.methods.svg2.method import _svg2_attention
    from svg.kmeans_utils import batch_kmeans_Euclid, identify_dynamic_map
    from svg.kernels.triton.permute import (
        apply_inverse_permutation_triton,
        permute_tensor_by_labels_triton,
    )

    torch.manual_seed(0)
    batch, seq_len, heads, head_dim = 1, 128, 2, 64
    query = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    nqc = 8
    nkc = 12
    max_iters = 2
    top_p = 0.9
    min_kc_ratio = 0.1

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    q_flat = q_bhsd.reshape(batch * heads, seq_len, head_dim)
    k_flat = k_bhsd.reshape(batch * heads, seq_len, head_dim)

    torch.manual_seed(1234)
    q_labels, q_centroids, q_sizes, _ = batch_kmeans_Euclid(
        q_flat, n_clusters=nqc, max_iters=max_iters,
    )
    k_labels, k_centroids, k_sizes, _ = batch_kmeans_Euclid(
        k_flat, n_clusters=nkc, max_iters=max_iters,
    )
    dynamic_map = identify_dynamic_map(
        q_centroids.view(batch, heads, nqc, head_dim),
        k_centroids.view(batch, heads, nkc, head_dim),
        q_sizes.view(batch, heads, nqc),
        k_sizes.view(batch, heads, nkc),
        top_p,
        min_kc_ratio,
    )
    q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(q_bhsd, q_labels, dim=2)
    k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(k_bhsd, k_labels, dim=2)
    v_permuted, _ = permute_tensor_by_labels_triton(
        v_bhsd, k_labels, dim=2, sorted_indices=k_sorted_indices,
    )
    expected = variable_block_sparse_attn(
        q_permuted.reshape(batch * heads, seq_len, head_dim),
        k_permuted.reshape(batch * heads, seq_len, head_dim),
        v_permuted.reshape(batch * heads, seq_len, head_dim),
        dynamic_map.reshape(batch * heads, nqc, nkc),
        q_sizes.to(torch.int32),
        k_sizes.to(torch.int32),
    ).reshape(batch, heads, seq_len, head_dim)
    expected = apply_inverse_permutation_triton(
        expected, q_sorted_indices, dim=2,
    ).permute(0, 2, 1, 3).contiguous()

    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    torch.manual_seed(1234)
    actual = _svg2_attention(
        query,
        key,
        value,
        top_p,
        min_kc_ratio,
        nqc,
        nkc,
        max_iters,
        max_iters,
        state,
        model_type="wan",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
