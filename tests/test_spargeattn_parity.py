from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sparsevideo.methods.spargeattn import SpargeAttnMethod
from sparsevideo.methods.spargeattn.method import (
    _clear_spas_sage_modules,
    _has_spas_sage_extensions,
    _is_training_free_runtime,
    _load_spas_sage_attn_functions,
    _resolve_torch_dtype,
    _sparge_dense_attention,
    _sparge_kernel_head_dim,
    _sparge_sparse_rejection_reason,
)
from sparsevideo.kernels.spas_sage_runtime import _candidate_spas_sage_attn_roots, load_spas_sage_attn_module
from sparsevideo.methods.spargeattn.config import default_config


def test_spargeattn_rejects_training_free_runtime_paths():
    module = SimpleNamespace(
        __file__=str(Path("/repo") / "training_free" / "SpargeAttn" / "spas_sage_attn" / "__init__.py")
    )

    assert _is_training_free_runtime(module)


def test_spargeattn_allows_non_training_free_runtime_paths():
    module = SimpleNamespace(__file__="/env/site-packages/spas_sage_attn/__init__.py")

    assert not _is_training_free_runtime(module)


def test_spargeattn_detects_local_owned_extensions(tmp_path):
    package = tmp_path / "spas_sage_attn"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_qattn.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    (package / "_fused.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")

    assert _has_spas_sage_extensions(tmp_path)


def test_spargeattn_loader_prefers_sparsevideo_owned_root(monkeypatch, tmp_path):
    package = tmp_path / "spas_sage_attn"
    package.mkdir()
    (package / "_qattn.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    (package / "_fused.cpython-312-x86_64-linux-gnu.so").write_bytes(b"")
    (package / "__init__.py").write_text(
        "def spas_sage2_attn_meansim_cuda(*args, **kwargs):\n"
        "    return 'cdf'\n"
        "def spas_sage2_attn_meansim_topk_cuda(*args, **kwargs):\n"
        "    return 'topk'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sparsevideo.kernels.spas_sage_runtime._candidate_spas_sage_attn_roots", lambda: [tmp_path])
    _clear_spas_sage_modules()

    cdf_fn, topk_fn = _load_spas_sage_attn_functions()

    assert cdf_fn() == "cdf"
    assert topk_fn() == "topk"


def test_spargeattn_env_root_cannot_select_external_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSEVIDEO_SPARGEATTN_ROOT", str(tmp_path))

    with pytest.raises(ImportError, match="outside the SparseVideo-owned runtime root"):
        _candidate_spas_sage_attn_roots()


def test_spargeattn_env_root_rejects_training_free_runtime(monkeypatch, tmp_path):
    upstream_root = tmp_path / "training_free" / "SpargeAttn"
    monkeypatch.setenv("SPARSEVIDEO_SPARGEATTN_ROOT", str(upstream_root))

    with pytest.raises(ImportError, match="inside training_free"):
        _candidate_spas_sage_attn_roots()


def test_spargeattn_loader_rejects_environment_runtime_without_owned_root(monkeypatch, tmp_path):
    env_root = tmp_path / "site-packages"
    package = env_root / "spas_sage_attn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("SELECTED = 'environment'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(env_root))
    monkeypatch.setattr("sparsevideo.kernels.spas_sage_runtime._candidate_spas_sage_attn_roots", lambda: [])
    _clear_spas_sage_modules()

    with pytest.raises(ImportError, match="SparseVideo-owned spas_sage_attn runtime"):
        load_spas_sage_attn_module()


def test_spargeattn_owned_runtime_sources_match_upstream_references():
    repo_root = Path(__file__).resolve().parents[1]
    owned_root = repo_root / "src/sparsevideo/kernels/native/spargeattn"
    upstream_root = repo_root / "training_free/SpargeAttn"

    for relative_path in ["LICENSE", "setup.py"]:
        assert (owned_root / relative_path).read_bytes() == (upstream_root / relative_path).read_bytes()

    for relative_path in ["spas_sage_attn", "csrc", "tools"]:
        owned = owned_root / relative_path
        upstream = upstream_root / relative_path
        owned_files = sorted(
            path.relative_to(owned)
            for path in owned.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".so")
        )
        upstream_files = sorted(
            path.relative_to(upstream)
            for path in upstream.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".so")
        )
        assert owned_files == upstream_files
        for source in owned_files:
            assert (owned / source).read_bytes() == (upstream / source).read_bytes()


def test_spargeattn_default_tuning_options_match_upstream_meansim_defaults():
    config = default_config()

    assert config["sim_rule"] == "l1"
    assert config["l1"] == 0.07
    assert config["pv_l1"] == 0.08
    assert config["cos_sim"] == 0.98
    assert config["rmse"] == 0.07
    assert config["rearrange_kwargs"] == {}
    assert config["tune_pv"] is True


def test_spargeattn_sparse_mode_rejects_when_upstream_conditions_fail(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf sparse path should not run")),
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("topk sparse path should not run")),
        ),
    )
    method = SpargeAttnMethod(
        config={"mode": "topk", "value": 0.5},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    query = torch.randn(1, 128, 2, 64)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attention_mask = torch.ones(1, 1, 128, 128, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="attention_mask is not supported"):
        processor.attn_fn(query, key, value, attention_mask)


def test_spargeattn_sparse_rejection_reason_reports_kernel_requirements():
    query = torch.randn(1, 4, 2, 8)

    assert _sparge_sparse_rejection_reason(query, None) == "query/key/value are not CUDA tensors"


def test_spargeattn_wan_full_mode_uses_upstream_dispatch_layout(monkeypatch):
    calls = {}

    def fake_dispatch(query, key, value, **kwargs):
        calls["query_shape"] = tuple(query.shape)
        calls["kwargs"] = kwargs
        return query + 2

    monkeypatch.setattr("sparsevideo.methods.spargeattn.method.dispatch_attention_fn", fake_dispatch)
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.F.scaled_dot_product_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Wan full branch should use dispatch")),
    )
    method = SpargeAttnMethod(
        config={"mode": "full"},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 7, 2, 4)
    attention_mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)

    out = processor.attn_fn(query, query, query, attention_mask)

    assert calls["query_shape"] == (1, 7, 2, 4)
    assert calls["kwargs"]["attn_mask"] is attention_mask
    assert calls["kwargs"]["dropout_p"] == 0.0
    assert calls["kwargs"]["is_causal"] is False
    torch.testing.assert_close(out, query + 2)


def test_spargeattn_wan_processor_default_fused_norm_rope():
    method = SpargeAttnMethod(
        config={"mode": "full"},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )

    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )

    assert processor.use_fused_qk_norm_rope is True


def test_spargeattn_wan_processor_can_disable_fused_norm_rope():
    method = SpargeAttnMethod(
        config={"mode": "full", "use_fused_qk_norm_rope": False},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )

    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )

    assert processor.use_fused_qk_norm_rope is False


def test_spargeattn_hunyuan_full_mode_uses_upstream_sdpa_layout(monkeypatch):
    calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        calls["query_shape"] = tuple(query.shape)
        calls["kwargs"] = kwargs
        return query + 3

    monkeypatch.setattr("sparsevideo.methods.spargeattn.method.F.scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.dispatch_attention_fn",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Hunyuan full branch should use SDPA")),
    )
    method = SpargeAttnMethod(
        config={"mode": "full"},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 7, 2, 4)
    attention_mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)

    out = processor.attn_fn(query, query, query, attention_mask)

    assert calls["query_shape"] == (1, 2, 7, 4)
    assert calls["kwargs"]["attn_mask"] is attention_mask
    assert calls["kwargs"]["dropout_p"] == 0.0
    assert calls["kwargs"]["is_causal"] is False
    torch.testing.assert_close(out, query + 3)


def test_spargeattn_allegro_dense_attention_matches_diffusers_sdpa():
    torch.manual_seed(23)
    query = torch.randn(2, 7, 4, 8)
    key = torch.randn(2, 7, 4, 8)
    value = torch.randn(2, 7, 4, 8)

    expected = F.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3).contiguous(),
        key.permute(0, 2, 1, 3).contiguous(),
        value.permute(0, 2, 1, 3).contiguous(),
        dropout_p=0.0,
        is_causal=False,
    ).permute(0, 2, 1, 3).contiguous()
    actual = _sparge_dense_attention(
        query, key, value, None, model_type="allegro"
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_spargeattn_hunyuan_processor_default_fused_norm_rope():
    method = SpargeAttnMethod(
        config={"mode": "full"},
        model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[]),
    )

    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )

    assert processor.use_fused_qk_norm_rope is True


def test_spargeattn_topk_path_forwards_upstream_kernel_names(monkeypatch):
    calls = {}

    def fake_topk(q, k, v, **kwargs):
        calls.update(kwargs)
        return torch.empty_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf path should not run")),
            fake_topk,
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    method = SpargeAttnMethod(
        config={"mode": "topk", "simthreshd1": -0.1, "pvthreshd": 42},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 64)

    processor.attn_fn(query, query, query, None)

    assert calls["topk"] == 0.5
    assert calls["simthreshd1"] == -0.1
    assert calls["pvthreshd"] == 42
    assert calls["tensor_layout"] == "HND"
    assert calls["is_causal"] is False


def test_spargeattn_sparse_path_forwards_cfg_batches(monkeypatch):
    calls = {}

    def fake_topk(q, k, v, **kwargs):
        calls["shape"] = tuple(q.shape)
        return torch.empty_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf path should not run")),
            fake_topk,
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    method = SpargeAttnMethod(
        config={"mode": "topk"},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(2, 128, 2, 64)

    processor.attn_fn(query, query, query, None)

    assert calls["shape"] == (2, 2, 128, 64)


def test_spargeattn_sparse_path_pads_non_kernel_head_dim(monkeypatch):
    calls = {}

    def fake_topk(q, k, v, **kwargs):
        calls["shape"] = tuple(q.shape)
        calls["scale"] = kwargs["scale"]
        return torch.empty_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf path should not run")),
            fake_topk,
        ),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    method = SpargeAttnMethod(
        config={"mode": "topk"},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 96)

    out = processor.attn_fn(query, query, query, None)

    assert _sparge_kernel_head_dim(96) == 128
    assert calls["shape"] == (1, 2, 128, 128)
    assert calls["scale"] == 96 ** -0.5
    assert out.shape == query.shape


def test_spargeattn_block_sparse_path_forwards_mask_id(monkeypatch):
    calls = {}

    def fake_block_sparse(q, k, v, **kwargs):
        calls.update(kwargs)
        return torch.empty_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cdf path should not run")),
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("topk path should not run")),
        ),
    )
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.load_block_sparse_sage2_attn_function",
        lambda: fake_block_sparse,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    mask_id = torch.ones(1, 2, 1, 2, dtype=torch.int32)
    method = SpargeAttnMethod(
        config={"mode": "block_sparse", "mask_id": mask_id},
        model_info=SimpleNamespace(model_type="wan", transformers=[]),
    )
    processor = method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 128, 2, 64)

    processor.attn_fn(query, query, query, None)

    assert calls["mask_id"] is mask_id
    assert calls["tensor_layout"] == "HND"
    assert "is_causal" not in calls


def test_spargeattn_resolves_upstream_output_dtype_names():
    assert _resolve_torch_dtype("float16") is torch.float16
    assert _resolve_torch_dtype("torch.bfloat16") is torch.bfloat16


class _FakeSparseAttentionMeansim:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_sparse = torch.nn.Parameter(torch.tensor([True]), requires_grad=False)
        self.cdfthreshd = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.simthreshd1 = torch.nn.Parameter(torch.tensor([-0.1]), requires_grad=False)
        self.simthreshd2 = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.pvthreshd = torch.nn.Parameter(torch.tensor([20.0]), requires_grad=False)


def test_spargeattn_tune_path_exports_upstream_named_state(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.load_sparse_attention_meansim_class",
        lambda: _FakeSparseAttentionMeansim,
    )
    method = SpargeAttnMethod(
        config={"tune": True, "l1": 0.07, "pv_l1": 0.08, "tune_pv": True},
        model_info=SimpleNamespace(
            model_type="wan",
            transformers=[],
            _self_attn_paths=[("transformer.blocks.0.attn1", object())],
        ),
    )

    method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    state = method.export_state_dict()
    tuned = method._tuned_attentions[0]

    assert set(state) == {
        "transformer.blocks.0.attn1.inner_attention.is_sparse",
        "transformer.blocks.0.attn1.inner_attention.cdfthreshd",
        "transformer.blocks.0.attn1.inner_attention.simthreshd1",
        "transformer.blocks.0.attn1.inner_attention.simthreshd2",
        "transformer.blocks.0.attn1.inner_attention.pvthreshd",
    }
    assert tuned.kwargs == {
        "sim_rule": "l1",
        "l1": 0.07,
        "pv_l1": 0.08,
        "cos_sim": 0.98,
        "rmse": 0.07,
        "rearrange_kwargs": {},
        "tune_pv": True,
    }


def test_spargeattn_tune_path_forwards_upstream_similarity_options(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.load_sparse_attention_meansim_class",
        lambda: _FakeSparseAttentionMeansim,
    )
    method = SpargeAttnMethod(
        config={
            "tune": True,
            "sim_rule": "cosine",
            "cos_sim": 0.99,
            "rmse": 0.05,
            "rearrange_kwargs": {"pattern": "b h l d -> b h l d"},
        },
        model_info=SimpleNamespace(
            model_type="wan",
            transformers=[],
            _self_attn_paths=[("transformer.blocks.0.attn1", object())],
        ),
    )

    method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    tuned = method._tuned_attentions[0]

    assert tuned.kwargs["sim_rule"] == "cosine"
    assert tuned.kwargs["cos_sim"] == 0.99
    assert tuned.kwargs["rmse"] == 0.05
    assert tuned.kwargs["rearrange_kwargs"] == {"pattern": "b h l d -> b h l d"}


def test_spargeattn_hunyuan_tune_defaults_match_upstream_wrapper(monkeypatch):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.load_sparse_attention_meansim_class",
        lambda: _FakeSparseAttentionMeansim,
    )
    method = SpargeAttnMethod(
        config={"tune": True},
        model_info=SimpleNamespace(
            model_type="hunyuan_video",
            transformers=[],
            _self_attn_paths=[("transformer.transformer_blocks.0.attn", object())],
        ),
    )

    method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    tuned = method._tuned_attentions[0]

    assert tuned.kwargs["l1"] == 0.07
    assert tuned.kwargs["pv_l1"] == 0.08
    assert tuned.kwargs["tune_pv"] is True


def test_spargeattn_tuned_state_loads_by_sparsevideo_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method.load_sparse_attention_meansim_class",
        lambda: _FakeSparseAttentionMeansim,
    )
    state_path = tmp_path / "state.pt"
    torch.save(
        {
            "transformer.blocks.0.attn1.inner_attention.is_sparse": torch.tensor([True, False]),
            "transformer.blocks.0.attn1.inner_attention.cdfthreshd": torch.tensor([0.3, 1.0]),
            "transformer.blocks.0.attn1.inner_attention.simthreshd1": torch.tensor([-0.2, 1.0]),
            "transformer.blocks.0.attn1.inner_attention.simthreshd2": torch.tensor([0.0, 0.0]),
            "transformer.blocks.0.attn1.inner_attention.pvthreshd": torch.tensor([18.0, 20.0]),
        },
        state_path,
    )
    method = SpargeAttnMethod(
        config={"model_out_path": str(state_path)},
        model_info=SimpleNamespace(
            model_type="wan",
            transformers=[],
            _self_attn_paths=[("transformer.blocks.0.attn1", object())],
        ),
    )

    method.create_processor(
        layer_idx=0,
        total_layers=1,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    tuned = method._tuned_attentions[0]

    torch.testing.assert_close(tuned.cdfthreshd, torch.tensor([0.3, 1.0]))
    torch.testing.assert_close(tuned.pvthreshd, torch.tensor([18.0, 20.0]))
