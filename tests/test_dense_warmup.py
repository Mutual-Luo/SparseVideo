from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo

SPARSE_METHODS = [
    method for method in sparsevideo.list_methods()
    if method != "dense"
]


def test_clear_dense_warmup_ratio_names_are_supported_by_sparse_methods():
    for method in SPARSE_METHODS:
        config = sparsevideo.default_method_config(method)
        assert config["dense_warmup_step_ratio"] == 0.1
        assert config["dense_warmup_layer_ratio"] == 0.03
        assert sparsevideo.normalize_method_config(
            method,
            {
                "dense_warmup_step_ratio": 0.2,
                "dense_warmup_layer_ratio": 0.1,
            },
        ) == {
            "dense_warmup_step_ratio": 0.2,
            "dense_warmup_layer_ratio": 0.1,
        }


def test_dense_warmup_defaults_are_uniform_across_model_contexts():
    contexts = [
        {},
        {"model_key": "wan21-t2v-1.3b"},
        {"model_key": "wan21-t2v-14b"},
        {"model_key": "wan22-t2v-a14b"},
        {"model_key": "hunyuan-t2v"},
        {"model_key": "cogvideox-t2v"},
        {"model_key": "ltx-video"},
        {"model_key": "allegro"},
        {"model_key": "mochi-1"},
        {"model_key": "easyanimate-v5-t2v-12b"},
    ]

    for method in SPARSE_METHODS:
        for context in contexts:
            config = sparsevideo.default_method_config(method, **context)
            assert config["dense_warmup_step_ratio"] == 0.1
            assert config["dense_warmup_layer_ratio"] == 0.03


def test_legacy_svg_warmup_names_are_rejected():
    for method in ("svg1", "svg2", "svoo"):
        config = sparsevideo.default_method_config(method, model_key="wan21-t2v-1.3b")
        assert "first_times_fp" not in config
        assert "first_layers_fp" not in config
        with pytest.raises(ValueError, match="Unknown config keys"):
            sparsevideo.normalize_method_config(
                method,
                {
                    "first_times_fp": 0.2,
                    "first_layers_fp": 0.03,
                },
            )


def test_radial_legacy_dense_timestep_keys_are_rejected():
    for key in ("dense_timesteps", "dense_layers"):
        with pytest.raises(ValueError, match="Unknown config keys"):
            sparsevideo.normalize_method_config("radial", {key: 1})


def test_method_local_dense_switches_are_rejected():
    from sparsevideo.methods.spargeattn import SpargeAttnMethod

    with pytest.raises(ValueError, match="mode"):
        SpargeAttnMethod(
            config={"mode": "full"},
            model_info=SimpleNamespace(model_type="wan", model_key=None),
        )
    with pytest.raises(ValueError, match="Unknown config keys"):
        sparsevideo.normalize_method_config("flashomni", {"is_full": True})


def test_dense_warmup_ratio_helper_uses_first_step_and_layer_counts():
    from sparsevideo.methods._schedule import (
        configured_dense_warmup_layer_count,
        configured_dense_warmup_requires_dense,
    )

    config = {
        "dense_warmup_step_ratio": 0.2,
        "dense_warmup_layer_ratio": 0.1,
    }

    assert configured_dense_warmup_layer_count(config, 30) == 3
    assert configured_dense_warmup_requires_dense(config, 50, step=10)
    assert not configured_dense_warmup_requires_dense(config, 50, step=11)
    assert not configured_dense_warmup_requires_dense(config, None, step=10)
    assert configured_dense_warmup_requires_dense({"dense_warmup_step_ratio": 0.1}, 50, step=5)
    assert not configured_dense_warmup_requires_dense({"dense_warmup_step_ratio": 0.1}, 50, step=6)
    assert configured_dense_warmup_layer_count({"dense_warmup_layer_ratio": 0.03}, 30) == 1


def test_dense_warmup_ratio_zero_disables_svg_warmup(monkeypatch):
    from sparsevideo.methods.svg1 import method as svg1_method
    from sparsevideo.methods.svg1.method import SVG1Method

    calls = []

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        svg1_method,
        "_svg1_dense_attention",
        lambda *args, **kwargs: pytest.fail("dense warmup should be disabled"),
    )

    def fake_sparse(query, *args, **kwargs):
        calls.append("sparse")
        return query

    monkeypatch.setattr(svg1_method, "_svg_attention", fake_sparse)

    method = SVG1Method(
        {
            "dense_warmup_step_ratio": 0.0,
            "dense_warmup_layer_ratio": 0.0,
        },
        SimpleNamespace(model_type="wan", model_key=None),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1, timestep=926),
    )
    query = torch.zeros(1, 8, 1, 4)

    processor.attn_fn(query, query, query, None, timestep=torch.tensor([926]))

    assert calls == ["sparse"]


def test_common_dense_warmup_routes_non_svg_methods_to_dense(monkeypatch):
    from sparsevideo.methods.adacluster import AdaClusterMethod
    from sparsevideo.methods.draft import DraftMethod
    from sparsevideo.methods.flashomni import method as flashomni_method
    from sparsevideo.methods.flashomni import FlashOmniMethod
    from sparsevideo.methods.radial import method as radial_method
    from sparsevideo.methods.radial import RadialMethod
    from sparsevideo.methods.spargeattn import method as sparge_method
    from sparsevideo.methods.spargeattn import SpargeAttnMethod
    from sparsevideo.methods.sta import STAMethod

    monkeypatch.setattr(radial_method, "_radial_attention", lambda query, *args, **kwargs: query)
    monkeypatch.setattr(
        flashomni_method,
        "_flashomni_dense_warmup_attention",
        lambda query, *args, **kwargs: query,
    )
    monkeypatch.setattr(
        sparge_method,
        "_load_spas_sage_attn_functions",
        lambda: (
            lambda query, key, value, **kwargs: query,
            lambda query, key, value, **kwargs: query,
        ),
    )

    query = torch.zeros(1, 8, 1, 4)
    configs = {
        "adacluster": (AdaClusterMethod, {}),
        "draft": (DraftMethod, {}),
        "flashomni": (FlashOmniMethod, {}),
        "radial": (RadialMethod, {}),
        "spargeattn": (SpargeAttnMethod, {"mode": "topk"}),
        "sta": (STAMethod, {}),
    }

    for name, (method_cls, extra_config) in configs.items():
        config = {
            "dense_warmup_step_ratio": 1.0,
            "dense_warmup_layer_ratio": 0.0,
            **extra_config,
        }
        method = method_cls(config=config, model_info=SimpleNamespace(model_type="wan", model_key=None))
        processor = method.create_processor(
            layer_idx=5,
            total_layers=8,
            original_processor=None,
            step_tracker=SimpleNamespace(step=1, timestep=0),
        )

        out = processor.attn_fn(query, query, query, None)

        assert out.shape == query.shape
        assert method.runtime_summary()["dispatch_counts"]["dense"] == 1
