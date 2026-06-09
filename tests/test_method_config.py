from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BACKBONE_KEYS = {
    "wan21-t2v-1.3b",
    "wan21-t2v-14b",
    "wan22-t2v-a14b",
    "hunyuan-t2v",
    "wan21-i2v-14b",
    "wan22-i2v-a14b",
    "hunyuan-i2v",
    "skyreels-v2-t2v-14b",
    "skyreels-v2-i2v-14b",
    "wan22-animate-14b",
    "wan21-vace-1.3b",
    "wan21-vace-14b",
    "cogvideox-t2v",
    "cogvideox-i2v",
    "ltx-video",
    "ltx-video-i2v",
    "allegro",
    "mochi-1",
    "easyanimate-v5-t2v-12b",
}


def _config_module(method: str):
    return importlib.import_module(f"sparsevideo.methods.{method}.config")


def _module_default_config(method: str, **context):
    return dict(_config_module(method).default_config(**context))


def test_public_methods_are_registered():
    assert sparsevideo.list_methods() == [
        "adacluster",
        "dense",
        "draft",
        "flashomni",
        "radial",
        "spargeattn",
        "sta",
        "svg1",
        "svg2",
        "svgear",
        "svoo",
    ]


def test_method_default_configs_are_backed_by_yaml_files():
    for method in sparsevideo.list_methods():
        path = REPO_ROOT / "src" / "sparsevideo" / "methods" / method / "config.yaml"
        assert path.exists()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert isinstance(data["defaults"], dict)
        assert "aliases" not in data
        assert isinstance(data.get("model_defaults", {}), dict)
        assert isinstance(data.get("compat_keys", []), list)

        config_module = importlib.import_module(f"sparsevideo.methods.{method}.config")
        assert config_module.CONFIG_DEFAULTS == data["defaults"]
        assert isinstance(config_module.CONFIG_ALIASES, dict)
        assert set(getattr(config_module, "CONFIG_COMPAT_KEYS", set())) == set(
            data.get("compat_keys", [])
        )


def test_method_configs_have_public_backbone_override_slots():
    for method in sparsevideo.list_methods():
        path = REPO_ROOT / "src" / "sparsevideo" / "methods" / method / "config.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_defaults = data.get("model_defaults", {})
        assert PUBLIC_BACKBONE_KEYS <= set(model_defaults), method


def test_method_configs_use_explicit_backbone_blocks_not_type_groups():
    runtime_type_groups = {"wan", "hunyuan_video", "cogvideox", "ltx_video", "mochi", "easyanimate"}
    for method in sparsevideo.list_methods():
        path = REPO_ROOT / "src" / "sparsevideo" / "methods" / method / "config.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_defaults = data.get("model_defaults", {})
        assert not runtime_type_groups & set(model_defaults), method
        for model_key in PUBLIC_BACKBONE_KEYS:
            if method == "dense":
                assert model_defaults[model_key] == {}, (method, model_key)
            else:
                assert model_defaults[model_key], (method, model_key)


def test_default_configs_select_backbone_by_model_key(monkeypatch):
    monkeypatch.delenv("SVOO_ENABLE_MEM_SAVE", raising=False)
    assert sparsevideo.default_method_config(
        "svg2",
        model_key="skyreels-v2-t2v-14b",
    ) == _module_default_config("svg2", model_key="skyreels-v2-t2v-14b")


def test_public_default_config_keys_are_exposed():
    required_keys = {
        "svg1": {"sparsity", "context_length", "prompt_length"},
        "svg2": {"top_p_kmeans", "context_length", "prompt_length"},
        "svgear": {"top_p_kmeans", "gamma", "context_length", "prompt_length"},
        "draft": {"sparsity_ratio"},
        "adacluster": {"topk_num"},
        "sta": {"tile_size", "window_size", "STA_mode"},
        "flashomni": {
            "implementation",
            "backend",
            "sparse_pattern",
            "sparse_size",
            "tau_q",
            "tau_kv",
            "N",
            "D",
            "S_q",
            "use_sparse_gemm",
            "taylor_cache_device",
        },
        "radial": {"decay_factor", "block_size", "use_sage_attention"},
        "spargeattn": {"topk", "cdfthreshd", "pvthreshd", "tensor_layout", "sim_rule"},
        "svoo": {"sparsity_csv_path", "enable_mem_save"},
    }

    for method, keys in required_keys.items():
        assert keys <= set(sparsevideo.default_method_config(method)), method

    assert "allow_triton_fallback" not in sparsevideo.default_method_config("svg2")
    assert "allow_triton_fallback" not in sparsevideo.default_method_config("draft")
    assert "allow_flex_fallback" not in sparsevideo.default_method_config("radial")
    assert "implementation" not in sparsevideo.default_method_config("svoo")
    assert "sparse_backend" not in sparsevideo.default_method_config("svoo")
    assert "context_length" not in sparsevideo.default_method_config("svoo")
    assert "prompt_length" not in sparsevideo.default_method_config("svoo")


def test_svoo_rejects_non_method_public_options():
    for key, value in (
        ("use_global_constraints", True),
        ("lambda_schedule", "cosine"),
        ("diverse_top_p_k", 0.1),
        ("use_fused_rope", False),
        ("context_length", 256),
        ("prompt_length", 128),
        ("implementation", "native"),
        ("sparse_backend", "flashinfer"),
    ):
        with pytest.raises(ValueError, match=key):
            sparsevideo.normalize_method_config("svoo", {key: value})


def test_dense_warmup_methods_accept_num_inference_steps_fallback():
    warmup_methods = {
        "adacluster",
        "draft",
        "flashomni",
        "radial",
        "spargeattn",
        "sta",
        "svg1",
        "svg2",
        "svoo",
    }
    for method in warmup_methods:
        assert sparsevideo.normalize_method_config(method, {"num_inference_steps": 37}) == {
            "num_inference_steps": 37,
        }
        assert sparsevideo.default_method_config(method, num_inference_steps=37)["num_inference_steps"] == 37

    with pytest.raises(ValueError, match="num_inference_steps"):
        sparsevideo.normalize_method_config("dense", {"num_inference_steps": 37})


def test_num_inference_steps_prefers_runtime_over_config_fallback():
    from sparsevideo.methods._schedule import runtime_or_config_num_inference_steps

    assert runtime_or_config_num_inference_steps(
        SimpleNamespace(num_inference_steps=lambda: 17),
        {"num_inference_steps": 37},
    ) == 17
    assert runtime_or_config_num_inference_steps(
        SimpleNamespace(),
        {"num_inference_steps": 37},
    ) == 37
    assert runtime_or_config_num_inference_steps(SimpleNamespace(), {}) is None


def test_svoo_enable_mem_save_follows_upstream_env_default(monkeypatch):
    from sparsevideo.methods.svoo.config import CONFIG_DEFAULTS, default_config

    monkeypatch.delenv("SVOO_ENABLE_MEM_SAVE", raising=False)
    assert default_config()["enable_mem_save"] is CONFIG_DEFAULTS["enable_mem_save"]

    monkeypatch.setenv("SVOO_ENABLE_MEM_SAVE", "0")
    assert default_config()["enable_mem_save"] is False

    monkeypatch.setenv("SVOO_ENABLE_MEM_SAVE", "1")
    assert default_config()["enable_mem_save"] is True


def test_model_key_default_configs_use_method_modules(monkeypatch):
    monkeypatch.delenv("SVOO_ENABLE_MEM_SAVE", raising=False)
    for method in sparsevideo.list_methods():
        for model_key in PUBLIC_BACKBONE_KEYS:
            assert sparsevideo.default_method_config(
                method,
                model_key=model_key,
            ) == _module_default_config(method, model_key=model_key)


def test_svoo_apply_api_defaults_use_model_context():
    from sparsevideo.methods.svoo import SVOOMethod
    from sparsevideo.methods.svoo.ops import resolve_sparsity_csv_path

    with pytest.raises(ValueError, match="start_reuse_step"):
        SVOOMethod(
            config={"start_reuse_step": 999},
            model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
        )
    with pytest.raises(ValueError, match="start_reuse_step"):
        sparsevideo.normalize_method_config("svoo", {"start_reuse_step": 999})
    with pytest.raises(ValueError, match="use_svoo"):
        sparsevideo.normalize_method_config("svoo", {"use_svoo": False})

    for model_key in sorted(PUBLIC_BACKBONE_KEYS):
        expected = _module_default_config("svoo", model_key=model_key)
        if int(expected.get("kmeans_iter_init", 0)) <= 0:
            continue
        if expected.get("use_dynamic_min_kc_ratio"):
            expected["sparsity_csv_path"] = str(
                resolve_sparsity_csv_path(expected["sparsity_csv_path"])
            )
        method = SVOOMethod(
            config={},
            model_info=SimpleNamespace(model_type="wan", model_key=model_key, transformers=[object()]),
        )
        assert method.config == expected

    with pytest.raises(ValueError, match="kmeans_iter_init > 0"):
        SVOOMethod(
            config={"kmeans_iter_init": 0},
            model_info=SimpleNamespace(
                model_type="cogvideox",
                model_key="cogvideox-t2v",
                transformers=[object()],
            ),
        )


def test_svoo_rejects_missing_dynamic_sparsity_csv():
    from sparsevideo.methods.svoo import SVOOMethod

    with pytest.raises(FileNotFoundError, match="sparsity_csv_path"):
        SVOOMethod(
            config={"sparsity_csv_path": "/tmp/sparsevideo-missing-svoo.csv"},
            model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
        )


def test_base_method_defaults_use_model_context():
    from sparsevideo.methods.adacluster import AdaClusterMethod
    from sparsevideo.methods.draft import DraftMethod
    from sparsevideo.methods.radial import RadialMethod
    from sparsevideo.methods.sta import STAMethod

    cases = (
        (AdaClusterMethod, "adacluster", "hunyuan_video", "hunyuan-t2v"),
        (DraftMethod, "draft", "wan", "wan21-t2v-14b"),
        (RadialMethod, "radial", "hunyuan_video", "hunyuan-t2v"),
        (STAMethod, "sta", "wan", "wan21-t2v-14b"),
        (STAMethod, "sta", "hunyuan_video", "hunyuan-t2v"),
    )

    for method_cls, method_name, model_type, model_key in cases:
        method = method_cls(
            config={},
            model_info=SimpleNamespace(model_type=model_type, model_key=model_key),
        )
        assert method.config == sparsevideo.default_method_config(method_name, model_key=model_key)


def test_draft_dense_switch_is_not_exposed_as_sparse_method():
    from sparsevideo.methods.draft import DraftMethod

    with pytest.raises(NotImplementedError, match="block_sparse_attention=False"):
        DraftMethod(
            config={"block_sparse_attention": False},
            model_info=SimpleNamespace(model_type="wan"),
        )


def test_local_aliases_are_not_primary_api():
    assert sparsevideo.normalize_method_config("svg2", {"top_p_kmeans": 0.8}) == {"top_p_kmeans": 0.8}
    assert sparsevideo.normalize_method_config("svoo", {"top_p_kmeans": 0.8}) == {"top_p_kmeans": 0.8}
    assert sparsevideo.normalize_method_config("spargeattn", {"value": 0.4}) == {"value": 0.4}
    assert sparsevideo.normalize_method_config("flashomni", {"sparse_kv_budget": 0.5}) == {"sparse_kv_budget": 0.5}
    assert sparsevideo.normalize_method_config("flashomni", {"cdfthreshd": 0.2}) == {"tau_kv": 0.2}
    assert sparsevideo.normalize_method_config("flashomni", {"tau_c": 0.3}) == {"tau_q": 0.3}

    local_aliases = {
        "svg1": {"skip_first_steps": 0.2},
        "svg2": {"budget": 0.2},
        "svoo": {"kmeans_iters": 2},
        "spargeattn": {"mask": "mask.pt"},
        "flashomni": {"pattern_source": "global_random"},
        "radial": {"skip_first_layers": 1},
        "sta": {"kernel_size": [3, 6, 10]},
        "adacluster": {"num_clusters": 200},
        "draft": {"budget": 0.2},
    }
    for method, config in local_aliases.items():
        with pytest.raises(ValueError):
            sparsevideo.normalize_method_config(method, config)

    with pytest.raises(ValueError):
        sparsevideo.normalize_method_config("adacluster", {"late_q_kernel_num": 250})


def test_720p_token_layout_inference_matches_upstream_shapes():
    from sparsevideo.methods._layout import infer_video_frame_count, infer_video_frame_shape

    assert infer_video_frame_count(21 * 45 * 80, model_type="wan") == 21
    assert infer_video_frame_shape(21 * 45 * 80, model_type="wan") == (21, 45, 80)
    assert infer_video_frame_shape(20 * 48 * 80, model_type="wan") == (20, 48, 80)
    assert infer_video_frame_shape(18 * 48 * 80, model_type="wan") == (18, 48, 80)
    assert infer_video_frame_shape(13 * 45 * 80, model_type="wan") == (13, 45, 80)
    assert infer_video_frame_shape(33 * 45 * 80, model_type="hunyuan_video") == (33, 45, 80)
    assert infer_video_frame_shape(30 * 48 * 80, model_type="hunyuan_video") == (30, 48, 80)
    assert infer_video_frame_shape(13 * 60 * 90, model_type="cogvideox") == (13, 60, 90)
    assert infer_video_frame_shape(22 * 45 * 80, model_type="allegro") == (22, 45, 80)
    assert infer_video_frame_shape(8 * 45 * 80, model_type="allegro") == (8, 45, 80)
    assert infer_video_frame_shape(4 * 45 * 80, model_type="mochi") == (4, 45, 80)
    assert infer_video_frame_shape(4 * 30 * 53, model_type="mochi") == (4, 30, 53)


def test_sta_seq_shape_override_must_match_video_tokens():
    from sparsevideo.methods.sta.method import _infer_video_shape

    assert _infer_video_shape(18 * 48 * 80, model_type="wan", seq_shape="18x48x80") == (18, 48, 80)
    with pytest.raises(ValueError, match="does not match"):
        _infer_video_shape(21 * 45 * 80, model_type="wan", seq_shape="18x48x80")


def test_svoo_hunyuan_text_padding_cluster_matches_upstream_policy():
    from sparsevideo.methods.svoo.text import pad_text_clusters
    import torch

    dynamic_map = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    q_sizes = torch.ones(1, 1, 2, dtype=torch.long)
    k_sizes = torch.ones(1, 1, 3, dtype=torch.long)
    q_sorted_indices = torch.arange(6, dtype=torch.int32).unsqueeze(0)

    full_map, q_sizes, k_sizes, q_sorted_indices = pad_text_clusters(
        dynamic_map, q_sizes, k_sizes, q_sorted_indices, text_len=4, prompt_length=3,
    )

    assert full_map.shape == (1, 1, 4, 5)
    assert q_sizes.tolist() == [[[1, 1, 3, 1]]]
    assert k_sizes.tolist() == [[[1, 1, 1, 3, 1]]]
    assert full_map[0, 0, :2, 3].all()
    assert not full_map[0, 0, :2, 4].any()
    assert full_map[0, 0, 2, :4].all()
    assert not full_map[0, 0, 2, 4]
    assert full_map[0, 0, 3, 4]
    assert not full_map[0, 0, 3, :4].any()
    assert q_sorted_indices.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    full_map, q_sizes, k_sizes, _ = pad_text_clusters(
        dynamic_map,
        torch.ones(1, 1, 2, dtype=torch.long),
        torch.ones(1, 1, 3, dtype=torch.long),
        torch.arange(6, dtype=torch.int32).unsqueeze(0),
        text_len=4,
        prompt_length=4,
    )
    assert full_map.shape == (1, 1, 4, 5)
    assert q_sizes.tolist() == [[[1, 1, 4, 0]]]
    assert k_sizes.tolist() == [[[1, 1, 1, 4, 0]]]
    assert full_map[0, 0, 3, 4]
