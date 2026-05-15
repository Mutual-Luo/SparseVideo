from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "infer.py"


def _load_infer_module():
    spec = importlib.util.spec_from_file_location("sparsevideo_infer_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_infer_dry_run(tmp_path: Path, *args: str) -> dict:
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_infer(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *args,
            "--metrics-file",
            str(tmp_path / "metrics.jsonl"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_infer_dry_run_resolves_wan_svoo_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "svoo")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["fps"] == 16
    assert payload["wan_flow_shift"] == 5.0
    assert cfg["implementation"] == "native"
    assert cfg["sparse_backend"] == "flashinfer"
    assert cfg["num_q_centroids"] == 256
    assert cfg["num_k_centroids"] == 1024
    assert cfg["kmeans_iter_init"] == 2
    assert cfg["kmeans_iter_step"] == 2
    assert cfg["use_dynamic_min_kc_ratio"] is True
    assert cfg["sparsity_csv_path"].endswith("sparsity_wan_1.3B_t2v.csv")
    assert "optional_kernels" in payload["runtime"]
    assert "cuda_available" in payload["runtime"]["torch"]
    assert set(payload["runtime"]["preflight"]) == {"errors", "warnings"}
    assert "svg_svoo_fused_kernels" in payload["runtime"]["optional_kernels"]


def test_infer_dry_run_resolves_wan_svg2_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "svg2")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["wan_flow_shift"] == 5.0
    assert cfg["num_q_centroids"] == 300
    assert cfg["num_k_centroids"] == 1000
    assert cfg["min_kc_ratio"] == 0.10
    assert cfg["kmeans_iter_init"] == 50
    assert cfg["kmeans_iter_step"] == 2


def test_infer_dry_run_resolves_hunyuan_svoo_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "hunyuan", "--method", "svoo")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["num_frames"] == 129
    assert payload["fps"] == 24
    assert cfg["top_p_kmeans"] == 0.88
    assert cfg["start_reuse_step"] == 6
    assert cfg["reuse_interval"] == 50
    assert cfg["sparsity_csv_path"].endswith("sparsity_hunyuan10_13B_t2v.csv")


def test_infer_dry_run_flashomni_defaults_use_upstream_kernel(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "flashomni")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert cfg["implementation"] == "upstream"
    assert cfg["backend"] == "auto"
    assert cfg["workspace_bytes"] == 268435456
    assert payload["runtime"]["optional_kernels"]["flashomni"]["methods"] == ["flashomni"]


def test_infer_dry_run_warns_when_method_config_is_dense_mode(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "spargeattn")

    assert any("mode=full runs dense attention" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_infer_dry_run_reports_sta_native_shape_boundary(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "sta")
    cfg = payload["method_config"]

    assert cfg["tile_size"] == [6, 8, 8]
    assert cfg["window_size"] == [3, 3, 5]
    assert any("FastVideo C++ native STA" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_infer_preflight_fails_before_model_load_for_cpu_sparse_method(tmp_path):
    result = _run_infer(tmp_path, "--model", "wan1.3b", "--method", "svoo", "--device", "cpu")
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "preflight"
    assert "Sparse methods require --device cuda" in payload["error"]
    assert payload["timings"] == {}


def test_preflight_reports_required_flashomni_kernel_missing():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "fastvideo_kernel": {"native_extension": True},
            "flashinfer": {"package": True},
            "flashomni": {"package": False, "aot_config": False},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni", {"implementation": "upstream"}, "cuda", runtime,
    )

    assert any("flashomni implementation=upstream" in error for error in preflight["errors"])


def test_validate_rejects_draft_dense_switch():
    infer = _load_infer_module()

    with pytest.raises(NotImplementedError, match="block_sparse_attention=False"):
        infer.validate_method_config(
            "draft",
            {"block_sparse_attention": False},
        )


def test_preflight_requires_flashinfer_sparse_for_svoo():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "fastvideo_kernel": {"native_extension": True},
            "flashinfer": {"package": True, "sparse_module": False},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("flashinfer.sparse" in error for error in preflight["errors"])
    assert any("SparseVideo _kernels extension is not detected" in warning for warning in preflight["warnings"])


def test_strict_preflight_fails_svoo_missing_native_fused_kernel():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "fastvideo_kernel": {"native_extension": True},
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "flashinfer"},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("SparseVideo _kernels extension is not detected" in error for error in preflight["errors"])
    assert preflight["warnings"] == []


def test_strict_preflight_fails_sta_slow_fallback_boundary():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "fastvideo_kernel": {"native_extension": False},
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": None}, "cuda", runtime, strict_kernels=True,
    )

    assert any("fastvideo_kernel native extension is not detected" in error for error in preflight["errors"])
    assert any("seq_shape is not set" in error for error in preflight["errors"])
    assert preflight["warnings"] == []


def test_run_restores_sparse_attention_after_generation_failure(monkeypatch, tmp_path):
    infer = _load_infer_module()
    restored = {"value": False}

    class _Handle:
        def restore(self):
            restored["value"] = True

    class _Pipe:
        def __call__(self, **kwargs):
            raise RuntimeError("generation failed")

    import sparsevideo

    monkeypatch.setattr(infer, "load_pipeline", lambda *args, **kwargs: _Pipe())
    monkeypatch.setattr(infer, "prepare_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(sparsevideo, "apply_sparse_attention", lambda *args, **kwargs: _Handle())

    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan1.3b",
            "--method",
            "dense",
            "--device",
            "cpu",
            "--metrics-file",
            str(tmp_path / "metrics.jsonl"),
        ]
    )

    assert infer.run(args) == 1
    assert restored["value"] is True
    payload = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "generate"


@pytest.mark.skipif(
    os.environ.get("SPARSEVIDEO_RUN_REAL_PIPELINE_SMOKE") != "1",
    reason="set SPARSEVIDEO_RUN_REAL_PIPELINE_SMOKE=1 to run an actual local pipeline inference smoke",
)
def test_real_pipeline_inference_smoke(tmp_path):
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    model = env.get("SPARSEVIDEO_SMOKE_MODEL", "wan1.3b")
    method = env.get("SPARSEVIDEO_SMOKE_METHOD", "svoo")
    height = env.get("SPARSEVIDEO_SMOKE_HEIGHT", "720")
    width = env.get("SPARSEVIDEO_SMOKE_WIDTH", "1280")
    frames = env.get("SPARSEVIDEO_SMOKE_FRAMES", "81")
    steps = env.get("SPARSEVIDEO_SMOKE_STEPS", "2")
    output_file = tmp_path / "smoke.mp4"
    metrics_file = tmp_path / "metrics.jsonl"
    strict_args = ["--strict-kernels"] if env.get("SPARSEVIDEO_SMOKE_STRICT_KERNELS") == "1" else []

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            model,
            "--method",
            method,
            "--height",
            height,
            "--width",
            width,
            "--num-frames",
            frames,
            "--num-inference-steps",
            steps,
            "--local-files-only",
            "--output-file",
            str(output_file),
            "--metrics-file",
            str(metrics_file),
            *strict_args,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    assert payload["runtime"]["preflight"]["errors"] == []
    assert "apply_sparse_attention_sec" in payload["timings"]
    assert "generate_sec" in payload["timings"]
    assert payload["seconds_per_frame"] > 0
    assert metrics_file.exists()
    metrics_payload = json.loads(metrics_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert metrics_payload["status"] == "ok"
    assert metrics_payload["output_file"] == str(output_file)
