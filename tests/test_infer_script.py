from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import pytest
import torch


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
        [sys.executable, str(SCRIPT), *args, "--dry-run", "--print-json"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_infer_dry_run_unchecked(tmp_path: Path, *args: str) -> tuple[int, dict]:
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--dry-run", "--print-json"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, json.loads(result.stdout)


def _run_infer_dry_run_preflight_failure(tmp_path: Path, *args: str) -> dict:
    returncode, payload = _run_infer_dry_run_unchecked(tmp_path, *args)
    assert returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "preflight"
    assert payload["runtime"]["preflight"]["errors"]
    return payload


def _run_infer(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *args,
            "--print-json",
            "--metrics-file",
            str(tmp_path / "metrics.jsonl"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_print_run_summary_omits_full_config(capsys, tmp_path):
    infer = _load_infer_module()
    args = types.SimpleNamespace(metrics_file=tmp_path / "metrics.jsonl")
    infer.print_run_summary(
        args,
        {
            "status": "ok",
            "model": "wan21-t2v-1.3b",
            "method": "svg1",
            "method_config": {"num_sampled_rows": 64},
            "output_file": str(tmp_path / "sample.mp4"),
            "timings": {"generate_sec": 12.3456, "total_sec": 14.5678},
            "seconds_per_frame": 0.1524,
            "cuda_peak_allocated_gb": 1.25,
            "cuda_peak_reserved_gb": 2.5,
        },
    )

    stdout = capsys.readouterr().out
    assert "status=ok" in stdout
    assert f"output_file={tmp_path / 'sample.mp4'}" in stdout
    assert f"metrics_file={tmp_path / 'metrics.jsonl'}" in stdout
    assert "generate_sec=12.346" in stdout
    assert "method_config" not in stdout


def test_default_preflight_failure_stdout_is_concise(tmp_path):
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    metrics_file = tmp_path / "metrics.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wan1.3b",
            "--method",
            "svoo",
            "--device",
            "cpu",
            "--metrics-file",
            str(metrics_file),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "status=failed" in result.stdout
    assert "failed_stage=preflight" in result.stdout
    assert "Sparse methods require --device cuda" in result.stdout
    assert "method_config" not in result.stdout
    assert metrics_file.exists()
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout)


def _draft_mit_backend_ready(payload: dict) -> bool:
    mit = payload["runtime"]["optional_kernels"]["draft_kernels"]["mit_block_sparse_attn"]
    return bool(
        mit["source_files"]
        and mit["cuda_extension"]
        and (not mit.get("load_checked") or not mit.get("import_error"))
        and (not mit.get("load_checked") or mit.get("block_sparse_attn_func"))
        and (not mit.get("load_checked") or mit.get("cuda_fwd_block"))
    )


def _radial_runtime_ready() -> dict:
    return {
        "method_source": {"source_files": True},
        "flashinfer_bsr_wrapper": {"source_files": True},
        "owned_runtime": {
            "load_checked": True,
            "imported": True,
            "owned_runtime": True,
            "radial_bsr_mask": True,
            "shrink_mask_strict": True,
            "radial_flashinfer_attention": True,
            "radial_sage_attention": True,
            "radial_sage_dense_attention": True,
            "sparge_mask_convert": True,
            "sparge_sage_qk_block_sizes": True,
            "radial_append_tail_blocks": True,
            "expand_attention_mask": True,
            "radial_is_dense_layer_or_timestep": True,
            "radial_window_width": True,
            "build_bsr_from_mask": True,
            "variable_block_sparse_attn": True,
            "bsr_sparse_attn": True,
            "ensure_cuda_home_for_flashinfer_jit": True,
        },
    }


def _flashinfer_runtime_ready() -> dict:
    return {
        "package": True,
        "sparse_module": True,
        "cuda_toolkit": {"available": True},
        "load_checked": True,
        "imported": True,
        "sparse_imported": True,
        "top_level_block_sparse_attention_wrapper": True,
        "top_level_single_prefill_with_kv_cache": True,
        "top_level_merge_state": True,
        "sparse_variable_block_sparse_attention_wrapper": True,
        "sparse_canonicalize_torch_dtype": True,
        "sparse_mask_mode": True,
        "sparse_pos_encoding_mode": True,
        "sparse_determine_attention_backend": True,
        "sparse_get_batch_prefill_module": True,
    }


def _svg1_runtime_ready() -> dict:
    return {
        "triton_package": True,
        "method_source": {"source_files": True},
        "triton_placement": {"source_files": True},
        "owned_triton_runtime": {
            "load_checked": True,
            "imported": True,
            "owned_runtime": True,
            "svg_attention": True,
            "svg_flex_attention": True,
            "svg1_dense_attention": True,
            "svg1_hunyuan_flash_attn_varlen": True,
            "profile_masks": True,
            "svg_profile_mask_rows": True,
            "build_svg_block_mask": True,
            "svg_kv_blocks": True,
            "svg_kv_block_partitions": True,
            "svg_common_mask": True,
            "place_svg_heads": True,
            "restore_svg_heads": True,
            "round_svg_window_width": True,
            "svg_window_width": True,
            "sparsity_to_width": True,
            "resolve_prompt_length": True,
            "sparse_head_placement": True,
            "hidden_states_placement": True,
            "sparse_head_placement_kernel": True,
            "hidden_states_placement_kernel": True,
        },
    }


def test_infer_dry_run_resolves_wan_svoo_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "svoo")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["seed"] == 0
    assert payload["cpu_offload"] is False
    assert payload["cpu_offload_mode"] == "model"
    assert payload["vae_tiling"] is False
    assert payload["vae_slicing"] is False
    assert payload["vae_decoder_chunk_size"] is None
    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["fps"] == 16
    assert payload["wan_flow_shift"] == 5.0
    assert payload["vae_dtype"] == "fp32"
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
    assert payload["runtime"]["optional_kernels"]["svg_svoo_fused_kernels"]["native_load_checked"] is True
    assert "svoo_kernels" in payload["runtime"]["optional_kernels"]
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["triton_kmeans"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["triton_l2norm"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["triton_layernorm"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["triton_modulate"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["triton_permute"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["wan_fast_block_patch"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["hunyuan_sparse_forward_patch"]["source_files"] is True
    assert payload["runtime"]["optional_kernels"]["svoo_kernels"]["sparsity_profiler"]["source_files"] is True
    svoo_runtime = payload["runtime"]["optional_kernels"]["svoo_kernels"]["owned_triton_runtime"]
    assert svoo_runtime["load_checked"] is True
    assert svoo_runtime["imported"] is True
    assert svoo_runtime["owned_runtime"] is True
    assert svoo_runtime["co_cluster_tokens"] is True
    assert svoo_runtime["variable_block_sparse_attn"] is True


def test_infer_dry_run_resolves_skyreels_v2_sparse_alias(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "skyreels-v2", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "skyreels-v2-t2v-14b"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 97
    assert payload["fps"] == 24
    assert payload["wan_flow_shift"] == 5.0


def test_infer_dry_run_resolves_wan_animate_sparse_alias(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wananimate", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "wan22-animate-14b"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 77
    assert payload["fps"] == 16
    assert payload["wan_flow_shift"] == 5.0


def test_infer_dry_run_resolves_wan_vace_sparse_alias(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan-vace", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "wan21-vace-1.3b"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["method_config"]["sparsity_csv_path"] == "sparsity_profiles/sparsity_results.csv"
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model_key", "message"),
    [
        ("wan22-animate-14b", "WanAnimate real inference requires image, pose_video, and face_video inputs"),
        ("wan21-vace-1.3b", "WanVACE real inference requires video and mask inputs"),
    ],
)
def test_build_call_kwargs_rejects_auxiliary_wan_pipelines_without_cli_inputs(model_key, message):
    infer = _load_infer_module()
    args = types.SimpleNamespace(guidance_scale=None, num_inference_steps=None, skip_decode=False)

    with pytest.raises(RuntimeError, match=message):
        infer.build_call_kwargs(
            args,
            infer.MODEL_SPECS[model_key],
            prompt="test prompt",
            negative_prompt="",
            generator=None,
            num_frames=1,
            fps=16,
        )


def test_infer_dry_run_allows_cogvideox_svg2_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "cogvideox", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "cogvideox-t2v"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 49
    assert payload["fps"] == 8


@pytest.mark.parametrize(
    ("model", "resolved"),
    [
        ("cogvideox", "cogvideox-t2v"),
        ("cogvideox-i2v", "cogvideox-i2v"),
        ("ltx", "ltx-video"),
        ("ltx-i2v", "ltx-video-i2v"),
        ("allegro", "allegro"),
        ("mochi", "mochi-1"),
        ("easyanimate", "easyanimate-v5-t2v-12b"),
    ],
)
def test_infer_dry_run_allows_new_backbone_svg1_sparse_processors(tmp_path, model, resolved):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "svg1")

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "svg1"
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model", "resolved"),
    [
        ("cogvideox", "cogvideox-t2v"),
        ("cogvideox-i2v", "cogvideox-i2v"),
        ("ltx", "ltx-video"),
        ("ltx-i2v", "ltx-video-i2v"),
        ("allegro", "allegro"),
        ("mochi", "mochi-1"),
        ("easyanimate", "easyanimate-v5-t2v-12b"),
    ],
)
def test_infer_dry_run_allows_new_backbone_spargeattn_sparse_processors(tmp_path, model, resolved):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "spargeattn")

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "spargeattn"
    if model == "allegro":
        assert payload["method_config"]["topk"] == 0.5
        assert payload["method_config"]["dense_warmup_step_ratio"] == 0.1
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model", "resolved"),
    [
        ("cogvideox", "cogvideox-t2v"),
        ("cogvideox-i2v", "cogvideox-i2v"),
        ("ltx", "ltx-video"),
        ("ltx-i2v", "ltx-video-i2v"),
        ("allegro", "allegro"),
        ("mochi", "mochi-1"),
        ("easyanimate", "easyanimate-v5-t2v-12b"),
    ],
)
def test_infer_dry_run_allows_new_backbone_adacluster_sparse_processors(tmp_path, model, resolved):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "adacluster")

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "adacluster"
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model", "resolved"),
    [
        ("cogvideox", "cogvideox-t2v"),
        ("cogvideox-i2v", "cogvideox-i2v"),
        ("ltx", "ltx-video"),
        ("ltx-i2v", "ltx-video-i2v"),
        ("allegro", "allegro"),
        ("mochi", "mochi-1"),
        ("easyanimate", "easyanimate-v5-t2v-12b"),
    ],
)
def test_infer_dry_run_allows_new_backbone_flashomni_sparse_processors(tmp_path, model, resolved):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", model,
        "--method", "flashomni",
        "--method-config", "sparse_pattern=paper_mmdit",
        "--method-config", "max_order=0",
        "--method-config", "use_sparse_gemm=false",
    )

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "flashomni"
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model", "resolved"),
    [
        ("cogvideox", "cogvideox-t2v"),
        ("cogvideox-i2v", "cogvideox-i2v"),
        ("ltx", "ltx-video"),
        ("ltx-i2v", "ltx-video-i2v"),
        ("allegro", "allegro"),
        ("mochi", "mochi-1"),
        ("easyanimate", "easyanimate-v5-t2v-12b"),
    ],
)
def test_infer_dry_run_allows_new_backbone_sta_sparse_processors(tmp_path, model, resolved):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "sta")

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "sta"
    assert payload["runtime"]["preflight"]["errors"] == []


def test_sta_strategy_shapes_cover_sparsevideo_backbones():
    infer = _load_infer_module()
    from sparsevideo.methods.sta.search import MODEL_STRATEGY_SHAPES

    sta_families = (
        "wan",
        "hunyuan_video",
        "cogvideox",
        "ltx_video",
        "allegro",
        "mochi",
        "easyanimate",
    )
    expected = {
        key
        for key, spec in infer.MODEL_SPECS.items()
        if spec.family in sta_families
    }

    assert expected <= set(infer.STA_STRATEGY_SHAPES)
    assert infer.STA_STRATEGY_SHAPES == MODEL_STRATEGY_SHAPES
    assert infer.STA_UNSUPPORTED_STRATEGY_MODELS == {}

    strategy_root = REPO_ROOT / "src" / "sparsevideo" / "methods" / "sta" / "mask_strategies"
    missing = []
    for model_key in MODEL_STRATEGY_SHAPES:
        safe_key = "".join(ch if ch.isalnum() else "_" for ch in model_key.lower()).strip("_")
        path = strategy_root / f"mask_strategy_{safe_key}.json"
        if not path.exists():
            missing.append(path.name)
    assert missing == []


def test_inference_sh_has_runnable_sta_line_for_every_backbone():
    infer = _load_infer_module()
    script = REPO_ROOT / "scripts" / "inference.sh"
    sta_families = (
        "wan",
        "hunyuan_video",
        "cogvideox",
        "ltx_video",
        "allegro",
        "mochi",
        "easyanimate",
    )
    expected = {
        spec.key
        for spec in infer.MODEL_SPECS.values()
        if spec.family in sta_families
    }
    seen = set()
    missing_inputs = []

    for lineno, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
        if "--method sta" not in line or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line, comments=True, posix=True)
        while tokens and "=" in tokens[0] and not tokens[0].startswith("--"):
            tokens.pop(0)
        model = tokens[tokens.index("--model") + 1]
        seen.add(infer.MODEL_ALIASES[model])
        assert "/path/to/" not in line
        for flag in ("--image", "--pose-video", "--face-video", "--reference-video", "--mask-video"):
            if flag in tokens:
                value = tokens[tokens.index(flag) + 1]
                if not (REPO_ROOT / value).exists():
                    missing_inputs.append((lineno, flag, value))

    assert seen == expected
    assert missing_inputs == []


def test_inference_sh_has_runnable_inputs_for_every_command():
    infer = _load_infer_module()
    script = REPO_ROOT / "scripts" / "inference.sh"
    model_families = (
        "wan",
        "hunyuan_video",
        "cogvideox",
        "ltx_video",
        "allegro",
        "mochi",
        "easyanimate",
    )
    expected_models = {
        spec.key
        for spec in infer.MODEL_SPECS.values()
        if spec.family in model_families
    }
    expected_grid = {
        (model, method)
        for model in expected_models
        for method in infer.METHODS
    }
    seen_grid = set()
    missing_inputs = []

    for lineno, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "/path/to/" not in line
        tokens = shlex.split(line, comments=True, posix=True)
        while tokens and "=" in tokens[0] and not tokens[0].startswith("--"):
            tokens.pop(0)
        assert tokens[:2] == ["python", "scripts/infer.py"]
        model_arg = tokens[tokens.index("--model") + 1]
        method = tokens[tokens.index("--method") + 1]
        model = infer.MODEL_ALIASES[model_arg]
        spec = infer.MODEL_SPECS[model]
        seen_grid.add((model, method))

        for flag in ("--image", "--pose-video", "--face-video", "--reference-video", "--mask-video"):
            if flag in tokens:
                value = tokens[tokens.index(flag) + 1]
                if not (REPO_ROOT / value).exists():
                    missing_inputs.append((lineno, flag, value))

        if spec.pipeline_class in (
            "WanImageToVideoPipeline",
            "SkyReelsV2ImageToVideoPipeline",
            "HunyuanVideoImageToVideoPipeline",
            "CogVideoXImageToVideoPipeline",
            "LTXImageToVideoPipeline",
        ):
            assert "--image" in tokens
        if spec.pipeline_class == "WanAnimatePipeline":
            assert "--image" in tokens
            assert "--pose-video" in tokens
            assert "--face-video" in tokens
        if spec.pipeline_class == "WanVACEPipeline":
            assert "--reference-video" in tokens
            assert "--mask-video" in tokens

    assert seen_grid == expected_grid
    assert missing_inputs == []


def test_sta_new_backbone_rejects_wrong_strategy_shape(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(
        tmp_path,
        "--model", "cogvideox",
        "--method", "sta",
        "--method-config", "mask_strategy_file_path=src/sparsevideo/methods/sta/mask_strategies/mask_strategy_wan.json",
    )

    assert any(
        "expected cogvideox-t2v strategy shape (50, 42, 48)" in item
        for item in payload["runtime"]["preflight"]["errors"]
    )


def test_sta_wan_variant_uses_owned_tuned_mask(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan22", "--method", "sta")

    assert payload["status"] == "dry_run"
    assert payload["runtime"]["preflight"]["errors"] == []
    assert payload["method_config"]["mask_strategy_file_path"].endswith(
        "mask_strategy_wan22_t2v_a14b.json"
    )
    assert not any(
        "no tuned mask_strategy_file_path" in item
        for item in payload["runtime"]["preflight"]["warnings"]
    )


@pytest.mark.parametrize(
    ("model", "resolved", "shape_args"),
    [
        ("cogvideox", "cogvideox-t2v", ["--height", "256", "--width", "256", "--num-frames", "5"]),
        ("cogvideox-i2v", "cogvideox-i2v", ["--height", "480", "--width", "720", "--num-frames", "5"]),
        ("ltx", "ltx-video", ["--height", "512", "--width", "512", "--num-frames", "5"]),
        ("ltx-i2v", "ltx-video-i2v", ["--height", "512", "--width", "512", "--num-frames", "5"]),
        ("allegro", "allegro", ["--height", "256", "--width", "256", "--num-frames", "5"]),
        ("mochi", "mochi-1", ["--height", "512", "--width", "512", "--num-frames", "7"]),
        ("easyanimate", "easyanimate-v5-t2v-12b", ["--height", "256", "--width", "256", "--num-frames", "5"]),
    ],
)
def test_infer_dry_run_allows_new_backbone_draft_sparse_processors(tmp_path, model, resolved, shape_args):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "draft", *shape_args)

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "draft"
    assert payload["runtime"]["preflight"]["errors"] == []


@pytest.mark.parametrize(
    ("model", "resolved", "shape_args"),
    [
        ("cogvideox", "cogvideox-t2v", ["--height", "128", "--width", "128", "--num-frames", "5"]),
        (
            "cogvideox-i2v",
            "cogvideox-i2v",
            [
                "--height", "480",
                "--width", "720",
                "--num-frames", "125",
                "--method-config", "block_size=64",
            ],
        ),
        ("ltx", "ltx-video", ["--height", "512", "--width", "512", "--num-frames", "5"]),
        ("ltx-i2v", "ltx-video-i2v", ["--height", "512", "--width", "512", "--num-frames", "5"]),
        ("allegro", "allegro", ["--height", "128", "--width", "128", "--num-frames", "5"]),
        ("mochi", "mochi-1", ["--height", "256", "--width", "256", "--num-frames", "7"]),
        ("easyanimate", "easyanimate-v5-t2v-12b", ["--height", "128", "--width", "128", "--num-frames", "5"]),
    ],
)
def test_infer_dry_run_allows_new_backbone_radial_sparse_processors(tmp_path, model, resolved, shape_args):
    payload = _run_infer_dry_run(tmp_path, "--model", model, "--method", "radial", *shape_args)

    assert payload["status"] == "dry_run"
    assert payload["model"] == resolved
    assert payload["method"] == "radial"
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_cogvideox_svoo_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "cogvideox", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "cogvideox-t2v"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["kmeans_iter_init"] == 2
    assert payload["method_config"]["kmeans_iter_step"] == 2
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_cogvideox_i2v_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "cogvideox-i2v", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "cogvideox-i2v"
    assert payload["model_id"].endswith("CogVideoX-5b-I2V")
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_ltx_svg2_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "ltx", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "ltx-video"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 161
    assert payload["fps"] == 25


def test_infer_dry_run_allows_ltx_i2v_svoo_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "ltx-i2v", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "ltx-video-i2v"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["method_config"]["kmeans_iter_init"] == 2
    assert payload["runtime"]["preflight"]["errors"] == []


def test_ltx_single_file_checkpoint_prefers_base_checkpoint(tmp_path):
    infer = _load_infer_module()
    model_dir = tmp_path / "ltx-video"
    model_dir.mkdir()
    (model_dir / "ltxv-spatial-upscaler-0.9.8.safetensors").write_bytes(b"")
    preferred = model_dir / "ltx-video-2b-v0.9.5.safetensors"
    preferred.write_bytes(b"")

    assert infer._ltx_single_file_checkpoint(str(model_dir)) == preferred


def test_ltx_single_file_checkpoint_uses_component_layout_when_present(tmp_path):
    infer = _load_infer_module()
    model_dir = tmp_path / "ltx-video"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "ltx-video-2b-v0.9.5.safetensors").write_bytes(b"")

    assert infer._ltx_single_file_checkpoint(str(model_dir)) is None


def test_resolve_ltx_text_component_root_finds_compatible_sibling_t5(tmp_path):
    infer = _load_infer_module()
    model_dir = tmp_path / "ltx-video"
    text_encoder = model_dir / "text_encoder"
    text_encoder.mkdir(parents=True)
    text_encoder.joinpath("config.json").write_text(
        json.dumps({"model_type": "t5", "d_model": 4096, "num_layers": 24, "vocab_size": 32128}),
        encoding="utf-8",
    )

    sibling = tmp_path / "CogVideoX-5b"
    sibling_text_encoder = sibling / "text_encoder"
    sibling_tokenizer = sibling / "tokenizer"
    sibling_text_encoder.mkdir(parents=True)
    sibling_tokenizer.mkdir(parents=True)
    sibling_text_encoder.joinpath("config.json").write_text(
        json.dumps({"model_type": "t5", "d_model": 4096, "num_layers": 24, "vocab_size": 32128}),
        encoding="utf-8",
    )
    sibling_text_encoder.joinpath("model.safetensors").write_bytes(b"")
    sibling_tokenizer.joinpath("spiece.model").write_bytes(b"")

    assert infer._resolve_ltx_text_component_root(str(model_dir)) == sibling


def test_infer_dry_run_allows_allegro_svg2_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "allegro", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "allegro"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 88
    assert payload["fps"] == 15
    assert payload["method_config"]["kmeans_iter_init"] == 50
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_allegro_svoo_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "allegro", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "allegro"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["method_config"]["kmeans_iter_init"] == 2
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_mochi_svg2_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "mochi", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "mochi-1"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 19
    assert payload["fps"] == 8
    assert payload["method_config"]["kmeans_iter_init"] == 50
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_mochi_svoo_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "mochi", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "mochi-1"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["method_config"]["kmeans_iter_init"] == 2
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_easyanimate_svg2_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "easyanimate", "--method", "svg2")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "easyanimate-v5-t2v-12b"
    assert payload["method"] == "svg2"
    assert payload["num_frames"] == 49
    assert payload["fps"] == 8
    assert payload["method_config"]["kmeans_iter_init"] == 50
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_allows_easyanimate_svoo_sparse_processor(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "easyanimate", "--method", "svoo")

    assert payload["status"] == "dry_run"
    assert payload["model"] == "easyanimate-v5-t2v-12b"
    assert payload["method"] == "svoo"
    assert payload["method_config"]["use_dynamic_min_kc_ratio"] is False
    assert payload["method_config"]["kmeans_iter_init"] == 2
    assert payload["runtime"]["preflight"]["errors"] == []


def test_infer_dry_run_rejects_invalid_svoo_cocluster_iterations(tmp_path):
    code, payload = _run_infer_dry_run_unchecked(
        tmp_path,
        "--model",
        "cogvideox",
        "--method",
        "svoo",
        "--method-config",
        "kmeans_iter_init=0",
    )

    assert code == 1
    assert payload["failed_stage"] == "validate_method_config"
    assert "kmeans_iter_init > 0" in payload["error"]


def test_new_backbone_specs_enable_all_public_sparse_methods():
    infer = _load_infer_module()
    from sparsevideo._support import LIMITED_METHODS_BY_MODEL_TYPE

    all_method_models = [
        "cogvideox-t2v",
        "cogvideox-i2v",
        "ltx-video",
        "ltx-video-i2v",
        "allegro",
        "mochi-1",
        "easyanimate-v5-t2v-12b",
    ]

    assert LIMITED_METHODS_BY_MODEL_TYPE == {}
    for model_key in all_method_models:
        spec = infer.MODEL_SPECS[model_key]
        assert spec.sparse_supported is True
        assert spec.sparse_methods is None


def test_infer_dry_run_labels_sana_video_as_incompatible(tmp_path):
    code, payload = _run_infer_dry_run_unchecked(tmp_path, "--model", "sana-video", "--method", "svg2")

    assert code == 0
    assert payload["status"] == "unsupported_dry_run"
    assert payload["model"] == "sana-video"
    assert payload["compatibility_label"] == "incompatible"
    assert "SanaLinearAttnProcessor3_0 linear attention" in payload["unsupported_reason"]
    assert "compatibility_label=incompatible" in payload["error"]


def test_infer_dry_run_labels_kandinsky5_as_native_na(tmp_path):
    code, payload = _run_infer_dry_run_unchecked(tmp_path, "--model", "kandinsky5", "--method", "svoo")

    assert code == 0
    assert payload["status"] == "unsupported_dry_run"
    assert payload["model"] == "kandinsky5-t2v"
    assert payload["compatibility_label"] == "native-N/A"
    assert "native sparse attention controls" in payload["unsupported_reason"]
    assert "compatibility_label=native-N/A" in payload["error"]


@pytest.mark.parametrize(
    ("model", "expected_key", "reason"),
    [
        ("motif-video", "motif-video", "MotifVideo is not available"),
        ("ltx-video-2", "ltx-video-2", "LTX Video 2 is not available"),
    ],
)
def test_infer_dry_run_labels_unavailable_backbones_as_unknown(tmp_path, model, expected_key, reason):
    code, payload = _run_infer_dry_run_unchecked(tmp_path, "--model", model, "--method", "svg2")

    assert code == 0
    assert payload["status"] == "unsupported_dry_run"
    assert payload["model"] == expected_key
    assert payload["compatibility_label"] == "unknown"
    assert reason in payload["unsupported_reason"]
    assert "compatibility_label=unknown" in payload["error"]


def test_cogvideox_sparse_dry_run_does_not_preload_fused_native_kernels(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "cogvideox", "--method", "svg2")

    fused = payload["runtime"]["optional_kernels"]["svg_svoo_fused_kernels"]
    assert fused["native_load_checked"] is False


def test_ltx_sparse_dry_run_does_not_preload_fused_native_kernels(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "ltx", "--method", "svg2")

    fused = payload["runtime"]["optional_kernels"]["svg_svoo_fused_kernels"]
    assert fused["native_load_checked"] is False


def test_mochi_sparse_dry_run_does_not_preload_fused_native_kernels(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "mochi", "--method", "svg2")

    fused = payload["runtime"]["optional_kernels"]["svg_svoo_fused_kernels"]
    assert fused["native_load_checked"] is False


def test_wan_sparse_dry_run_keeps_fused_native_kernel_preload(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "svg2")

    fused = payload["runtime"]["optional_kernels"]["svg_svoo_fused_kernels"]
    assert fused["native_load_checked"] is True


def test_infer_dry_run_warns_for_wan13b_720p_quality_baseline(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "dense")

    assert payload["vae_tiling"] is False
    assert payload["vae_slicing"] is False
    assert any("Wan2.1 T2V 1.3B is a 480P model" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_infer_dry_run_keeps_wan13b_480p_quality_baseline_clean(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "dense",
        "--height", "480",
        "--width", "832",
    )

    assert not any("Wan2.1 T2V 1.3B is a 480P model" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_infer_dry_run_marks_skip_decode_as_latent_smoke(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "dense",
        "--skip-decode",
    )

    assert payload["status"] == "dry_run"
    assert payload["skip_decode"] is True
    assert payload["output_type"] == "latent"
    assert payload["output_file"] is None


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
    assert cfg["allow_triton_fallback"] is False
    assert "svg2_kernels" in payload["runtime"]["optional_kernels"]
    svg2_runtime = payload["runtime"]["optional_kernels"]["svg2_kernels"]["owned_triton_runtime"]
    assert svg2_runtime["load_checked"] is True
    assert svg2_runtime["imported"] is True
    assert svg2_runtime["owned_runtime"] is True
    assert svg2_runtime["triton_kmeans"] is True
    assert svg2_runtime["variable_block_sparse_attn"] is True


def test_infer_dry_run_resolves_hunyuan_svg2_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "hunyuan", "--method", "svg2")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["num_frames"] == 129
    assert cfg["dense_warmup_step_ratio"] == 0.1
    assert cfg["dense_warmup_layer_ratio"] == 0.03
    assert "first_times_fp" not in cfg
    assert "first_layers_fp" not in cfg
    assert cfg["num_q_centroids"] == 400
    assert cfg["num_k_centroids"] == 1000
    assert cfg["zero_step_kmeans_init"] is True
    assert cfg["context_length"] == 256
    assert cfg["prompt_length"] is None


def test_svg2_upstream_profile_uses_sparse_videogen_wan14_shell(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "svg2",
        "--profile", "upstream",
    )

    assert payload["profile"] == "upstream"
    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["num_inference_steps"] == 50
    assert payload["profile_overrides"]["source"].endswith(
        "Sparse-VideoGen/scripts/wan/wan_t2v_720p_sap.sh"
    )


def test_svg2_upstream_profile_does_not_borrow_wan14_shell_for_wan13(tmp_path):
    result = _run_infer(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "svg2",
        "--profile", "upstream",
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["failed_stage"] == "profile"
    assert "No upstream inference profile" in payload["error"]
    assert "method='svg2'" in payload["error"]
    assert "model='wan21-t2v-1.3b'" in payload["error"]


def test_infer_dry_run_resolves_wan_svg1_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "svg1")
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert payload["wan_flow_shift"] == 5.0
    assert cfg["dense_warmup_step_ratio"] == 0.1
    assert cfg["dense_warmup_layer_ratio"] == 0.03
    assert "first_times_fp" not in cfg
    assert "first_layers_fp" not in cfg
    assert cfg["num_sampled_rows"] == 64
    assert cfg["sparsity"] == 0.3
    svg1_runtime = payload["runtime"]["optional_kernels"]["svg1_kernels"]["owned_triton_runtime"]
    assert svg1_runtime["load_checked"] is True
    assert svg1_runtime["imported"] is True
    assert svg1_runtime["owned_runtime"] is True
    assert svg1_runtime["svg_attention"] is True
    assert svg1_runtime["sparse_head_placement"] is True


def test_svg1_hunyuan_upstream_profile_uses_sparse_videogen_shell(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "svg1",
        "--profile", "upstream",
    )

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 129
    assert payload["fps"] == 24
    assert payload["method_config"]["context_length"] == 256
    assert payload["method_config"]["prompt_length"] is None
    assert payload["scheduler_flow_shift"] == 7.0
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["negative_prompt"].startswith("Aerial view, aerial view")
    assert payload["profile_overrides"]["flow_shift"] == 7.0
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["negative_prompt"].startswith("Aerial view, aerial view")
    svg1_runtime = payload["runtime"]["optional_kernels"]["svg1_kernels"]["owned_triton_runtime"]
    assert svg1_runtime["load_checked"] is True
    assert svg1_runtime["svg1_hunyuan_flash_attn_varlen"] is True
    assert payload["profile_overrides"]["source"].endswith(
        "Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_svg.sh"
    )


def test_svg2_hunyuan_upstream_profile_uses_sparse_videogen_shell(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "svg2",
        "--profile", "upstream",
    )

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 129
    assert payload["fps"] == 24
    assert payload["scheduler_flow_shift"] == 7.0
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["negative_prompt"].startswith("Aerial view, aerial view")
    assert payload["profile_overrides"]["flow_shift"] == 7.0
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["negative_prompt"].startswith("Aerial view, aerial view")
    assert payload["profile_overrides"]["source"].endswith(
        "Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_sap.sh"
    )


def test_infer_dry_run_resolves_wan_adacluster_fixed_cluster_defaults(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "adacluster")
    cfg = payload["method_config"]
    adacluster_runtime = payload["runtime"]["optional_kernels"]["adacluster_kernels"]["owned_triton_runtime"]

    assert payload["status"] == "dry_run"
    assert cfg["topk_num"] == 128
    assert cfg["q_kernel_num"] == 100
    assert cfg["kv_kernel_num"] == 500
    assert cfg["use_thresholded_kmeans_loop"] is False
    assert cfg["initial_q_kernel_num"] == 50
    assert cfg["initial_kv_kernel_num"] == 200
    assert cfg["q_distance_threshold"] == 9.0
    assert cfg["kv_distance_threshold"] == 5.5
    assert cfg["thresholded_kmeans_iter_time"] == 3
    assert cfg["thresholded_kmeans_max_iterations"] == 10
    assert "adacluster_kernels" in payload["runtime"]["optional_kernels"]
    assert adacluster_runtime["load_checked"] is True
    assert adacluster_runtime["imported"] is True
    assert adacluster_runtime["owned_runtime"] is True
    assert adacluster_runtime["flash_kmeans_single"] is True
    assert adacluster_runtime["triton_cluster_sparse_attn"] is True
    assert adacluster_runtime["triton_cluster_sparse_attn_topk"] is True


def test_adacluster_upstream_profile_uses_wan_runwan_reference(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "adacluster",
        "--profile", "upstream",
    )

    assert payload["profile"] == "upstream"
    assert payload["height"] == 480
    assert payload["width"] == 832
    assert payload["num_frames"] == 81
    assert payload["num_inference_steps"] == 50
    assert payload["fps"] == 16
    assert payload["wan_flow_shift"] == 5.0
    assert payload["runtime"]["preflight"]["errors"] == []
    assert payload["negative_prompt"].startswith("色调艳丽")
    assert payload["profile_overrides"]["negative_prompt"].startswith("色调艳丽")
    assert payload["profile_overrides"]["source"].endswith("Adacluster/runwan/runwan.py")


def test_adacluster_upstream_profile_uses_hunyuan_reference(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "adacluster",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["profile"] == "upstream"
    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["num_inference_steps"] == 30
    assert payload["fps"] == 15
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "model"
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["runtime"]["preflight"]["errors"] == []
    assert cfg["topk_num"] == 94
    assert cfg["q_kernel_num"] == 250
    assert cfg["kv_kernel_num"] == 1243
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["source"].endswith("Adacluster/runhunyuan/run_hunyuan.py")


def test_infer_dry_run_resolves_draft_defaults_for_default_target_shape(tmp_path):
    returncode, payload = _run_infer_dry_run_unchecked(tmp_path, "--model", "wan1.3b", "--method", "draft")
    cfg = payload["method_config"]

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert cfg["pool_h"] == 8
    assert cfg["pool_w"] == 16
    assert cfg["sparsity_ratio"] == 0.75
    assert cfg["latent_h"] == 45
    assert cfg["latent_w"] == 80
    assert cfg["visual_len"] == 75_600
    errors = payload["runtime"]["preflight"]["errors"]
    assert not any("draft upstream sparse path requires latent_h" in item for item in errors)
    if _draft_mit_backend_ready(payload):
        assert returncode == 0
        assert errors == []
    else:
        assert returncode == 1
        assert any("MIT Han Lab Block-Sparse-Attention" in item for item in errors)


def test_infer_dry_run_resolves_radial_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(tmp_path, "--model", "wan1.3b", "--method", "radial")
    cfg = payload["method_config"]

    assert payload["strict_kernels"] is True
    assert cfg["dense_layers"] == 1
    assert cfg["dense_timesteps"] == 12
    assert cfg["decay_factor"] == 0.2
    assert cfg["block_size"] == 128
    assert cfg["allow_flex_fallback"] is False
    assert any("allow_flex_fallback" in item for item in payload["runtime"]["preflight"]["errors"])


def test_infer_dry_run_resolves_wan22_radial_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(tmp_path, "--model", "wan22", "--method", "radial")
    cfg = payload["method_config"]

    assert cfg["dense_layers"] == 1
    assert cfg["dense_timesteps"] == 11
    assert cfg["decay_factor"] == 0.8
    assert cfg["block_size"] == 64


def test_infer_dry_run_allows_radial_debug_fallback_when_explicit(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path, "--model", "wan1.3b", "--method", "radial", "--allow-debug-fallbacks",
    )

    assert payload["strict_kernels"] is False
    assert payload["allow_debug_fallbacks"] is True
    assert payload["method_config"]["allow_flex_fallback"] is True
    assert any("allow_flex_fallback" in item for item in payload["runtime"]["preflight"]["warnings"])
    assert not any("allow_flex_fallback" in item for item in payload["runtime"]["preflight"]["errors"])


def test_infer_dry_run_resolves_hunyuan_radial_upstream_defaults(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(tmp_path, "--model", "hunyuan", "--method", "radial")
    cfg = payload["method_config"]

    assert cfg["dense_layers"] == 0
    assert cfg["dense_timesteps"] == 12
    assert cfg["decay_factor"] == 0.95
    assert cfg["block_size"] == 128
    assert any("allow_flex_fallback" in item for item in payload["runtime"]["preflight"]["errors"])


def test_radial_upstream_shape_avoids_flex_fallback_warning(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "radial",
        "--height", "768",
        "--width", "1280",
        "--num-frames", "69",
    )

    radial_runtime = payload["runtime"]["optional_kernels"]["radial_kernels"]["owned_runtime"]
    assert radial_runtime["load_checked"] is True
    assert radial_runtime["imported"] is True
    assert radial_runtime["owned_runtime"] is True
    assert radial_runtime["radial_bsr_mask"] is True
    assert radial_runtime["bsr_sparse_attn"] is True
    assert not any("FlexAttention fallback" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("FlexAttention fallback" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_radial_upstream_profile_resolves_benchmark_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "radial",
        "--profile", "upstream",
    )

    assert payload["profile"] == "upstream"
    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 69
    assert payload["profile_overrides"]["source"].endswith("radial-attention/scripts/wan_t2v_inference.sh")
    radial_runtime = payload["runtime"]["optional_kernels"]["radial_kernels"]["owned_runtime"]
    assert radial_runtime["load_checked"] is True
    assert radial_runtime["imported"] is True
    assert radial_runtime["build_bsr_from_mask"] is True
    assert not any("FlexAttention fallback" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("FlexAttention fallback" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_radial_wan22_upstream_profile_uses_reference_vae_policy(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan22",
        "--method", "radial",
        "--profile", "upstream",
    )

    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 77
    assert payload["num_inference_steps"] == 40
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["source"].endswith("radial-attention/scripts/wan_22_t2v_inference.sh")


def test_radial_hunyuan_upstream_profile_uses_reference_vae_policy(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "radial",
        "--profile", "upstream",
    )

    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 117
    assert payload["num_inference_steps"] == 50
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["source"].endswith("radial-attention/scripts/hunyuan_t2v_inference.sh")


def test_radial_upstream_profile_is_not_claimed_for_wan13b(tmp_path):
    code, payload = _run_infer_dry_run_unchecked(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "radial",
        "--profile", "upstream",
    )

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "profile"
    assert any("No upstream inference profile" in item for item in payload["runtime"]["preflight"]["errors"])


def test_resolve_inference_profile_rejects_missing_upstream_profile():
    infer = _load_infer_module()

    with pytest.raises(ValueError, match="No upstream inference profile"):
        infer.resolve_inference_profile(
            "upstream",
            infer.MODEL_SPECS["wan21-t2v-1.3b"],
            "sta",
        )


def test_missing_upstream_profile_fails_before_model_load(tmp_path):
    result = _run_infer(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "sta",
        "--profile", "upstream",
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "profile"
    assert "No upstream inference profile" in payload["error"]
    assert "method='sta'" in payload["error"]
    assert "model='wan21-t2v-1.3b'" in payload["error"]
    assert payload["profile_overrides"] == {}
    assert payload["runtime"]["preflight"]["errors"] == [payload["error"]]
    assert payload["timings"] == {}


def test_dense_can_use_sparse_method_upstream_profile_for_fair_baseline(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "dense",
        "--profile", "upstream",
        "--profile-for-method", "draft",
    )

    assert payload["profile"] == "upstream"
    assert payload["profile_method"] == "draft"
    assert payload["method"] == "dense"
    assert payload["method_config"] == {}
    assert payload["height"] == 512
    assert payload["width"] == 768
    assert payload["num_frames"] == 81
    assert payload["seed"] == 42
    assert payload["wan_flow_shift"] == 5.0
    assert payload["runtime"]["preflight"]["errors"] == []
    assert payload["profile_overrides"]["source"].endswith(
        "draft-attention/wan/run-single-inference.sh"
    )


def test_profile_for_method_does_not_apply_sparse_method_config_to_dense(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "dense",
        "--profile", "upstream",
        "--profile-for-method", "sta",
    )

    assert payload["profile_method"] == "sta"
    assert payload["method"] == "dense"
    assert payload["method_config"] == {}
    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 69
    assert payload["profile_overrides"]["method_config"] == {"seq_shape": "18x48x80"}


def test_all_upstream_profiles_resolve_for_dense_baselines_without_sparse_config():
    infer = _load_infer_module()
    import sparsevideo

    expected_profiles = {
        ("adacluster", "hunyuan_video"),
        ("adacluster", "wan21-t2v-1.3b"),
        ("draft", "hunyuan_video"),
        ("draft", "wan21-t2v-14b"),
        ("radial", "hunyuan_video"),
        ("radial", "wan21-t2v-14b"),
        ("radial", "wan22-t2v-a14b"),
        ("spargeattn", "hunyuan_video"),
        ("spargeattn", "wan21-t2v-1.3b"),
        ("spargeattn", "wan21-t2v-14b"),
        ("spargeattn", "wan22-t2v-a14b"),
        ("sta", "hunyuan_video"),
        ("sta", "wan21-t2v-14b"),
        ("svg1", "hunyuan_video"),
        ("svg1", "wan21-t2v-14b"),
        ("svg2", "hunyuan_video"),
        ("svg2", "wan21-t2v-14b"),
        ("svoo", "hunyuan_video"),
        ("svoo", "wan21-t2v-1.3b"),
        ("svoo", "wan21-t2v-14b"),
        ("svoo", "wan22-t2v-a14b"),
    }
    source_snippets = {
        ("adacluster", "hunyuan_video"): [
            "height=720, width=1280",
            "num_frames=81, num_inference_steps=30",
            "export_to_video(output, save_path, fps=15)",
        ],
        ("adacluster", "wan21-t2v-1.3b"): [
            'task = "t2v-1.3B"',
            'size = "832*480"',
            '"--base_seed", "0"',
            "args.sample_steps = 50",
            'args.frame_num = 1 if "t2i" in args.task else 81',
            "wan_shared_cfg.sample_fps = 16",
        ],
        ("draft", "hunyuan_video"): [
            "seed 42 in 768p",
            "pool_h=8",
            "visual_len=126_720",
            "sparsity_ratio=0.9",
        ],
        ("draft", "wan21-t2v-14b"): [
            "--size 768*512",
            "--task t2v-14B",
            'args.sample_steps = 40 if "i2v" in args.task else 50',
            'args.frame_num = 1 if "t2i" in args.task else 81',
            "wan_shared_cfg.sample_fps = 16",
        ],
        ("radial", "hunyuan_video"): [
            "dense_layers=0",
            "dense_timesteps=12",
            "--num_frames 117",
            "--decay_factor 0.95",
        ],
        ("radial", "wan21-t2v-14b"): [
            "dense_layers=1",
            "dense_timesteps=12",
            "--num_frames 69",
            "--decay_factor 0.2",
        ],
        ("radial", "wan22-t2v-a14b"): [
            "dense_timesteps=11",
            "decay_factor=0.8",
            "--num_frames 77",
            "--guidance_scale 4.0",
            "--guidance_scale_2 3.0",
        ],
        ("spargeattn", "hunyuan_video"): [
            'default="full"',
            "height=320",
            "width=512",
            "num_frames=61",
            "manual_seed(42)",
        ],
        ("spargeattn", "wan21-t2v-1.3b"): [
            'default="full"',
            "height=480",
            "width=832",
            "num_frames=81",
            "guidance_scale=5.0",
        ],
        ("spargeattn", "wan21-t2v-14b"): [
            'default="full"',
            "height=480",
            "width=832",
            "num_frames=81",
            "guidance_scale=5.0",
        ],
        ("spargeattn", "wan22-t2v-a14b"): [
            "--model wan2_2-14b",
            "--mode topk --value 0.4",
            "Memory helpers enabled",
            "height=720",
            "width=1280",
            "num_frames=81",
            "num_inference_steps=40",
            "guidance_scale=4.0",
            "guidance_scale_2=3.0",
        ],
        ("sta", "hunyuan_video"): [
            "# Sliding Tile Attention (STA)",
            "sta_do_not_delete",
            "STA_inference",
        ],
        ("sta", "wan21-t2v-14b"): [
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "--height 768",
            "--width 1280",
            "--num-frames 69",
            "--num-inference-steps 50",
        ],
        ("svg1", "hunyuan_video"): [
            "infer_step=50",
            "first_times_fp=0.1",
            "sparsity=0.25",
            "--height 720",
            "--width 1280",
        ],
        ("svg1", "wan21-t2v-14b"): [
            "infer_step=50",
            "first_times_fp=0.2",
            "sparsity=0.3",
            "--height 720",
            "--width 1280",
        ],
        ("svg2", "hunyuan_video"): [
            "qc_kmeans=400",
            "kc_kmeans=1000",
            "--zero_step_kmeans_init",
            "--height 720",
            "--width 1280",
        ],
        ("svg2", "wan21-t2v-14b"): [
            "qc_kmeans=300",
            "kc_kmeans=1000",
            "kmeans_iter_init=50",
            "--height 720",
            "--width 1280",
        ],
        ("svoo", "hunyuan_video"): [
            "num_inference_steps=50",
            "first_times_fp=0.1",
            "top_p_kmeans=0.88",
            "num_frames=\"${NUM_FRAMES:-129}\"",
            "torch_dtype\": torch.bfloat16",
            "flow_shift = 7.0",
            "pipe.vae.enable_tiling()",
        ],
        ("svoo", "wan21-t2v-1.3b"): [
            "model_size=\"${MODEL_SIZE:-1.3B}\"",
            "num_inference_steps=50",
            "first_times_fp=0.2",
            "start_reuse_step=11",
            "WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)",
            "export_to_video(output, args.output_file, fps=16)",
        ],
        ("svoo", "wan21-t2v-14b"): [
            "14B|14b|wan21_14b|wan21_t2v_14b)",
            "num_inference_steps=50",
            "start_reuse_step=11",
            "sparsity_profiles/sparsity_wan_14B_t2v.csv",
            "WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)",
        ],
        ("svoo", "wan22-t2v-a14b"): [
            "A14B|a14b|wan22|wan22_t2v_a14b)",
            "num_inference_steps=40",
            "start_reuse_step=9",
            "sparsity_profiles/sparsity_wan22_A14B_t2v.csv",
            "WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)",
            "guidance_scale_2=3.0",
        ],
    }
    assert set(infer.UPSTREAM_INFERENCE_PROFILES) == expected_profiles
    assert set(source_snippets) == expected_profiles

    for profile_method, selector in sorted(expected_profiles):
        model_arg = "hunyuan" if selector == "hunyuan_video" else selector
        args = infer.build_parser().parse_args(
            [
                "--model",
                model_arg,
                "--method",
                "dense",
                "--profile",
                "upstream",
                "--profile-for-method",
                profile_method,
                "--dry-run",
            ]
        )
        spec = infer.MODEL_SPECS[infer.MODEL_ALIASES[args.model]]
        fps = args.fps if args.fps is not None else spec.fps
        num_frames = spec.default_frames
        steps = spec.default_steps

        profile = infer.resolve_inference_profile(args.profile, spec, args.profile_for_method)
        height, width, fps, num_frames, steps = infer.apply_profile_runtime_defaults(
            args, profile, fps, num_frames, steps,
        )

        profile_sources = [profile["source"], *profile.get("evidence_sources", [])]
        assert len(profile_sources) == len(set(profile_sources))
        for source in profile_sources:
            assert source.startswith("training_free/")
            assert (REPO_ROOT / source).exists()
        source_text = "\n".join(
            (REPO_ROOT / source).read_text(encoding="utf-8")
            for source in profile_sources
        )
        for snippet in source_snippets[(profile_method, selector)]:
            assert snippet in source_text, (profile_method, selector, snippet, profile["source"])
        assert height == profile["height"]
        assert width == profile["width"]
        assert num_frames == profile["num_frames"]
        assert steps == profile.get("num_inference_steps", spec.default_steps)
        assert fps == profile.get("fps", spec.fps)
        assert sparsevideo.default_method_config(
            "dense",
            num_inference_steps=steps,
            model_family=spec.family,
            model_key=spec.key,
        ) == {}


def test_draft_layout_preflight_accepts_upstream_wan_shape():
    infer = _load_infer_module()
    spec = infer.MODEL_SPECS["wan21-t2v-14b"]

    error = infer.draft_upstream_layout_error(
        spec,
        height=768,
        width=1280,
        num_frames=81,
        config={"pool_h": 8, "pool_w": 16, "block_sparse_attention": True},
    )

    assert error is None


def test_draft_upstream_profile_uses_wan_shell_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "draft",
        "--upstream-profile",
    )

    assert payload["profile"] == "upstream"
    assert payload["height"] == 512
    assert payload["width"] == 768
    assert payload["num_frames"] == 81
    assert payload["seed"] == 42
    assert payload["method_config"]["latent_h"] == 32
    assert payload["method_config"]["latent_w"] == 48
    assert payload["method_config"]["visual_len"] == 32_256
    assert payload["method_config"]["text_len"] == 0
    assert payload["method_config"]["batch_size"] == 1
    if _draft_mit_backend_ready(payload):
        assert not any("MIT Han Lab Block-Sparse-Attention" in item for item in payload["runtime"]["preflight"]["errors"])
    else:
        assert any("MIT Han Lab Block-Sparse-Attention" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("draft upstream sparse path requires latent_h" in item for item in payload["runtime"]["preflight"]["errors"])
    assert payload["profile_overrides"]["source"].endswith("draft-attention/wan/run-single-inference.sh")


def test_upstream_profile_seed_keeps_user_override(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "draft",
        "--profile", "upstream",
        "--seed", "0",
    )

    assert payload["seed"] == 0
    assert payload["profile_overrides"]["seed"] == 42


def test_draft_upstream_profile_is_not_claimed_for_wan13b(tmp_path):
    code, payload = _run_infer_dry_run_unchecked(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "draft",
        "--profile", "upstream",
    )

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "profile"
    assert any("No upstream inference profile" in item for item in payload["runtime"]["preflight"]["errors"])


def test_draft_debug_fallback_sets_local_triton_switch(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "draft",
        "--profile", "upstream",
        "--allow-debug-fallbacks",
    )

    assert payload["strict_kernels"] is False
    assert payload["method_config"]["allow_triton_fallback"] is True
    assert payload["runtime"]["preflight"]["errors"] == []
    if _draft_mit_backend_ready(payload):
        assert not any("debug fallback only" in item for item in payload["runtime"]["preflight"]["warnings"])
    else:
        assert any("debug fallback only" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_infer_dry_run_resolves_hunyuan_draft_defaults_for_default_target_shape(tmp_path):
    returncode, payload = _run_infer_dry_run_unchecked(tmp_path, "--model", "hunyuan", "--method", "draft")
    cfg = payload["method_config"]

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 129
    assert cfg["pool_h"] == 8
    assert cfg["pool_w"] == 16
    assert cfg["sparsity_ratio"] == 0.9
    assert cfg["latent_h"] == 45
    assert cfg["latent_w"] == 80
    assert cfg["visual_len"] == 118_800
    errors = payload["runtime"]["preflight"]["errors"]
    assert not any("draft upstream latent_h config expects 48" in item for item in errors)
    if _draft_mit_backend_ready(payload):
        assert returncode == 0
        assert errors == []
    else:
        assert returncode == 1
        assert any("MIT Han Lab Block-Sparse-Attention" in item for item in errors)


def test_draft_layout_preflight_accepts_upstream_hunyuan_diffusers_shape():
    infer = _load_infer_module()
    spec = infer.MODEL_SPECS["hunyuan-t2v"]

    error = infer.draft_upstream_layout_error(
        spec,
        height=768,
        width=1280,
        num_frames=129,
        config={"pool_h": 8, "pool_w": 16, "block_sparse_attention": True},
    )

    assert error is None


def test_hunyuan_draft_upstream_profile_uses_768p_demo_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "draft",
        "--profile", "upstream",
    )

    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 129
    assert payload["seed"] == 42
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "sequential"
    assert payload["method_config"]["latent_h"] == 48
    assert payload["method_config"]["latent_w"] == 80
    assert payload["method_config"]["visual_len"] == 126_720
    assert payload["method_config"]["text_len"] == 256
    if _draft_mit_backend_ready(payload):
        assert not any("MIT Han Lab Block-Sparse-Attention" in item for item in payload["runtime"]["preflight"]["errors"])
    else:
        assert any("MIT Han Lab Block-Sparse-Attention" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("draft upstream sparse path requires latent_h" in item for item in payload["runtime"]["preflight"]["errors"])


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


def test_svoo_wan13_upstream_profile_uses_reference_shell(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "svoo",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["num_inference_steps"] == 50
    assert payload["scheduler_flow_shift"] == 3.0
    assert payload["profile_overrides"]["flow_shift"] == 3.0
    assert payload["vae_tiling"] is False
    assert payload["vae_slicing"] is False
    assert payload["profile_overrides"]["vae_tiling"] is False
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert cfg["num_q_centroids"] == 256
    assert cfg["num_k_centroids"] == 1024
    assert cfg["start_reuse_step"] == 11
    assert payload["profile_overrides"]["source"].endswith(
        "SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh"
    )


def test_svoo_wan14_upstream_profile_keeps_reference_scheduler(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "svoo",
        "--profile", "upstream",
    )

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["num_inference_steps"] == 50
    assert payload["scheduler_flow_shift"] == 3.0
    assert payload["vae_dtype"] == "bf16"
    assert payload["profile_overrides"]["flow_shift"] == 3.0
    assert payload["profile_overrides"]["vae_dtype"] == "bf16"
    assert payload["profile_overrides"]["source"].endswith(
        "SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh"
    )


def test_dense_baseline_for_svoo_wan14_keeps_reference_scheduler(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "dense",
        "--profile", "upstream",
        "--profile-for-method", "svoo",
    )

    assert payload["method"] == "dense"
    assert payload["profile_method"] == "svoo"
    assert payload["scheduler_flow_shift"] == 3.0
    assert payload["vae_dtype"] == "bf16"
    assert payload["profile_overrides"]["flow_shift"] == 3.0
    assert payload["profile_overrides"]["vae_dtype"] == "bf16"


def test_svoo_wan22_upstream_profile_uses_a14b_steps(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan22",
        "--method", "svoo",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["num_inference_steps"] == 40
    assert payload["scheduler_flow_shift"] == 3.0
    assert payload["vae_dtype"] == "bf16"
    assert payload["profile_overrides"]["flow_shift"] == 3.0
    assert payload["profile_overrides"]["vae_dtype"] == "bf16"
    assert payload["profile_overrides"]["guidance_scale"] == 5.0
    assert payload["profile_overrides"]["guidance_scale_2"] == 3.0
    assert payload["vae_tiling"] is False
    assert payload["vae_slicing"] is False
    assert cfg["start_reuse_step"] == 9
    assert cfg["sparsity_csv_path"].endswith("sparsity_wan22_A14B_t2v.csv")


def test_svoo_hunyuan_upstream_profile_uses_reference_shell(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "svoo",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 129
    assert payload["fps"] == 24
    assert payload["seed"] == 23
    assert payload["scheduler_flow_shift"] == 7.0
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is False
    assert payload["profile_overrides"]["seed"] == 23
    assert payload["profile_overrides"]["flow_shift"] == 7.0
    assert payload["profile_overrides"]["vae_tiling"] is True
    assert payload["profile_overrides"]["vae_slicing"] is False
    assert payload["profile_overrides"]["negative_prompt"].startswith("Aerial view, aerial view")
    assert payload["negative_prompt"].startswith("Aerial view, aerial view")
    assert cfg["top_p_kmeans"] == 0.88
    assert cfg["start_reuse_step"] == 6
    assert cfg["reuse_interval"] == 50
    assert payload["profile_overrides"]["source"].endswith(
        "SVOO/scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh"
    )


def test_hunyuan_load_pipeline_uses_profile_flow_shift(monkeypatch):
    module = _load_infer_module()
    captured = {}

    class FakeScheduler:
        def __init__(self, shift):
            self.shift = shift
            captured["shift"] = shift

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            FlowMatchEulerDiscreteScheduler=FakeScheduler,
            HunyuanVideoPipeline=FakePipeline,
        ),
    )

    module.load_pipeline(
        module.MODEL_SPECS["hunyuan-t2v"],
        "local-hunyuan",
        torch.bfloat16,
        None,
        local_files_only=True,
        height=720,
        flow_shift=7.0,
    )

    assert captured["model_id"] == "local-hunyuan"
    assert captured["shift"] == 7.0
    assert captured["kwargs"]["scheduler"].shift == 7.0


def test_wan_load_pipeline_uses_resolved_vae_dtype(monkeypatch):
    module = _load_infer_module()
    captured = {}

    class FakeVae:
        pass

    class FakeAutoencoderKLWan:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["vae_model_id"] = model_id
            captured["vae_kwargs"] = kwargs
            return FakeVae()

    class FakeWanPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["pipe_model_id"] = model_id
            captured["pipe_kwargs"] = kwargs
            instance = cls()
            instance.scheduler = types.SimpleNamespace(config={"flow_shift": 3.0})
            return instance

    class FakeUniPCMultistepScheduler:
        @classmethod
        def from_config(cls, config, **kwargs):
            captured["scheduler_config"] = config
            captured["scheduler_kwargs"] = kwargs
            return types.SimpleNamespace(config={**config, **kwargs})

    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.__path__ = []
    diffusers_module.AutoencoderKLWan = FakeAutoencoderKLWan
    diffusers_module.WanPipeline = FakeWanPipeline
    schedulers_module = types.ModuleType("diffusers.schedulers")
    schedulers_module.__path__ = []
    scheduler_module = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    scheduler_module.UniPCMultistepScheduler = FakeUniPCMultistepScheduler

    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)
    monkeypatch.setitem(sys.modules, "diffusers.schedulers", schedulers_module)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.schedulers.scheduling_unipc_multistep",
        scheduler_module,
    )

    pipe = module.load_pipeline(
        module.MODEL_SPECS["wan21-t2v-14b"],
        "local-wan",
        torch.bfloat16,
        torch.bfloat16,
        local_files_only=True,
        height=720,
        flow_shift=3.0,
    )

    assert pipe.scheduler.config["flow_shift"] == 3.0
    assert captured["vae_model_id"] == "local-wan"
    assert captured["vae_kwargs"]["torch_dtype"] is torch.bfloat16
    assert captured["pipe_kwargs"]["torch_dtype"] is torch.bfloat16
    assert captured["pipe_kwargs"]["vae"].__class__ is FakeVae
    assert captured["scheduler_kwargs"]["flow_shift"] == 3.0


def test_prepare_pipeline_respects_separate_vae_tiling_and_slicing():
    module = _load_infer_module()
    calls = []

    class FakeVAE:
        def enable_tiling(self):
            calls.append("tiling")

        def enable_slicing(self):
            calls.append("slicing")

    class FakePipe:
        vae = FakeVAE()

        def to(self, device):
            calls.append(f"to:{device}")

    module.prepare_pipeline(
        FakePipe(),
        device="cuda",
        cpu_offload=False,
        vae_tiling=True,
        vae_slicing=False,
    )

    assert calls == ["tiling", "to:cuda"]


def test_infer_dry_run_flashomni_default_requires_explicit_sparse_info(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(tmp_path, "--model", "wan1.3b", "--method", "flashomni")
    cfg = payload["method_config"]

    assert cfg["implementation"] == "upstream"
    assert cfg["sparse_pattern"] == "explicit"
    assert cfg["backend"] == "auto"
    assert cfg["workspace_bytes"] == 268435456
    assert cfg["causal"] is False
    assert cfg["pos_encoding_mode"] == "NONE"
    assert cfg["use_fp16_qk_reduction"] is False
    assert cfg["logits_soft_cap"] == 0.0
    assert cfg["sm_scale"] is None
    assert cfg["rope_scale"] is None
    assert cfg["rope_theta"] is None
    assert payload["runtime"]["optional_kernels"]["flashomni"]["methods"] == ["flashomni"]
    assert any(
        "sparse_pattern=explicit" in item and "Missing:" in item
        for item in payload["runtime"]["preflight"]["errors"]
    )


def test_infer_dry_run_flashomni_global_random_allows_debug_fallbacks(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--allow-debug-fallbacks",
        "--model",
        "wan1.3b",
        "--method",
        "flashomni",
        "--method-config",
        "sparse_pattern=global_random",
        "--method-config",
        "spq_Q=0.0",
        "--method-config",
        "spq_KV=0.8",
    )
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert cfg["sparse_pattern"] == "global_random"
    assert cfg["spq_Q"] == 0.0
    assert cfg["spq_KV"] == 0.8
    assert cfg["sparse_size"] == 128
    assert not any(
        "sparse_pattern=explicit" in error
        for error in payload["runtime"]["preflight"]["errors"]
    )
    assert any(
        "not a video diffusion quality-parity sparsity policy" in warning
        for warning in payload["runtime"]["preflight"]["warnings"]
    )


def test_infer_dry_run_flashomni_paper_mmdit_warns_not_upstream_parity(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model",
        "hunyuan",
        "--method",
        "flashomni",
        "--method-config",
        "sparse_pattern=paper_mmdit",
        "--method-config",
        "tau_q=0.4",
        "--method-config",
        "tau_kv=0.01",
        "--method-config",
        "N=3",
        "--method-config",
        "D=0",
        "--method-config",
        "S_q=0.0",
    )
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert cfg["sparse_pattern"] == "paper_mmdit"
    assert cfg["tau_q"] == 0.4
    assert cfg["tau_kv"] == 0.01
    assert cfg["N"] == 3
    assert cfg["D"] == 0
    assert cfg["S_q"] == 0.0
    assert cfg["threshold_q"] == 0.4
    assert cfg["threshold_kv"] == 0.01
    assert cfg["fresh_threshold"] == 3
    assert cfg["max_order"] == 0
    assert cfg["saving_threshold_q_for_taylor"] == 0.0
    assert not any(
        "sparse_pattern=explicit" in error
        for error in payload["runtime"]["preflight"]["errors"]
    )
    assert any(
        "paper_mmdit" in warning and "SparseVideo-owned FlashOmni attention" in warning
        for warning in payload["runtime"]["preflight"]["warnings"]
    )
    assert any(
        "GEMM-Q/GEMM-O" in warning
        for warning in payload["runtime"]["preflight"]["warnings"]
    )
    assert any(
        "anonymous FlashOmni Hunyuan sparse-symbol policy" in warning
        and "forward/Taylor-cache patch" in warning
        for warning in payload["runtime"]["preflight"]["warnings"]
    )


def test_infer_dry_run_flashomni_hunyuan_paper_mmdit_uses_quality_safe_defaults(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model",
        "hunyuan",
        "--method",
        "flashomni",
        "--method-config",
        "sparse_pattern=paper_mmdit",
    )
    cfg = payload["method_config"]

    assert payload["status"] == "dry_run"
    assert cfg["sparse_pattern"] == "paper_mmdit"
    assert cfg["max_order"] == 0
    assert cfg["D"] == 0
    assert cfg["use_sparse_gemm"] is False
    assert any(
        "quality-safe defaults" in warning and "max_order=0,use_sparse_gemm=false" in warning
        for warning in payload["runtime"]["preflight"]["warnings"]
    )


def test_infer_flashomni_hunyuan_paper_mmdit_rejects_taylor_gemm_path(tmp_path):
    result = _run_infer(
        tmp_path,
        "--model",
        "hunyuan",
        "--method",
        "flashomni",
        "--method-config",
        "sparse_pattern=paper_mmdit",
        "--method-config",
        "D=1",
        "--method-config",
        "use_sparse_gemm=true",
        "--dry-run",
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "validate_method_config"
    assert "use_sparse_gemm=false" in payload["error"]
    assert "Sparse GEMM projection" in payload["error"]
    assert "quality degradation and performance regression" in payload["error"]


def test_infer_flashomni_default_explicit_fails_before_model_load(tmp_path):
    result = _run_infer(tmp_path, "--model", "wan1.3b", "--method", "flashomni")
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "preflight"
    assert "sparse_pattern=explicit" in payload["error"]
    assert "Missing:" in payload["error"]
    assert payload["timings"] == {}


def test_infer_flashomni_global_random_fails_before_model_load_by_default(tmp_path):
    result = _run_infer(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "flashomni",
        "--method-config", "sparse_pattern=global_random",
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "preflight"
    assert "synthetic kernel benchmark mask" in payload["error"]
    assert payload["timings"] == {}


def test_infer_dry_run_flashomni_loads_explicit_sparse_info_tensor_paths(tmp_path):
    sparse_bundle = tmp_path / "flashomni_sparse.pt"
    torch.save(
        {
            "sparse_info": torch.ones(1, dtype=torch.uint8),
            "sparse_kv_info": torch.ones(1, dtype=torch.uint8),
            "sparse_info_indptr": torch.tensor([0, 1], dtype=torch.int32),
            "sparse_kv_info_indptr": torch.tensor([0, 1], dtype=torch.int32),
        },
        sparse_bundle,
    )

    payload = _run_infer_dry_run(
        tmp_path,
        "--model",
        "wan1.3b",
        "--method",
        "flashomni",
        "--method-config",
        "sparse_pattern=explicit",
        "--method-config",
        f"sparse_info={sparse_bundle}",
        "--method-config",
        f"sparse_kv_info={sparse_bundle}",
        "--method-config",
        f"sparse_info_indptr={sparse_bundle}",
        "--method-config",
        f"sparse_kv_info_indptr={sparse_bundle}",
    )
    cfg = payload["method_config"]

    assert cfg["sparse_pattern"] == "explicit"
    assert cfg["sparse_info"] == {
        "device": "cpu",
        "dtype": "torch.uint8",
        "shape": [1],
        "type": "torch.Tensor",
    }
    assert cfg["sparse_info_indptr"]["dtype"] == "torch.int32"
    assert not any(
        "sparse_pattern=explicit" in error
        for error in payload["runtime"]["preflight"]["errors"]
    )


def test_infer_dry_run_spargeattn_defaults_to_sparse_topk(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "spargeattn")

    assert payload["method_config"]["mode"] == "topk"
    assert payload["method_config"]["topk"] == 0.5
    assert not any("mode=full runs dense attention" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_spargeattn_wan21_upstream_profile_uses_diffusers_example_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "spargeattn",
        "--profile", "upstream",
    )

    assert payload["height"] == 480
    assert payload["width"] == 832
    assert payload["num_frames"] == 81
    assert payload["fps"] == 15
    assert payload["seed"] == 42
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "sequential"
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is True
    assert payload["vae_decoder_chunk_size"] == 1
    assert payload["profile_overrides"]["guidance_scale"] == 5.0
    assert payload["method_config"]["mode"] == "full"
    assert payload["method_config"]["value"] is None
    assert payload["profile_overrides"]["method_config"] == {"mode": "full", "value": None}
    assert any("mode=full runs dense attention" in item for item in payload["runtime"]["preflight"]["warnings"])
    assert payload["profile_overrides"]["source"].endswith(
        "SpargeAttn/inference_examples/wan_infer.py"
    )


def test_spargeattn_wan22_upstream_profile_uses_readme_topk_value(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan22",
        "--method", "spargeattn",
        "--profile", "upstream",
    )

    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_inference_steps"] == 40
    assert payload["seed"] == 42
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "sequential"
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is True
    assert payload["vae_decoder_chunk_size"] == 1
    assert payload["profile_overrides"]["guidance_scale"] == 4.0
    assert payload["profile_overrides"]["guidance_scale_2"] == 3.0
    assert payload["method_config"]["mode"] == "topk"
    assert payload["method_config"]["value"] == 0.4
    assert payload["profile_overrides"]["source"].endswith(
        "SpargeAttn/inference_examples/README.md"
    )


def test_spargeattn_hunyuan_upstream_profile_uses_diffusers_example_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "spargeattn",
        "--profile", "upstream",
    )

    assert payload["height"] == 320
    assert payload["width"] == 512
    assert payload["num_frames"] == 61
    assert payload["num_inference_steps"] == 30
    assert payload["fps"] == 8
    assert payload["seed"] == 42
    assert payload["cpu_offload"] is True
    assert payload["cpu_offload_mode"] == "sequential"
    assert payload["vae_tiling"] is True
    assert payload["vae_slicing"] is True
    assert payload["vae_decoder_chunk_size"] == 1
    assert payload["method_config"]["mode"] == "full"
    assert payload["profile_overrides"]["method_config"] == {"mode": "full", "value": None}
    assert payload["method_config"]["l1"] == 0.07
    assert payload["method_config"]["pv_l1"] == 0.08
    assert payload["method_config"]["tune_pv"] is True
    assert any("mode=full runs dense attention" in item for item in payload["runtime"]["preflight"]["warnings"])
    assert payload["profile_overrides"]["source"].endswith(
        "SpargeAttn/inference_examples/hunyuan_infer.py"
    )


def test_spargeattn_hunyuan_sparse_mode_fails_preflight_before_model_load(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(
        tmp_path,
        "--model", "hunyuan",
        "--method", "spargeattn",
        "--method-config", "mode=topk",
    )

    assert any(
        "HunyuanVideo" in error and "attention_mask" in error
        for error in payload["runtime"]["preflight"]["errors"]
    )


def test_upstream_profile_cpu_offload_keeps_user_override(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "spargeattn",
        "--profile", "upstream",
        "--no-cpu-offload",
    )

    assert payload["cpu_offload"] is False
    assert payload["profile_overrides"]["cpu_offload"] is True


def test_infer_dry_run_warns_when_spargeattn_mode_is_explicit_dense(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model",
        "wan1.3b",
        "--method",
        "spargeattn",
        "--method-config",
        "mode=full",
    )

    assert any("mode=full runs dense attention" in item for item in payload["runtime"]["preflight"]["warnings"])


def test_validate_accepts_spargeattn_topk_upstream_default_value():
    infer = _load_infer_module()

    infer.validate_method_config(
        "spargeattn",
        {"mode": "topk", "topk": 0.5, "l1": 0.07, "pv_l1": 0.08},
    )


def test_validate_rejects_spargeattn_block_sparse_without_mask_id():
    infer = _load_infer_module()

    with pytest.raises(ValueError, match="mask_id"):
        infer.validate_method_config(
            "spargeattn",
            {"mode": "block_sparse", "l1": 0.07, "pv_l1": 0.08},
        )


def test_spargeattn_materializes_mask_id_tensor_path(tmp_path):
    infer = _load_infer_module()
    mask_path = tmp_path / "mask.pt"
    mask = torch.ones(1, 2, 1, 2, dtype=torch.int32)
    torch.save(mask, mask_path)
    cfg = {"mask_id": str(mask_path)}

    infer.materialize_method_config_values("spargeattn", cfg)

    torch.testing.assert_close(cfg["mask_id"], mask)


def test_preflight_rejects_spargeattn_training_free_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": True,
            },
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert any("training_free/" in error for error in preflight["errors"])


def test_preflight_rejects_spargeattn_environment_runtime_without_owned_root():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": _flashinfer_runtime_ready(),
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": False,
                "environment_runtime_detected": True,
                "selected_runtime": "missing",
                "sparsevideo_runtime": {
                    "package": False,
                    "qattn_extension": False,
                    "fused_extension": False,
                },
            },
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert any("Environment spas_sage_attn packages are not accepted" in error for error in preflight["errors"])


def test_preflight_requires_spargeattn_load_checked_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert any("extension/source presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_prefers_owned_spargeattn_runtime_over_training_free():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": _flashinfer_runtime_ready(),
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": True,
                "load_checked": True,
                "imported": True,
                "spas_sage2_attn_meansim_cuda": True,
                "spas_sage2_attn_meansim_topk_cuda": True,
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert preflight == {"errors": [], "warnings": []}


def test_preflight_uses_spas_sage_load_failure_for_spargeattn():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
                "load_checked": True,
                "import_error_type": "ImportError",
                "import_error": "libqattn.so: undefined symbol",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert any("spas_sage_attn failed to import during preflight" in error for error in preflight["errors"])
    assert any("undefined symbol" in error for error in preflight["errors"])


def test_preflight_rejects_hunyuan_spargeattn_sparse_attention_mask_gap():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "svg_svoo_fused_kernels": {
                "backend_env": "auto",
                "native_extension": True,
                "candidate_dirs": [],
            },
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn",
        {"mode": "topk", "value": 0.5},
        "cuda",
        runtime,
        model_family="hunyuan_video",
    )

    assert any(
        "HunyuanVideo" in error and "attention_mask" in error
        for error in preflight["errors"]
    )


def test_preflight_rejects_spargeattn_invalid_env_root_even_when_owned_runtime_ready():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {
                "env_root": {"error": "Refusing SPARSEVIDEO_SPARGEATTN_ROOT inside training_free"},
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "spargeattn", {"mode": "topk", "value": 0.5}, "cuda", runtime,
    )

    assert any("SPARSEVIDEO_SPARGEATTN_ROOT inside training_free" in error for error in preflight["errors"])


def test_preflight_requires_owned_radial_sage_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "environment",
                "sparsevideo_runtime": {
                    "package": False,
                    "qattn_extension": False,
                    "fused_extension": False,
                    "block_sparse_sage2_attn_cuda": False,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial", {"use_sage_attention": True}, "cuda", runtime,
    )

    assert any("block_sparse_sage2_attn_cuda" in error for error in preflight["errors"])


def test_preflight_uses_spas_sage_load_failure_for_radial_sage_path():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
                "load_checked": True,
                "import_error_type": "ImportError",
                "import_error": "bad spas_sage_attn runtime",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 0, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert any("spas_sage_attn failed to import during preflight" in error for error in preflight["errors"])


def test_preflight_uses_sageattention_load_failure_for_radial_dense_warmup():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
                "load_checked": True,
                "imported": True,
                "block_sparse_sage2_attn_cuda": True,
            },
            "sageattention": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
                "load_checked": True,
                "import_error_type": "ImportError",
                "import_error": "bad sageattention runtime",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 12, "dense_layers": 1},
        "cuda",
        runtime,
    )

    assert any("sageattention failed to import during preflight" in error for error in preflight["errors"])


def test_preflight_requires_radial_sage_load_checked_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 0, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert any("extension/source presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_radial_sageattention_load_checked_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
                "load_checked": True,
                "imported": True,
                "block_sparse_sage2_attn_cuda": True,
            },
            "sageattention": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 12, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert any("SparseVideo-owned SageAttention runtime" in error for error in preflight["errors"])
    assert any("extension/source presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_flashinfer_sparse_for_radial():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": False},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("radial", {"use_sage_attention": False}, "cuda", runtime)

    assert any("flashinfer.sparse" in error for error in preflight["errors"])


def test_strict_preflight_requires_flashinfer_sparse_for_svg2():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": False},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svg2_kernels": {
                "triton_package": True,
                "triton_kmeans": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "triton_permute": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime("svg2", {}, "cuda", runtime, strict_kernels=True)

    assert any("flashinfer.sparse" in error for error in preflight["errors"])
    assert not preflight["warnings"]


def test_strict_preflight_uses_flashinfer_load_failure_for_svg2():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {
                "package": True,
                "sparse_module": True,
                "cuda_toolkit": {"available": True},
                "load_checked": True,
                "import_error_type": "ImportError",
                "import_error": "libflashinfer.so: undefined symbol",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svg2_kernels": {
                "triton_package": True,
                "triton_kmeans": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "triton_permute": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime("svg2", {}, "cuda", runtime, strict_kernels=True)

    assert any("flashinfer failed to import during preflight" in error for error in preflight["errors"])
    assert any("undefined symbol" in error for error in preflight["errors"])
    assert not preflight["warnings"]


def test_preflight_requires_svg1_runtime_load_check_when_sources_exist():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flex_attention": {
                "module": True,
                "flex_attention": True,
                "block_mask": True,
                "torch_compile": True,
            },
            "svg1_kernels": {
                "triton_package": True,
                "method_source": {"source_files": True},
                "triton_placement": {"source_files": True},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("svg1", {}, "cuda", runtime, strict_kernels=True)

    assert any("source-file presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_svg1_loaded_runtime_apis():
    infer = _load_infer_module()
    svg1 = _svg1_runtime_ready()
    svg1["owned_triton_runtime"]["sparse_head_placement"] = False
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flex_attention": {
                "module": True,
                "flex_attention": True,
                "block_mask": True,
                "torch_compile": True,
            },
            "svg1_kernels": svg1,
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("svg1", {}, "cuda", runtime, strict_kernels=True)

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("sparse_head_placement" in error for error in preflight["errors"])


def test_preflight_reports_svg1_runtime_import_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flex_attention": {
                "module": True,
                "flex_attention": True,
                "block_mask": True,
                "torch_compile": True,
            },
            "svg1_kernels": {
                "triton_package": True,
                "method_source": {"source_files": True},
                "triton_placement": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "import_error_type": "ImportError",
                    "import_error": "bad svg1 placement",
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("svg1", {}, "cuda", runtime, strict_kernels=True)

    assert any("svg1 owned method/Triton placement modules failed to import" in error for error in preflight["errors"])
    assert any("bad svg1 placement" in error for error in preflight["errors"])


def test_strict_preflight_requires_flex_attention_for_svg1():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flex_attention": {
                "module": False,
                "flex_attention": False,
                "block_mask": False,
                "torch_compile": False,
                "error_type": "ImportError",
                "error": "missing flex attention",
            },
            "svg1_kernels": _svg1_runtime_ready(),
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("svg1", {}, "cuda", runtime, strict_kernels=True)

    assert any("PyTorch FlexAttention APIs" in error for error in preflight["errors"])
    assert any("BlockMask" in error for error in preflight["errors"])
    assert any("torch.compile" in error for error in preflight["errors"])
    assert not preflight["warnings"]


def test_strict_preflight_requires_flash_attn_varlen_for_svg1_hunyuan():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flex_attention": {
                "module": True,
                "flex_attention": True,
                "block_mask": True,
                "torch_compile": True,
            },
            "flash_attn": {
                "package": True,
                "flash_attn_func": True,
                "flash_attn_varlen_func": False,
            },
            "svg1_kernels": _svg1_runtime_ready(),
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "svg1", {}, "cuda", runtime, strict_kernels=True, model_family="hunyuan_video",
    )

    assert any("FlashAttention varlen" in error for error in preflight["errors"])
    assert any("flash_attn_varlen_func" in error for error in preflight["errors"])
    assert not preflight["warnings"]


def test_preflight_requires_cuda_toolkit_for_radial_flashinfer_jit():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {
                "package": True,
                "sparse_module": True,
                "cuda_toolkit": {"available": False},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("radial", {"use_sage_attention": False}, "cuda", runtime)

    assert any("CUDA toolkit with nvcc" in error for error in preflight["errors"])


def test_preflight_requires_radial_runtime_load_check_when_sources_exist():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": {
                "method_source": {"source_files": True},
                "flashinfer_bsr_wrapper": {"source_files": True},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("radial", {"use_sage_attention": False}, "cuda", runtime)

    assert any("source-file presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_radial_loaded_runtime_apis():
    infer = _load_infer_module()
    radial = _radial_runtime_ready()
    radial["owned_runtime"]["radial_bsr_mask"] = False
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": radial,
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("radial", {"use_sage_attention": False}, "cuda", runtime)

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("radial_bsr_mask" in error for error in preflight["errors"])


def test_infer_dry_run_allows_default_wan_radial_partial_block_shape(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "radial")

    assert payload["status"] == "dry_run"
    assert payload["height"] == 720
    assert payload["width"] == 1280
    assert payload["num_frames"] == 81
    assert payload["method_config"]["allow_flex_fallback"] is False
    assert payload["runtime"]["preflight"]["errors"] == []


def test_preflight_reports_radial_runtime_import_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": {
                "method_source": {"source_files": True},
                "flashinfer_bsr_wrapper": {"source_files": True},
                "owned_runtime": {
                    "load_checked": True,
                    "import_error_type": "ImportError",
                    "import_error": "bad radial helper",
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("radial", {"use_sage_attention": False}, "cuda", runtime)

    assert any("radial owned method/BSR wrapper modules failed to import" in error for error in preflight["errors"])
    assert any("bad radial helper" in error for error in preflight["errors"])


def test_preflight_accepts_owned_radial_sage_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": _flashinfer_runtime_ready(),
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": True,
                "selected_runtime": "sparsevideo",
                "load_checked": True,
                "imported": True,
                "block_sparse_sage2_attn_cuda": True,
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial", {"use_sage_attention": True}, "cuda", runtime,
    )

    assert preflight == {"errors": [], "warnings": []}


def test_preflight_blocks_radial_sage_dense_warmup_until_owned_sageattention():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 12, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert any("SparseVideo-owned SageAttention dense backend" in error for error in preflight["errors"])


def test_preflight_accepts_radial_sage_dense_warmup_with_owned_sageattention():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": _flashinfer_runtime_ready(),
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "package": True,
                "qattn_extension": True,
                "fused_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "load_checked": True,
                "imported": True,
                "block_sparse_sage2_attn_cuda": True,
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
            },
            "sageattention": {
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "load_checked": True,
                "imported": True,
                "sageattn": True,
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 12, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert preflight == {"errors": [], "warnings": []}


def test_preflight_rejects_radial_sageattention_invalid_env_root_even_when_owned_runtime_ready():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "radial_kernels": _radial_runtime_ready(),
            "spas_sage_attn": {
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                    "block_sparse_sage2_attn_cuda": True,
                },
            },
            "sageattention": {
                "env_root": {"error": "Refusing SPARSEVIDEO_SAGEATTENTION_ROOT inside training_free"},
                "sparsevideo_runtime": {
                    "package": True,
                    "qattn_extension": True,
                    "fused_extension": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "radial",
        {"use_sage_attention": True, "dense_timesteps": 12, "dense_layers": 0},
        "cuda",
        runtime,
    )

    assert any("SPARSEVIDEO_SAGEATTENTION_ROOT inside training_free" in error for error in preflight["errors"])


def test_radial_use_sage_partial_block_shape_is_allowed_by_preflight():
    infer = _load_infer_module()
    message = infer.radial_flashinfer_layout_warning(
        infer.MODEL_SPECS["wan21-t2v-1.3b"],
        height=720,
        width=1280,
        num_frames=81,
        config={"block_size": 128, "use_sage_attention": True},
    )

    assert message is None


def test_infer_dry_run_warns_for_sta_generalized_wan_shape(tmp_path):
    payload = _run_infer_dry_run(tmp_path, "--model", "wan1.3b", "--method", "sta")
    cfg = payload["method_config"]
    warnings = payload["runtime"]["preflight"]["warnings"]

    assert cfg["tile_size"] == [6, 8, 8]
    assert cfg["window_size"] == [3, 6, 10]
    assert cfg["has_text"] is False
    assert cfg["STA_mode"] == "STA_inference"
    assert cfg["mask_strategy_file_path"].endswith("mask_strategy_wan21_t2v_1_3b.json")
    assert payload["runtime"]["preflight"]["errors"] == []
    assert any("FastVideo STA native shapes" in item for item in warnings)
    assert any("Current latent layout is 21x45x80" in item for item in warnings)
    assert any("tile-padded canvas is 24x48x80" in item for item in warnings)


def test_sta_upstream_profile_sets_native_wan_shape(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "wan14b",
        "--method", "sta",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 69
    assert cfg["seq_shape"] == "18x48x80"
    assert cfg["mask_strategy_file_path"].endswith("mask_strategy_wan21_t2v_14b.json")
    assert payload["profile_overrides"]["source"].endswith("FastVideo/docs/attention/sta/index.md")
    assert not any("Current latent layout is 21x45x80" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("seq_shape is not set" in item for item in payload["runtime"]["preflight"]["errors"])
    assert not any("FastVideo STA native path only covers" in item for item in payload["runtime"]["preflight"]["errors"])


def test_sta_hunyuan_upstream_profile_sets_archive_strategy(tmp_path):
    payload = _run_infer_dry_run(
        tmp_path,
        "--model", "hunyuan",
        "--method", "sta",
        "--profile", "upstream",
    )
    cfg = payload["method_config"]

    assert payload["height"] == 768
    assert payload["width"] == 1280
    assert payload["num_frames"] == 117
    assert cfg["seq_shape"] == "30x48x80"
    assert cfg["mask_strategy_file_path"].endswith("mask_strategy_hunyuan_t2v.json")
    assert payload["profile_overrides"]["source"].endswith("FastVideo/docs/attention/sta/index.md")
    assert not payload["runtime"]["preflight"]["errors"]


def test_sta_wan13b_is_not_allowed_to_borrow_wan14b_strategy(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(
        tmp_path,
        "--model", "wan1.3b",
        "--method", "sta",
        "--height", "768",
        "--width", "1280",
        "--num-frames", "69",
        "--method-config", "seq_shape=18x48x80",
        "--method-config", "mask_strategy_file_path=src/sparsevideo/methods/sta/mask_strategies/mask_strategy_wan.json",
    )

    assert any("expected wan21-t2v-1.3b strategy shape (50, 30, 12)" in item for item in payload["runtime"]["preflight"]["errors"])
    assert any("steps/layers/heads=(50, 40, 40)" in item for item in payload["runtime"]["preflight"]["errors"])


def test_sta_preflight_rejects_training_free_mask_strategy_path(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(
        tmp_path,
        "--model", "wan14b",
        "--method", "sta",
        "--profile", "upstream",
        "--method-config", "mask_strategy_file_path=training_free/FastVideo/docs/attention/sta/index.md",
    )

    assert any("mask_strategy_file_path inside training_free" in item for item in payload["runtime"]["preflight"]["errors"])


def test_infer_dry_run_rejects_sta_seq_shape_mismatch(tmp_path):
    payload = _run_infer_dry_run_preflight_failure(
        tmp_path,
        "--model",
        "wan1.3b",
        "--method",
        "sta",
        "--method-config",
        "seq_shape=18x48x80",
    )

    assert any("does not match the current latent layout 21x45x80" in item for item in payload["runtime"]["preflight"]["errors"])


def test_preflight_rejects_sta_custom_tile_size_as_non_fastvideo_parity():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80", "tile_size": [3, 8, 8]}, "cuda", runtime,
        strict_kernels=False,
    )

    assert any("tile_size differs from FastVideo" in error for error in preflight["errors"])
    assert preflight["warnings"] == []

    strict = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80", "tile_size": [3, 8, 8]}, "cuda", runtime, strict_kernels=True,
    )
    assert any("tile_size differs from FastVideo" in error for error in strict["errors"])
    assert strict["warnings"] == []


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
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True},
            "flashomni": {"package": False, "aot_config": False},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni", {"implementation": "upstream", "sparse_pattern": "explicit"}, "cuda", runtime,
    )

    assert any("flashomni implementation=upstream" in error for error in preflight["errors"])


def test_preflight_flashomni_is_full_does_not_require_sparse_info():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": True},
                "load_checked": True,
                "imported": True,
                "native_extension_imported": True,
                "owned_runtime": True,
                "batch_flashomni_fa_with_ragged_kv_wrapper": True,
                "segment_packbits": True,
                "torch_ops_flashomni_kernels": True,
                "torch_ops_batch_sparseFA_with_kv_plan": True,
                "torch_ops_batch_sparseFA_with_ragged_kv_run": True,
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
    )

    assert not any("sparse_pattern=explicit" in error for error in preflight["errors"])
    assert any("is_full=true disables sparsity" in warning for warning in preflight["warnings"])


def test_preflight_adacluster_uses_owned_triton_not_flashinfer():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "adacluster_kernels": {
                "triton_package": True,
                "load_checked": True,
                "fast_kmeans_single": {"source_files": True},
                "triton_cluster_sparse_attn": {"source_files": True},
                "triton_cluster_sparse_attn_topk": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "imported": True,
                    "owned_runtime": True,
                    "flash_kmeans_single": True,
                    "triton_cluster_sparse_attn": True,
                    "triton_cluster_sparse_attn_topk": True,
                    "kmeans_jit_kernels": True,
                    "cluster_sparse_attn_jit_kernel": True,
                    "cluster_sparse_attn_topk_jit_kernel": True,
                },
            },
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": False, "sparse_module": False},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("adacluster", {}, "cuda", runtime)

    assert preflight == {"errors": [], "warnings": []}


def test_preflight_requires_flash_attn_for_hunyuan_adacluster_dense_gates():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "adacluster_kernels": {
                "triton_package": True,
                "load_checked": True,
                "fast_kmeans_single": {"source_files": True},
                "triton_cluster_sparse_attn": {"source_files": True},
                "triton_cluster_sparse_attn_topk": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "imported": True,
                    "owned_runtime": True,
                    "flash_kmeans_single": True,
                    "triton_cluster_sparse_attn": True,
                    "triton_cluster_sparse_attn_topk": True,
                    "kmeans_jit_kernels": True,
                    "cluster_sparse_attn_jit_kernel": True,
                    "cluster_sparse_attn_topk_jit_kernel": True,
                },
            },
            "flash_attn": {"package": True, "flash_attn_func": False},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "adacluster", {}, "cuda", runtime, model_family="hunyuan_video",
    )

    assert any("Hunyuan dense gates require FlashAttention" in error for error in preflight["errors"])
    assert any("flash_attn_func" in error for error in preflight["errors"])


def test_preflight_requires_adacluster_load_checked_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "adacluster_kernels": {
                "triton_package": True,
                "fast_kmeans_single": {"source_files": True},
                "triton_cluster_sparse_attn": {"source_files": True},
                "triton_cluster_sparse_attn_topk": {"source_files": True},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("adacluster", {}, "cuda", runtime)

    assert any("source-file presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_reports_adacluster_load_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "adacluster_kernels": {
                "triton_package": True,
                "load_checked": True,
                "fast_kmeans_single": {"source_files": True},
                "triton_cluster_sparse_attn": {"source_files": True},
                "triton_cluster_sparse_attn_topk": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "import_error_type": "ImportError",
                    "import_error": "bad triton jit abi",
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("adacluster", {}, "cuda", runtime)

    assert any("owned Triton kernels failed to import" in error for error in preflight["errors"])
    assert any("bad triton jit abi" in error for error in preflight["errors"])


def test_preflight_requires_adacluster_loaded_apis():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "adacluster_kernels": {
                "triton_package": True,
                "load_checked": True,
                "fast_kmeans_single": {"source_files": True},
                "triton_cluster_sparse_attn": {"source_files": True},
                "triton_cluster_sparse_attn_topk": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "imported": True,
                    "owned_runtime": True,
                    "flash_kmeans_single": True,
                    "triton_cluster_sparse_attn": False,
                    "triton_cluster_sparse_attn_topk": True,
                    "kmeans_jit_kernels": True,
                    "cluster_sparse_attn_jit_kernel": True,
                    "cluster_sparse_attn_topk_jit_kernel": True,
                },
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("adacluster", {}, "cuda", runtime)

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("triton_cluster_sparse_attn" in error for error in preflight["errors"])


def test_preflight_draft_requires_owned_mit_block_sparse_backend():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "triton_package": True,
                "mit_block_sparse_attn": {
                    "source_files": False,
                    "cuda_extension": False,
                    "selected_runtime": "missing",
                },
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime)

    assert any("MIT Han Lab Block-Sparse-Attention" in error for error in preflight["errors"])


def test_preflight_requires_flash_attn_varlen_for_draft_dense_gates():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "triton_package": True,
                "mit_block_sparse_attn": {
                    "source_files": True,
                    "cuda_extension": True,
                    "selected_runtime": "sparsevideo",
                },
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": False},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=True)

    assert any("draft dense gates require FlashAttention varlen" in error for error in preflight["errors"])
    assert any("flash_attn_varlen_func" in error for error in preflight["errors"])


def test_preflight_uses_flash_attn_load_failure_for_draft_dense_gates():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "triton_package": True,
                "mit_block_sparse_attn": {
                    "source_files": True,
                    "cuda_extension": True,
                    "selected_runtime": "sparsevideo",
                },
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {
                "package": True,
                "flash_attn_varlen_func": True,
                "load_checked": True,
                "import_error_type": "ImportError",
                "import_error": "libflash_attn.so: undefined symbol",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=True)

    assert any("flash_attn failed to import during preflight" in error for error in preflight["errors"])
    assert any("undefined symbol" in error for error in preflight["errors"])


def test_preflight_uses_draft_mit_load_failure_for_sparse_backend():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "mit_load_checked": True,
                "mit_block_sparse_attn": {
                    "source_files": True,
                    "cuda_extension": True,
                    "selected_runtime": "sparsevideo",
                    "load_checked": True,
                    "import_error_type": "ImportError",
                    "import_error": "undefined symbol: fwd_block",
                },
                "triton_package": True,
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=True)

    assert any("MIT Block-Sparse-Attention backend failed to import" in error for error in preflight["errors"])
    assert any("undefined symbol: fwd_block" in error for error in preflight["errors"])


def test_preflight_requires_draft_mit_loaded_apis():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "mit_load_checked": True,
                "mit_block_sparse_attn": {
                    "source_files": True,
                    "cuda_extension": True,
                    "selected_runtime": "sparsevideo",
                    "load_checked": True,
                    "imported": True,
                    "cuda_extension_imported": True,
                    "owned_runtime": True,
                    "block_sparse_attn_func": True,
                    "cuda_fwd_block": False,
                    "cuda_bwd_block": True,
                },
                "triton_package": True,
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=True)

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("cuda_fwd_block" in error for error in preflight["errors"])


def test_preflight_draft_allows_generic_triton_debug_fallback_only_when_explicit():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "triton_package": True,
                "mit_block_sparse_attn": {
                    "source_files": False,
                    "cuda_extension": False,
                    "selected_runtime": "missing",
                },
                "triton_block_sparse_attn": {"source_files": True},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=False)

    assert preflight["errors"] == []
    assert any("debug fallback only" in warning for warning in preflight["warnings"])


def test_preflight_draft_requires_owned_block_sparse_source():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "draft_kernels": {
                "triton_package": True,
                "mit_block_sparse_attn": {
                    "source_files": False,
                    "cuda_extension": False,
                    "selected_runtime": "missing",
                },
                "triton_block_sparse_attn": {"source_files": False},
            },
            "flash_attn": {"package": True, "flash_attn_varlen_func": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime("draft", {}, "cuda", runtime, strict_kernels=False)

    assert any("generic Triton block sparse source" in error for error in preflight["errors"])


def test_preflight_warns_flashomni_local_qk_topk_is_not_parity():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "sparsevideo_owned_source": {"source_files": False},
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "local_qk_topk"},
        "cuda",
        runtime,
        strict_kernels=False,
    )

    assert preflight["errors"] == []
    assert any("local_qk_topk" in warning for warning in preflight["warnings"])
    assert any("package-ready kernel parity" in warning for warning in preflight["warnings"])


def test_strict_preflight_fails_flashomni_without_owned_native_source():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "sparsevideo_owned_source": {"source_files": False},
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "local_qk_topk"},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("SparseVideo-owned FlashOmni native source" in error for error in preflight["errors"])
    assert any("local_qk_topk" in error for error in preflight["errors"])


def test_preflight_prefers_owned_flashomni_runtime_over_environment():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": True},
                "load_checked": True,
                "imported": True,
                "native_extension_imported": True,
                "owned_runtime": True,
                "batch_flashomni_fa_with_ragged_kv_wrapper": True,
                "segment_packbits": True,
                "torch_ops_flashomni_kernels": True,
                "torch_ops_batch_sparseFA_with_kv_plan": True,
                "torch_ops_batch_sparseFA_with_ragged_kv_run": True,
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert preflight["errors"] == []
    assert all("environment runtime" not in warning for warning in preflight["warnings"])


def test_preflight_requires_flashomni_load_checked_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": True},
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("extension/source presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_uses_flashomni_load_failure_for_upstream_runtime():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": True},
                "load_checked": True,
                "imported": True,
                "import_error_type": "ImportError",
                "import_error": "libflashomni_kernels.so: undefined symbol",
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("flashomni failed to import during preflight" in error for error in preflight["errors"])
    assert any("undefined symbol" in error for error in preflight["errors"])


def test_preflight_requires_flashomni_loaded_upstream_apis():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "selected_runtime": "sparsevideo",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": True},
                "load_checked": True,
                "imported": True,
                "native_extension_imported": True,
                "owned_runtime": True,
                "batch_flashomni_fa_with_ragged_kv_wrapper": False,
                "segment_packbits": True,
                "torch_ops_flashomni_kernels": True,
                "torch_ops_batch_sparseFA_with_kv_plan": True,
                "torch_ops_batch_sparseFA_with_ragged_kv_run": True,
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("Missing FlashOmni API" in error for error in preflight["errors"])
    assert any("batch_flashomni_fa_with_ragged_kv_wrapper" in error for error in preflight["errors"])


def test_preflight_rejects_flashomni_invalid_env_root_even_when_owned_runtime_ready():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "env_root": {"error": "Refusing SPARSEVIDEO_FLASHOMNI_ROOT inside training_free"},
                "sparsevideo_runtime": {"ready": True},
                "sparsevideo_owned_source": {"source_files": True},
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("SPARSEVIDEO_FLASHOMNI_ROOT inside training_free" in error for error in preflight["errors"])


def test_preflight_rejects_flashomni_environment_runtime_without_owned_root():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {
                "package": True,
                "aot_config": True,
                "native_extension": True,
                "training_free_runtime": False,
                "environment_runtime_detected": True,
                "selected_runtime": "missing",
                "sparsevideo_owned_source": {"source_files": True},
                "sparsevideo_runtime": {"ready": False},
            },
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "flashomni",
        {"implementation": "upstream", "sparse_pattern": "explicit", "is_full": True},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("Environment flashomni packages are not accepted" in error for error in preflight["errors"])


def test_validate_rejects_draft_dense_switch():
    infer = _load_infer_module()

    with pytest.raises(NotImplementedError, match="block_sparse_attention=False"):
        infer.validate_method_config(
            "draft",
            {"block_sparse_attention": False},
        )


def test_validate_accepts_spargeattn_tuning_options_before_model_load(tmp_path):
    infer = _load_infer_module()
    cfg = {
        "mode": "full",
        "value": None,
        "tune": True,
        "parallel_tune": True,
        "sim_rule": "rmse",
        "l1": 0.07,
        "pv_l1": 0.08,
        "cos_sim": 0.98,
        "rmse": 0.07,
        "rearrange_kwargs": {},
        "tune_pv": True,
        "verbose": True,
        "model_out_path": str(tmp_path / "state.pt"),
    }

    infer.validate_method_config("spargeattn", cfg)


def test_spargeattn_tune_defaults_model_out_path_to_output_file(tmp_path):
    infer = _load_infer_module()
    cfg = {"tune": True, "model_out_path": None}
    output_file = tmp_path / "video.mp4"

    infer.normalize_spargeattn_model_out_path(cfg, output_file)

    assert cfg["model_out_path"] == str(tmp_path / "video.spargeattn_state.pt")


def test_infer_module_does_not_expose_legacy_scheduler_threshold_resolution():
    infer = _load_infer_module()

    assert not hasattr(infer, "resolve_scheduler_first_times_fp")


def test_validate_accepts_svoo_sparsity_measurement_options(tmp_path):
    infer = _load_infer_module()

    infer.validate_method_config(
        "svoo",
        {
            "implementation": "native",
            "sparse_backend": "flashinfer",
            "measure_attention_sparsity": True,
            "sparsity_output_file": str(tmp_path / "attention_sparsity.txt"),
            "sparsity_batch_size": 4,
            "sparsity_query_samples": 2,
            "sparsity_threshold": 0.95,
            "sparsity_start_step": 1,
        },
    )


def test_validate_accepts_svoo_global_constraint_options():
    infer = _load_infer_module()

    infer.validate_method_config(
        "svoo",
        {
            "implementation": "native",
            "sparse_backend": "flashinfer",
            "use_global_constraints": True,
            "lambda_schedule": "cosine",
            "diverse_top_p_k": 0.1,
        },
    )


def test_infer_dry_run_reports_preflight_failure_as_json(tmp_path):
    env = os.environ.copy()
    env["SVOO_CACHE_ROOT"] = str(tmp_path / "svoo-cache")
    env["SPARSEVIDEO_SPARGEATTN_ROOT"] = str(REPO_ROOT / "training_free" / "SpargeAttn")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "wan1.3b",
            "--method",
            "spargeattn",
            "--method-config",
            "mode=topk",
            "--method-config",
            "value=0.5",
            "--dry-run",
            "--print-json",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert result.stderr == ""
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "preflight"
    assert payload["error_type"] == "RuntimeError"
    assert "SPARSEVIDEO_SPARGEATTN_ROOT inside training_free" in payload["error"]
    assert payload["runtime"]["preflight"]["errors"] == [payload["error"]]


def test_validate_rejects_svoo_missing_dynamic_sparsity_csv(tmp_path):
    infer = _load_infer_module()

    with pytest.raises(FileNotFoundError, match="sparsity_csv_path"):
        infer.validate_method_config(
            "svoo",
            {
                "implementation": "native",
                "sparse_backend": "flashinfer",
                "use_dynamic_min_kc_ratio": True,
                "sparsity_csv_path": str(tmp_path / "missing.csv"),
            },
        )


def test_validate_rejects_svoo_training_free_dynamic_sparsity_csv():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="inside training_free"):
        infer.validate_method_config(
            "svoo",
            {
                "implementation": "native",
                "sparse_backend": "flashinfer",
                "use_dynamic_min_kc_ratio": True,
                "sparsity_csv_path": "training_free/SVOO/sparsity_profiles/sparsity_wan_1.3B_t2v.csv",
            },
        )


def test_validate_resolves_svoo_dynamic_sparsity_csv_path():
    infer = _load_infer_module()

    config = {
        "implementation": "native",
        "sparse_backend": "flashinfer",
        "kmeans_iter_init": 2,
        "kmeans_iter_step": 2,
        "use_dynamic_min_kc_ratio": True,
        "sparsity_csv_path": "src/sparsevideo/methods/svoo/sparsity_profiles/sparsity_wan_1.3B_t2v.csv",
    }

    infer.validate_method_config("svoo", config)

    assert config["sparsity_csv_path"] == str(
        (infer.REPO_ROOT / "src/sparsevideo/methods/svoo/sparsity_profiles/sparsity_wan_1.3B_t2v.csv").resolve()
    )


def test_default_svoo_sparsity_csv_rejects_unprofiled_models():
    infer = _load_infer_module()
    spec = infer.MODEL_SPECS["cogvideox-t2v"]

    with pytest.raises(ValueError, match="no owned offline sparsity profile"):
        infer.default_svoo_sparsity_csv_path(spec)


def test_default_svoo_sparsity_csv_resolves_i2v_profiles():
    infer = _load_infer_module()

    assert infer.default_svoo_sparsity_csv_path(
        infer.MODEL_SPECS["wan21-i2v-14b"]
    ).endswith("sparsity_wan_14B_i2v.csv")
    assert infer.default_svoo_sparsity_csv_path(
        infer.MODEL_SPECS["wan22-i2v-a14b"]
    ).endswith("sparsity_wan22_A14B_i2v.csv")
    assert infer.default_svoo_sparsity_csv_path(
        infer.MODEL_SPECS["hunyuan-i2v"]
    ).endswith("sparsity_hunyuan10_13B_i2v.csv")


def test_infer_dry_run_resolves_svoo_i2v_profiles(tmp_path):
    cases = [
        ("wan14b-i2v", "sparsity_wan_14B_i2v.csv", True),
        ("wan22-i2v", "sparsity_wan22_A14B_i2v.csv", True),
        ("hunyuan-i2v", "sparsity_hunyuan10_13B_i2v.csv", True),
        ("skyreels-v2-i2v", "sparsity_profiles/sparsity_results.csv", False),
    ]

    for model, profile_name, use_dynamic in cases:
        payload = _run_infer_dry_run(
            tmp_path,
            "--model",
            model,
            "--method",
            "svoo",
            "--image",
            "/tmp/nonexistent.jpg",
        )
        cfg = payload["method_config"]

        assert payload["status"] == "dry_run"
        assert cfg["use_dynamic_min_kc_ratio"] is use_dynamic
        assert cfg["sparsity_csv_path"].endswith(profile_name)


def test_svoo_warmup_status_fails_strict_when_disabled():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="warmup is disabled"):
        infer.validate_svoo_warmup_status(
            {"enabled": False, "ran": False, "reason": "disabled"},
            strict_kernels=True,
        )


def test_svoo_warmup_status_reports_debug_warning_when_not_strict():
    infer = _load_infer_module()

    message = infer.validate_svoo_warmup_status(
        {"enabled": True, "ran": False, "reason": "non_cuda"},
        strict_kernels=False,
    )

    assert message == "SVOO kernel warmup did not run: non_cuda"


def test_svoo_warmup_status_fails_strict_on_kernel_error():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="TritonError"):
        infer.validate_svoo_warmup_status(
            {"enabled": True, "ran": False, "error": "TritonError: bad launch"},
            strict_kernels=True,
        )


def test_sparse_runtime_dispatch_accepts_observed_sparse_calls():
    infer = _load_infer_module()

    message = infer.validate_sparse_runtime_dispatch(
        "svoo",
        {},
        {
            "method_runtime": {
                "total_calls": 8,
                "dispatch_counts": {"dense": 2, "sparse": 6},
                "backend_counts": {"svoo_flashinfer": 6, "torch_sdpa": 2},
            },
        },
        strict_kernels=True,
    )

    assert message is None


def test_sparse_runtime_dispatch_fails_strict_without_backend_evidence():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="expected sparse backend"):
        infer.validate_sparse_runtime_dispatch(
            "svoo",
            {},
            {"method_runtime": {"total_calls": 8, "dispatch_counts": {"sparse": 8}}},
            strict_kernels=True,
        )


def test_sparse_runtime_dispatch_fails_strict_when_no_sparse_calls():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="did not dispatch sparse attention"):
        infer.validate_sparse_runtime_dispatch(
            "svg2",
            {},
            {"method_runtime": {"total_calls": 4, "dispatch_counts": {"dense": 4}}},
            strict_kernels=True,
        )


def test_sparse_runtime_dispatch_accepts_sta_searching_calls():
    infer = _load_infer_module()

    message = infer.validate_sparse_runtime_dispatch(
        "sta",
        {"STA_mode": "STA_searching"},
        {"method_runtime": {"total_calls": 4, "dispatch_counts": {"search": 4}}},
        strict_kernels=True,
    )

    assert message is None


def test_sparse_runtime_dispatch_warns_when_debug_fallbacks_allowed():
    infer = _load_infer_module()

    message = infer.validate_sparse_runtime_dispatch(
        "draft",
        {},
        {"method_runtime": {"total_calls": 4, "dispatch_counts": {"dense": 4}}},
        strict_kernels=False,
    )

    assert "draft did not dispatch sparse attention" in message


def test_sparse_runtime_dispatch_fails_strict_on_debug_fallback_backend():
    infer = _load_infer_module()

    with pytest.raises(RuntimeError, match="debug fallback backend"):
        infer.validate_sparse_runtime_dispatch(
            "draft",
            {},
            {
                "method_runtime": {
                    "total_calls": 4,
                    "dispatch_counts": {"sparse": 4},
                    "backend_counts": {"triton_debug_fallback": 4},
                },
            },
            strict_kernels=True,
        )


def test_sparse_runtime_dispatch_warns_on_debug_fallback_backend_when_allowed():
    infer = _load_infer_module()

    message = infer.validate_sparse_runtime_dispatch(
        "flashomni",
        {},
        {
            "method_runtime": {
                "total_calls": 4,
                "dispatch_counts": {"sparse": 4},
                "backend_counts": {"flex_debug_fallback": 4},
            },
        },
        strict_kernels=False,
    )

    assert "debug fallback backend" in message


def test_sparse_runtime_dispatch_skips_explicit_dense_configurations():
    infer = _load_infer_module()
    summary = {"method_runtime": {"total_calls": 4, "dispatch_counts": {"dense": 4}}}

    assert infer.validate_sparse_runtime_dispatch("dense", {}, summary, strict_kernels=True) is None
    assert infer.validate_sparse_runtime_dispatch(
        "spargeattn", {"mode": "full"}, summary, strict_kernels=True,
    ) is None
    assert infer.validate_sparse_runtime_dispatch(
        "flashomni", {"is_full": True}, summary, strict_kernels=True,
    ) is None


def test_run_fails_svoo_strict_when_warmup_is_disabled(monkeypatch, tmp_path):
    infer = _load_infer_module()
    import sparsevideo
    import sparsevideo._runtime as sparsevideo_runtime
    from sparsevideo.methods.svoo import warmup as svoo_warmup

    restored = {"value": False}

    class _Handle:
        def restore(self):
            restored["value"] = True

    class _Pipe:
        pass

    monkeypatch.setattr(infer, "load_pipeline", lambda *args, **kwargs: _Pipe())
    monkeypatch.setattr(infer, "prepare_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer, "preflight_runtime", lambda *args, **kwargs: {"errors": [], "warnings": []})
    monkeypatch.setattr(
        sparsevideo_runtime,
        "optional_kernel_status",
        lambda: {"svg_svoo_fused_kernels": {}},
    )
    monkeypatch.setattr(sparsevideo_runtime, "native_kernel_load_status", lambda: {})
    monkeypatch.setattr(sparsevideo_runtime, "torch_runtime_status", lambda: {"cuda_available": False})
    monkeypatch.setattr(sparsevideo, "apply_sparse_attention", lambda *args, **kwargs: _Handle())
    monkeypatch.setattr(
        svoo_warmup,
        "warmup_svoo_kernels_from_pipeline",
        lambda *args, **kwargs: {"enabled": False, "ran": False, "reason": "disabled"},
    )

    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan1.3b",
            "--method",
            "svoo",
            "--device",
            "cpu",
            "--method-config",
            "sparse_backend=triton",
            "--method-config",
            "use_dynamic_min_kc_ratio=false",
            "--metrics-file",
            str(tmp_path / "metrics.jsonl"),
        ]
    )

    assert infer.run(args) == 1
    assert restored["value"] is True
    payload = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failed_stage"] == "svoo_kernel_warmup"
    assert payload["svoo_kernel_warmup"]["reason"] == "disabled"
    assert "warmup is disabled" in payload["error"]


def test_run_fails_strict_when_sparse_method_dispatches_only_dense(monkeypatch, tmp_path):
    infer = _load_infer_module()
    import sparsevideo
    import sparsevideo._runtime as sparsevideo_runtime

    restored = {"value": False}

    class _Handle:
        def summary(self):
            return {
                "method_runtime": {
                    "total_calls": 3,
                    "dispatch_counts": {"dense": 3},
                    "backend_counts": {"torch_sdpa": 3},
                    "last_dispatch": {"dispatch": "dense", "backend": "torch_sdpa"},
                },
                "restored": restored["value"],
            }

        def restore(self):
            restored["value"] = True

    class _Pipe:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(frames=[["frame"]])

    monkeypatch.setattr(infer, "load_pipeline", lambda *args, **kwargs: _Pipe())
    monkeypatch.setattr(infer, "prepare_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer, "preflight_runtime", lambda *args, **kwargs: {"errors": [], "warnings": []})
    monkeypatch.setattr(
        sparsevideo_runtime,
        "optional_kernel_status",
        lambda: {"svg_svoo_fused_kernels": {}},
    )
    monkeypatch.setattr(sparsevideo_runtime, "native_kernel_load_status", lambda: {})
    monkeypatch.setattr(sparsevideo_runtime, "torch_runtime_status", lambda: {"cuda_available": False})
    monkeypatch.setattr(sparsevideo, "apply_sparse_attention", lambda *args, **kwargs: _Handle())

    args = infer.build_parser().parse_args(
        [
            "--model",
            "wan1.3b",
            "--method",
            "svg1",
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
    assert payload["failed_stage"] == "validate_sparse_dispatch"
    assert "did not dispatch sparse attention" in payload["error"]
    assert payload["sparse_attention_handle"]["method_runtime"]["dispatch_counts"] == {"dense": 3}
    assert payload["sparse_attention_handle_after_restore"]["restored"] is True


def test_preflight_requires_flashinfer_sparse_for_svoo():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": False},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("flashinfer.sparse" in error for error in preflight["errors"])
    assert any("SparseVideo _kernels extension is not detected" in error for error in preflight["errors"])


def test_preflight_requires_svoo_owned_triton_sources():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": False},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("co-clustering source" in error for error in preflight["errors"])


def test_preflight_requires_svoo_owned_triton_permute_source():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": False},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("Triton permutation source" in error for error in preflight["errors"])


def test_preflight_requires_svoo_owned_triton_l2norm_source():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": False},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("Triton L2 normalization source" in error for error in preflight["errors"])


def test_preflight_requires_cuda_toolkit_for_svoo_flashinfer_jit():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": False}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo", {"implementation": "native", "sparse_backend": "flashinfer"}, "cuda", runtime,
    )

    assert any("CUDA toolkit with nvcc" in error for error in preflight["errors"])


def test_preflight_rejects_hunyuan_svoo_triton_backend_as_non_upstream():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "hunyuan_sparse_forward_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "triton"},
        "cuda",
        runtime,
        model_family="hunyuan_video",
    )

    assert any("not Hunyuan SVOO parity" in error for error in preflight["errors"])


def test_preflight_requires_svoo_triton_kmeans_when_svoo_disabled():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "triton_kmeans": {"source_files": False},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {
            "implementation": "native",
            "sparse_backend": "triton",
            "use_svoo": False,
        },
        "cuda",
        runtime,
    )

    assert any("Triton k-means source" in error for error in preflight["errors"])


def test_preflight_requires_svoo_runtime_load_check_when_sources_exist():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "flashinfer"},
        "cuda",
        runtime,
    )

    assert any("source-file presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_svoo_loaded_runtime_apis():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "load_checked": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "imported": True,
                    "owned_runtime": True,
                    "triton_l2norm_forward": True,
                    "triton_layernorm_forward": True,
                    "triton_modulate_shift_forward": True,
                    "triton_modulate_gate_residual_forward": True,
                    "co_cluster_tokens": False,
                    "co_cluster_assign": True,
                    "identify_dynamic_map": True,
                    "permute_tensor_by_labels_triton": True,
                    "apply_inverse_permutation_triton": True,
                    "block_sparse_attention": True,
                    "variable_block_sparse_attn": True,
                    "hunyuan_flashinfer_varlen_attn": True,
                },
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "flashinfer"},
        "cuda",
        runtime,
    )

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("co_cluster_tokens" in error for error in preflight["errors"])


def test_preflight_requires_svg2_owned_triton_sources():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svg2_kernels": {
                "triton_package": True,
                "triton_kmeans": {"source_files": False},
                "dynamic_map": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime("svg2", {}, "cuda", runtime)

    assert any("Triton k-means source" in error for error in preflight["errors"])


def test_preflight_requires_svg2_runtime_load_check_when_sources_exist():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svg2_kernels": {
                "triton_package": True,
                "triton_kmeans": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "triton_permute": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime("svg2", {}, "cuda", runtime)

    assert any("source-file presence alone is not enough" in error for error in preflight["errors"])


def test_preflight_requires_svg2_loaded_runtime_apis():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
            "svg2_kernels": {
                "triton_package": True,
                "load_checked": True,
                "triton_kmeans": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "triton_permute": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "owned_triton_runtime": {
                    "load_checked": True,
                    "imported": True,
                    "owned_runtime": True,
                    "triton_kmeans": True,
                    "euclid_assign_triton": True,
                    "centroid_update_triton": True,
                    "identify_dynamic_map": False,
                    "identify_dynamic_map_global": True,
                    "block_sparse_attention": True,
                    "permute_tensor_by_labels_triton": True,
                    "apply_inverse_permutation_triton": True,
                    "variable_block_sparse_attn": True,
                    "hunyuan_flashinfer_varlen_attn": True,
                },
            },
        },
    }

    preflight = infer.preflight_runtime("svg2", {}, "cuda", runtime)

    assert any("missing loadable API" in error for error in preflight["errors"])
    assert any("identify_dynamic_map" in error for error in preflight["errors"])


def test_strict_preflight_fails_svoo_missing_native_fused_kernel():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": False, "candidate_dirs": []},
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
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


def test_strict_preflight_reports_svoo_native_fused_import_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {
                "backend_env": "auto",
                "built_extension": True,
                "native_extension": False,
                "native_load_checked": True,
                "native_import_error_type": "ImportError",
                "native_import_error": "libc10.so: cannot open shared object file",
                "candidate_dirs": [],
            },
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "flashinfer"},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("built but failed to load" in error for error in preflight["errors"])
    assert any("libc10.so" in error for error in preflight["errors"])
    assert preflight["warnings"] == []


def test_strict_preflight_reports_svoo_training_free_native_root():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_h100": {"native_extension": False},
            },
            "flashinfer": {"package": True, "sparse_module": True, "cuda_toolkit": {"available": True}},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {
                "backend_env": "auto",
                "built_extension": False,
                "native_extension": False,
                "native_load_checked": True,
                "native_import_error_type": "RuntimeError",
                "native_import_error": (
                    "Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free; "
                    "SparseVideo native kernels must be built under src/sparsevideo."
                ),
                "candidate_dirs": [],
            },
            "svoo_kernels": {
                "triton_package": True,
                "triton_l2norm": {"source_files": True},
                "triton_layernorm": {"source_files": True},
                "triton_modulate": {"source_files": True},
                "wan_fast_block_patch": {"source_files": True},
                "co_cluster": {"source_files": True},
                "dynamic_map": {"source_files": True},
                "triton_permute": {"source_files": True},
                "triton_block_sparse_attn": {"source_files": True},
                "flashinfer_block_sparse": {"source_files": True},
                "sparsity_counts": {"source_files": True},
                "sparsity_profiler": {"source_files": True},
            },
        },
    }

    preflight = infer.preflight_runtime(
        "svoo",
        {"implementation": "native", "sparse_backend": "flashinfer"},
        "cuda",
        runtime,
        strict_kernels=True,
    )

    assert any("Native root error: RuntimeError" in error for error in preflight["errors"])
    assert any("SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free" in error for error in preflight["errors"])
    assert preflight["warnings"] == []


def test_strict_preflight_warns_sta_inferred_shape_boundary():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True},
        "optional_kernels": {
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": None}, "cuda", runtime, strict_kernels=True,
    )

    assert preflight["errors"] == []
    assert any("seq_shape is not set" in warning for warning in preflight["warnings"])


def test_strict_preflight_fails_sta_h100_missing_owned_extension():
    infer = _load_infer_module()
    runtime = {
        "torch": {
            "cuda_available": True,
            "cuda_devices": [{"capability": [9, 0], "name": "NVIDIA H100"}],
        },
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": False, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80"}, "cuda", runtime, strict_kernels=True,
    )

    assert any("H100/TK C++ parity kernel" in error for error in preflight["errors"])
    assert preflight["warnings"] == []


def test_preflight_reports_sta_triton_load_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {"cuda_available": True, "cuda_devices": [{"capability": [8, 0]}]},
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "triton_load_checked": True,
                "triton_import_error_type": "ImportError",
                "triton_import_error": "bad triton abi",
                "sparsevideo_h100": {"native_extension": True, "source": {"source_files": True}},
                "h100_native_load_checked": True,
                "h100_native_extension_imported": True,
                "h100_sta_fwd": True,
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80"}, "cuda", runtime, strict_kernels=True,
    )

    assert any("Triton fallback failed to import during preflight" in error for error in preflight["errors"])
    assert any("bad triton abi" in error for error in preflight["errors"])


def test_preflight_reports_sta_h100_load_failure():
    infer = _load_infer_module()
    runtime = {
        "torch": {
            "cuda_available": True,
            "cuda_devices": [{"capability": [9, 0], "name": "NVIDIA H100"}],
        },
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "triton_load_checked": True,
                "triton_sliding_tile_attention_triton": True,
                "sparsevideo_h100": {"native_extension": True, "source": {"source_files": True}},
                "h100_native_load_checked": True,
                "h100_native_extension_imported": False,
                "h100_sta_fwd": False,
                "h100_import_error_type": "ImportError",
                "h100_import_error": "undefined symbol: sta_fwd",
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80"}, "cuda", runtime, strict_kernels=True,
    )

    assert any("H100/TK C++ extension failed to load during preflight" in error for error in preflight["errors"])
    assert any("undefined symbol: sta_fwd" in error for error in preflight["errors"])


def test_preflight_allows_fastvideo_sta_triton_fallback_on_a100():
    infer = _load_infer_module()
    runtime = {
        "torch": {
            "cuda_available": True,
            "cuda_devices": [{"capability": [8, 0], "name": "NVIDIA A100"}],
        },
        "optional_kernels": {
            "flashinfer": {"package": True, "sparse_module": True},
            "flashomni": {"package": True, "aot_config": True},
            "spas_sage_attn": {"package": True, "qattn_extension": True, "fused_extension": True},
            "sta_kernels": {
                "sparsevideo_fastvideo_triton": {"source_files": True},
                "sparsevideo_h100": {"native_extension": True, "source": {"source_files": True}},
            },
            "svg_svoo_fused_kernels": {"backend_env": "auto", "native_extension": True, "candidate_dirs": []},
        },
    }

    preflight = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80"}, "cuda", runtime, strict_kernels=False,
    )

    assert preflight["errors"] == []
    assert any("no Hopper GPU is visible" in warning for warning in preflight["warnings"])

    strict = infer.preflight_runtime(
        "sta", {"seq_shape": "18x48x80"}, "cuda", runtime, strict_kernels=True,
    )
    assert strict["errors"] == []
    assert any("non-Hopper fallback path" in warning for warning in strict["warnings"])


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
    assert payload["sparse_attention_handle"] == {
        "type": "_Handle",
        "summary_available": False,
    }
    assert payload["sparse_attention_handle_after_restore"] == {
        "type": "_Handle",
        "summary_available": False,
    }


def test_hunyuan_i2v_prompt_template_compat_overrides_missing_default_anchor():
    infer = _load_infer_module()

    class _Tokenizer:
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(
                input_ids=torch.tensor([[128000, 128006, 9125, 128007, 128009, 128006, 78191, 128007]])
            )

    pipe = types.SimpleNamespace(tokenizer=_Tokenizer())
    call_kwargs = {"prompt": "prompt"}

    status = infer.apply_hunyuan_i2v_prompt_template_compat(pipe, call_kwargs)

    assert status["override"] is True
    assert status["default_double_return_token_id"] == 271
    assert status["selected_double_return_token_id"] == 128007
    assert call_kwargs["prompt_template"]["double_return_token_id"] == 128007


def test_hunyuan_i2v_prompt_template_compat_keeps_default_anchor_when_present():
    infer = _load_infer_module()

    class _Tokenizer:
        def __call__(self, *args, **kwargs):
            return types.SimpleNamespace(input_ids=torch.tensor([[128000, 271, 128007]]))

    pipe = types.SimpleNamespace(tokenizer=_Tokenizer())
    call_kwargs = {"prompt": "prompt"}

    status = infer.apply_hunyuan_i2v_prompt_template_compat(pipe, call_kwargs)

    assert status["override"] is False
    assert status["selected_double_return_token_id"] == 271
    assert call_kwargs["prompt_template"]["double_return_token_id"] == 271


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
    assert f"output_file={output_file}" in result.stdout
    assert f"metrics_file={metrics_file}" in result.stdout
    assert "method_config" not in result.stdout
    assert metrics_file.exists()
    payload = json.loads(metrics_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    assert payload["runtime"]["preflight"]["errors"] == []
    assert "apply_sparse_attention_sec" in payload["timings"]
    assert "generate_sec" in payload["timings"]
    assert payload["seconds_per_frame"] > 0
    assert payload["output_file"] == str(output_file)
