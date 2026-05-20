#!/usr/bin/env python3
"""Audit SparseVideo parity evidence from current repo artifacts.

This script is intentionally conservative. It does not run inference and it
does not treat passing unit tests or old mp4 files as proof of upstream parity
unless the matching metrics also show strict kernel dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EXPECTED_METHODS = (
    "adacluster",
    "dense",
    "draft",
    "flashomni",
    "radial",
    "spargeattn",
    "sta",
    "svg1",
    "svg2",
    "svoo",
)
EXPECTED_SPARSE_METHODS = tuple(method for method in EXPECTED_METHODS if method != "dense")
EXPECTED_FULL_BACKBONE_METHODS = ("dense", *EXPECTED_SPARSE_METHODS)
EXPECTED_ALL_METHOD_BACKBONES = {
    "cogvideox-t2v",
    "cogvideox-i2v",
    "ltx-video",
    "ltx-video-i2v",
    "allegro",
    "mochi-1",
    "easyanimate-v5-t2v-12b",
}
EXPECTED_PUBLIC_API = (
    "apply",
    "apply_sparse_attention",
    "restore_sparse_attention",
    "SparseAttentionHandle",
    "default_method_config",
    "normalize_method_config",
    "list_methods",
)
EXPECTED_BACKBONE_MODEL_SPECS = {
    "wan21-t2v-1.3b": "wan",
    "wan21-t2v-14b": "wan",
    "wan22-t2v-a14b": "wan",
    "wan21-i2v-14b": "wan",
    "wan22-i2v-a14b": "wan",
    "wan22-animate-14b": "wan",
    "wan21-vace-1.3b": "wan",
    "wan21-vace-14b": "wan",
    "skyreels-v2-t2v-14b": "wan",
    "skyreels-v2-i2v-14b": "wan",
    "hunyuan-t2v": "hunyuan_video",
    "hunyuan-i2v": "hunyuan_video",
    "cogvideox-t2v": "cogvideox",
    "cogvideox-i2v": "cogvideox",
    "ltx-video": "ltx_video",
    "ltx-video-i2v": "ltx_video",
    "allegro": "allegro",
    "mochi-1": "mochi",
    "easyanimate-v5-t2v-12b": "easyanimate",
    "motif-video": "motif_video",
    "ltx-video-2": "ltx_video_2",
    "sana-video": "sana_video",
    "kandinsky5-t2v": "kandinsky5",
}
EXPECTED_BACKBONE_ALIASES = {
    "wananimate": "wan22-animate-14b",
    "wan-vace": "wan21-vace-1.3b",
    "cogvideox-i2v": "cogvideox-i2v",
    "ltx": "ltx-video",
    "ltx-i2v": "ltx-video-i2v",
    "allegro": "allegro",
    "mochi": "mochi-1",
    "easyanimate": "easyanimate-v5-t2v-12b",
    "motif-video": "motif-video",
    "ltx-video-2": "ltx-video-2",
    "sana-video": "sana-video",
    "kandinsky5": "kandinsky5-t2v",
}
EXPECTED_DEFERRED_LABELS = {
    "motif-video": "unknown",
    "ltx-video-2": "unknown",
    "sana-video": "incompatible",
    "kandinsky5-t2v": "native-N/A",
}
EXPECTED_PROCESSOR_CLASSES = {
    "cogvideox": "sparsevideo.processors.cogvideox.SparseCogVideoXAttnProcessor",
    "ltx_video": "sparsevideo.processors.ltx_video.SparseLTXVideoAttnProcessor",
    "allegro": "sparsevideo.processors.allegro.SparseAllegroAttnProcessor",
    "mochi": "sparsevideo.processors.mochi.SparseMochiAttnProcessor",
    "easyanimate": "sparsevideo.processors.easyanimate.SparseEasyAnimateAttnProcessor",
}
DEFAULT_METRICS_GLOB = "result/inference/**/*.jsonl"
DEFAULT_SMOKE_METRICS_GLOB = ".tmp_smoke/*.jsonl"
DEFAULT_METRICS_GLOBS = (DEFAULT_METRICS_GLOB, DEFAULT_SMOKE_METRICS_GLOB)
DEFAULT_MODEL_ROOT = Path("/home/dataset-assist-0/luojy/models")
TRAINING_FREE_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+training_free(?:\.|\s)|import\s+training_free(?:\.|\s|$))"
)
EXPECTED_BACKBONE_SMOKE_METHODS = {
    "wan21-t2v-1.3b": ("dense", "svg2", "svoo"),
    "wan21-t2v-14b": ("dense", "svg2", "svoo"),
    "wan22-t2v-a14b": ("dense", "svg2", "svoo"),
    "hunyuan-t2v": ("dense", "svg2", "svoo"),
    "wan21-i2v-14b": ("dense", "svg2", "svoo"),
    "wan22-i2v-a14b": ("dense", "svg2", "svoo"),
    "hunyuan-i2v": ("dense", "svg2", "svoo"),
    "skyreels-v2-t2v-14b": ("dense", "svg2", "svoo"),
    "skyreels-v2-i2v-14b": ("dense", "svg2", "svoo"),
    "cogvideox-t2v": EXPECTED_FULL_BACKBONE_METHODS,
    "cogvideox-i2v": EXPECTED_FULL_BACKBONE_METHODS,
    "ltx-video": EXPECTED_FULL_BACKBONE_METHODS,
    "ltx-video-i2v": EXPECTED_FULL_BACKBONE_METHODS,
    "allegro": EXPECTED_FULL_BACKBONE_METHODS,
    "mochi-1": EXPECTED_FULL_BACKBONE_METHODS,
    "easyanimate-v5-t2v-12b": EXPECTED_FULL_BACKBONE_METHODS,
}
FLASHOMNI_PUBLIC_SOURCE_STATUS = {
    "checked_on": "2026-05-18",
    "author_project_page": "https://qiaolian9.github.io/",
    "public_code_url": "https://github.com/qiaolian9/FlashOmni",
    "openreview_url": "https://openreview.net/forum?id=HljnvKxGRo",
    "anonymous_repo_url": "https://anonymous.4open.science/r/FlashOmni-B980",
    "anonymous_repo_api_status": (
        "API endpoints /api/repo/FlashOmni-B980/files and /file/<path> were "
        "reachable on 2026-05-18 and exposed example/hunyuan source files"
    ),
    "anonymous_hunyuan_policy_files": [
        "example/hunyuan/flashomni_hunyuan.py",
        "example/hunyuan/models/flashomni_attn_processor/attention_processor.py",
        "example/hunyuan/models/cache_functions/cache_init.py",
        "example/hunyuan/models/cache_functions/cal_type.py",
        "example/hunyuan/models/cache_functions/force_scheduler.py",
        "example/hunyuan/models/forwards/hunyuan_forward.py",
        "example/hunyuan/models/forwards/double_transformer_forward.py",
        "example/hunyuan/models/forwards/single_transformer_forward.py",
        "example/hunyuan/models/taylorseer_utils/__init__.py",
    ],
    "github_issues_status": (
        "repository homepage shows one issue, but /issues and the GitHub issues "
        "API returned 404 without accessible issue content on 2026-05-18"
    ),
    "openreview_rebuttal_referenced_paths": [
        "FlashOmni/example/flux/nvprof",
        "FlashOmni/example/hunyuan/nvprof",
        "FlashOmni/example/hunyuan/nvprof/e2e",
        "FlashOmni/benchmark/nvprof_attn/flux",
        "FlashOmni/benchmark/nvprof_attn/hunyuan",
        "FlashOmni/benchmark/nvprof_gemmq",
        "FlashOmni/benchmark/nvprof_gemmo",
    ],
    "openreview_rebuttal_missing_paths": (
        "OpenReview rebuttal comments reference example/hunyuan and nvprof "
        "artifact paths; these are absent from public GitHub/training_free but "
        "present in the anonymous artifact API"
    ),
    "openreview_revision_pdf_status": (
        "OpenReview revised PDF text was checked on 2026-05-18; it discusses HunyuanVideo "
        "experiments and a qualitative sparsity schedule but does not itself expose source code"
    ),
    "openreview_revision_threshold_schedule_status": (
        "OpenReview revised PDF says sparsity stays low in early denoising and gradually "
        "increases later, and repeats that tau_q/tau_kv progressively reach target values; "
        "it still provides no convergence schedule formula or implementation"
    ),
    "arxiv_source_status": (
        "arXiv e-print 2509.25401 contains paper TeX, figures, table text, and algorithms only; "
        "no code/example directory or Wan/Hunyuan video sparse-symbol policy source is included"
    ),
    "arxiv_threshold_schedule_status": (
        "sections/Appendix.tex states tau_q and tau_kv progressively converge to target values, "
        "but the e-print source contains no convergence schedule formula or implementation"
    ),
    "observed_public_dirs": [
        "3rdparty",
        "aot_build_utils",
        "benchmark",
        "csrc",
        "example/hunyuan",
        "flashomni",
        "include/flashomni",
    ],
    "observed_public_files": [
        "README.md",
        "custom_backend.py",
        "pyproject.toml",
        "setup.py",
    ],
    "completion_note": (
        "no remaining FlashOmni software gap in the current A100 audit: anonymous "
        "Hunyuan policy files are available, SparseVideo owns the Hunyuan sparse-symbol "
        "policy and transformer forward/Taylor-cache path, and reported-config "
        "50-step Hunyuan dispatch evidence exists; artifact visual acceptance remains separate"
    ),
}
FLASHOMNI_REPORTED_HUNYUAN_CONFIG = {
    "threshold_q": 0.5,
    "threshold_kv": 0.05,
    "fresh_threshold": 6,
    "max_order": 1,
    "saving_threshold_q_for_taylor": 0.3,
    "first_enhance": 8,
    "source": "anonymous FlashOmni Hunyuan cache_init.py plus paper HunyuanVideo row: (50%, 5%, 6, 1, 30%)",
}
FLASHOMNI_PAPER_THRESHOLD_SCHEDULE_STATUS = {
    "source": "anonymous FlashOmni example/hunyuan/models/flashomni_attn_processor/attention_processor.py",
    "observed_detail": (
        "Hunyuan flashomni_attn_score scales threshold_q and threshold_kv by "
        "(current_iter / 50) ** 1.7"
    ),
    "missing_detail": None,
    "current_sparsevideo_behavior": (
        "paper_mmdit Hunyuan sparse-symbol generation applies the anonymous "
        "FlashOmni threshold factor; complete Taylor-cache forward parity is still separate"
    ),
}


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _current_flashomni_policy_sha256() -> str | None:
    return _file_sha256(REPO_ROOT / "src" / "sparsevideo" / "methods" / "flashomni" / "policy.py")


def _current_flashomni_method_sha256() -> str | None:
    return _file_sha256(REPO_ROOT / "src" / "sparsevideo" / "methods" / "flashomni" / "method.py")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return records
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            records.append(
                {
                    "status": "invalid_json",
                    "source_file": str(path),
                    "source_line": line_no,
                }
            )
            continue
        if isinstance(record, dict):
            record.setdefault("source_file", str(path))
            record.setdefault("source_line", line_no)
            records.append(record)
    return records


def _collect_metrics(paths: list[Path], patterns: list[str]) -> list[dict[str, Any]]:
    files: list[Path] = []
    files.extend(path for path in paths if path.exists())
    for pattern in patterns:
        files.extend(REPO_ROOT.glob(pattern))
    unique_files = sorted({path.resolve() for path in files if path.is_file()})
    records: list[dict[str, Any]] = []
    for path in unique_files:
        records.extend(_read_jsonl(path))
    return records


def _handle(record: dict[str, Any]) -> dict[str, Any]:
    handle = record.get("sparse_attention_handle")
    return handle if isinstance(handle, dict) else {}


def _runtime_counts(record: dict[str, Any]) -> dict[str, Any]:
    handle_runtime = _handle(record).get("method_runtime")
    if isinstance(handle_runtime, dict):
        return handle_runtime
    runtime = record.get("method_runtime")
    return runtime if isinstance(runtime, dict) else {}


def _backend_counts(record: dict[str, Any]) -> dict[str, Any]:
    counts = _runtime_counts(record).get("backend_counts")
    return counts if isinstance(counts, dict) else {}


def _dispatch_counts(record: dict[str, Any]) -> dict[str, Any]:
    counts = _runtime_counts(record).get("dispatch_counts")
    return counts if isinstance(counts, dict) else {}


def _record_model_candidates(record: dict[str, Any], aliases: dict[str, str]) -> tuple[str, ...]:
    candidates: list[str] = []
    handle = _handle(record)
    for value in (handle.get("model_key"), record.get("model_arg"), record.get("model")):
        if not isinstance(value, str) or not value:
            continue
        candidates.append(value)
        alias = aliases.get(value)
        if alias:
            candidates.append(alias)
    return tuple(dict.fromkeys(candidates))


def _preflight_errors(record: dict[str, Any]) -> list[Any]:
    preflight = record.get("preflight")
    if not isinstance(preflight, dict):
        runtime = record.get("runtime")
        if isinstance(runtime, dict):
            preflight = runtime.get("preflight")
    if not isinstance(preflight, dict):
        return []
    errors = preflight.get("errors")
    return errors if isinstance(errors, list) else []


def _output_exists(record: dict[str, Any]) -> bool:
    path = _output_path(record)
    return path is not None and path.exists()


def _output_path(record: dict[str, Any]) -> Path | None:
    output = record.get("output_file")
    if not isinstance(output, str) or not output.endswith(".mp4"):
        return None
    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _video_artifact_qc(record: dict[str, Any]) -> dict[str, Any] | None:
    path = _output_path(record)
    if path is None:
        return None
    qc: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "warnings": [],
    }
    if not path.exists():
        qc["warnings"].append("output_file_missing")
        return qc

    size_bytes = path.stat().st_size
    qc["size_bytes"] = size_bytes
    height = int(record.get("height") or 0)
    width = int(record.get("width") or 0)
    frames = int(record.get("num_frames") or 0)
    if height >= 720 and width >= 1280 and frames >= 50 and size_bytes < 1_000_000:
        qc["warnings"].append("very_small_file_for_720p_video_quality_claim")

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,duration,avg_frame_rate,bit_rate",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        qc["warnings"].append(f"ffprobe_unavailable:{type(exc).__name__}")
        return qc
    if probe.returncode != 0:
        qc["warnings"].append("ffprobe_failed")
        if probe.stderr:
            qc["ffprobe_stderr"] = probe.stderr.strip()[:300]
        return qc
    try:
        data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        qc["warnings"].append("ffprobe_json_parse_failed")
        return qc
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        qc["warnings"].append("ffprobe_no_video_stream")
        return qc
    stream = streams[0]
    if isinstance(stream, dict):
        qc["ffprobe"] = {
            key: stream.get(key)
            for key in ("width", "height", "nb_frames", "duration", "avg_frame_rate", "bit_rate")
        }
        actual_width = _optional_int(stream.get("width"))
        actual_height = _optional_int(stream.get("height"))
        actual_frames = _stream_frame_count(stream)
        qc["actual_frames"] = actual_frames
        if width > 0 and actual_width is not None and actual_width != width:
            qc["warnings"].append(f"width_mismatch:{actual_width}!={width}")
        if height > 0 and actual_height is not None and actual_height != height:
            qc["warnings"].append(f"height_mismatch:{actual_height}!={height}")
        if frames > 0:
            if actual_frames is None:
                qc["warnings"].append("frame_count_unavailable")
            elif actual_frames < frames:
                qc["warnings"].append(f"frame_count_below_requested:{actual_frames}<{frames}")
        try:
            bit_rate = int(stream.get("bit_rate") or 0)
        except (TypeError, ValueError):
            bit_rate = 0
        if height >= 720 and width >= 1280 and frames >= 50 and 0 < bit_rate < 1_000_000:
            qc["warnings"].append("low_bitrate_for_720p_video_quality_claim")
    return qc


def _optional_int(value: Any) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value == "N/A":
        return None
    if "/" not in value:
        return _optional_float(value)
    numerator, denominator = value.split("/", 1)
    num = _optional_float(numerator)
    den = _optional_float(denominator)
    if num is None or den in (None, 0):
        return None
    return num / den


def _stream_frame_count(stream: dict[str, Any]) -> int | None:
    nb_frames = _optional_int(stream.get("nb_frames"))
    if nb_frames is not None:
        return nb_frames
    duration = _optional_float(stream.get("duration"))
    rate = _frame_rate(stream.get("avg_frame_rate"))
    if duration is None or rate is None:
        return None
    return int(round(duration * rate))


def _quality_artifact_ok(record: dict[str, Any]) -> bool:
    qc = _video_artifact_qc(record)
    if qc is None or not qc.get("exists"):
        return False
    warnings = qc.get("warnings")
    if not isinstance(warnings, list):
        return False
    non_blocking = {
        "very_small_file_for_720p_video_quality_claim",
        "low_bitrate_for_720p_video_quality_claim",
    }
    blocking = [warning for warning in warnings if warning not in non_blocking]
    return "ffprobe" in qc and not blocking


def _is_strict(record: dict[str, Any]) -> bool:
    return bool(record.get("strict_kernels")) and not bool(record.get("allow_debug_fallbacks"))


def _is_debug_or_nonparity(record: dict[str, Any]) -> bool:
    if record.get("allow_debug_fallbacks"):
        return True
    backend_counts = _backend_counts(record)
    if any("debug_fallback" in str(name) for name in backend_counts):
        return True
    config = record.get("method_config")
    if not isinstance(config, dict):
        return False
    if config.get("allow_triton_fallback") or config.get("allow_flex_fallback"):
        return True
    if record.get("method") == "flashomni":
        return config.get("sparse_pattern") in {"global_random", "local_qk_topk"}
    return False


def _is_ok(record: dict[str, Any]) -> bool:
    return record.get("status") == "ok" and not _preflight_errors(record)


def _has_strict_sparse_dispatch(record: dict[str, Any]) -> bool:
    backend_counts = _backend_counts(record)
    dispatch_counts = _dispatch_counts(record)
    return (
        _is_ok(record)
        and _is_strict(record)
        and bool(backend_counts)
        and not _is_debug_or_nonparity(record)
        and (record.get("method") == "dense" or int(dispatch_counts.get("sparse", 0) or 0) > 0)
    )


def _has_sparse_smoke_dispatch(record: dict[str, Any]) -> bool:
    backend_counts = _backend_counts(record)
    dispatch_counts = _dispatch_counts(record)
    return bool(backend_counts) and int(dispatch_counts.get("sparse", 0) or 0) > 0


def _has_backbone_smoke(record: dict[str, Any], method: str) -> bool:
    if record.get("method") != method:
        return False
    if not _is_ok(record):
        return False
    if int(record.get("num_inference_steps") or 0) < 1:
        return False
    if method == "dense":
        return True
    return _has_sparse_smoke_dispatch(record)


def _has_quality_output(record: dict[str, Any], min_steps: int) -> bool:
    return (
        _is_ok(record)
        and not record.get("skip_decode")
        and int(record.get("num_inference_steps") or 0) >= min_steps
        and _output_exists(record)
        and _quality_artifact_ok(record)
    )


def _summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    timings = record.get("timings")
    if not isinstance(timings, dict):
        timings = {}
    return {
        "source": f"{record.get('source_file')}:{record.get('source_line')}",
        "model": record.get("model_arg") or record.get("model"),
        "steps": record.get("num_inference_steps"),
        "shape": [
            record.get("height"),
            record.get("width"),
            record.get("num_frames"),
        ],
        "status": record.get("status"),
        "output_file": record.get("output_file"),
        "strict_kernels": record.get("strict_kernels"),
        "allow_debug_fallbacks": record.get("allow_debug_fallbacks"),
        "generate_sec": timings.get("generate_sec"),
        "total_sec": timings.get("total_sec"),
        "seconds_per_frame": record.get("seconds_per_frame"),
        "cuda_peak_allocated_gb": record.get("cuda_peak_allocated_gb"),
        "artifact_qc": _video_artifact_qc(record),
        "backend_counts": _backend_counts(record),
        "dispatch_counts": _dispatch_counts(record),
        "method_config": record.get("method_config"),
    }


def _same_benchmark_shape(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("model_arg", "num_inference_steps", "height", "width", "num_frames", "skip_decode")
    return all(left.get(key) == right.get(key) for key in keys)


def _sta_speed_comparison(
    baseline: dict[str, Any] | None,
    optimized: dict[str, Any] | None,
) -> dict[str, Any]:
    if baseline is None or optimized is None:
        return {
            "status": "missing",
            "baseline": _summarize_record(baseline) if baseline else None,
            "optimized": _summarize_record(optimized) if optimized else None,
            "note": "Need matching fastvideo_sta_triton and fastvideo_sta_a100_triton records",
        }
    if not _same_benchmark_shape(baseline, optimized):
        return {
            "status": "mismatched",
            "baseline": _summarize_record(baseline),
            "optimized": _summarize_record(optimized),
            "note": "STA A100 speed comparison requires matching model, steps, shape, and skip_decode",
        }
    baseline_generate = ((baseline.get("timings") or {}).get("generate_sec"))
    optimized_generate = ((optimized.get("timings") or {}).get("generate_sec"))
    if not baseline_generate or not optimized_generate:
        return {
            "status": "missing",
            "baseline": _summarize_record(baseline),
            "optimized": _summarize_record(optimized),
            "note": "STA A100 speed comparison requires generate_sec timings",
        }
    speedup = float(baseline_generate) / float(optimized_generate)
    return {
        "status": "pass",
        "baseline": _summarize_record(baseline),
        "optimized": _summarize_record(optimized),
        "generate_sec_speedup": speedup,
        "generate_sec_delta": float(baseline_generate) - float(optimized_generate),
    }


def _best_record(records: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    matches = [record for record in records if predicate(record)]
    if not matches:
        return None
    return max(matches, key=lambda item: int(item.get("num_inference_steps") or 0))


def _scan_training_free_imports() -> list[str]:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "sparsevideo").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if TRAINING_FREE_IMPORT_RE.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}:{line.strip()}")
    return offenders


def _public_methods_gate() -> dict[str, Any]:
    try:
        import sparsevideo

        methods = tuple(sparsevideo.list_methods())
    except Exception as exc:  # pragma: no cover - defensive audit output
        return {
            "gate": "public_method_registry",
            "status": "fail",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
            "missing": ["import sparsevideo and sparsevideo.list_methods() must work"],
        }
    missing = [method for method in EXPECTED_METHODS if method not in methods]
    extra = [method for method in methods if method not in EXPECTED_METHODS]
    status = "pass" if not missing and not extra else "fail"
    return {
        "gate": "public_method_registry",
        "status": status,
        "evidence": {"methods": list(methods)},
        "missing": [f"missing={missing}", f"extra={extra}"] if status == "fail" else [],
    }


def _public_api_and_config_gate() -> dict[str, Any]:
    try:
        import sparsevideo
    except Exception as exc:  # pragma: no cover - defensive audit output
        return {
            "gate": "public_api_and_config_contract",
            "status": "fail",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
            "missing": ["import sparsevideo must work without optional native kernels"],
        }

    missing_api = [name for name in EXPECTED_PUBLIC_API if not hasattr(sparsevideo, name)]
    config_errors: dict[str, str] = {}
    configs: dict[str, list[str]] = {}
    for method in EXPECTED_METHODS:
        if method == "dense":
            continue
        try:
            config = sparsevideo.default_method_config(method)
        except Exception as exc:
            config_errors[method] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(config, dict):
            config_errors[method] = f"default_method_config returned {type(config).__name__}, expected dict"
            continue
        configs[method] = sorted(str(key) for key in config)

    missing = []
    if missing_api:
        missing.append(f"missing public API names: {missing_api}")
    if config_errors:
        missing.append(f"default_method_config errors: {config_errors}")
    return {
        "gate": "public_api_and_config_contract",
        "status": "pass" if not missing else "fail",
        "evidence": {
            "public_api": [name for name in EXPECTED_PUBLIC_API if hasattr(sparsevideo, name)],
            "config_keys": configs,
        },
        "missing": missing,
    }


def _load_infer_script_module():
    module_name = "sparsevideo_audit_infer_script"
    script = REPO_ROOT / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _processor_class_import_status() -> dict[str, str]:
    status: dict[str, str] = {}
    for family, dotted_path in EXPECTED_PROCESSOR_CLASSES.items():
        module_name, class_name = dotted_path.rsplit(".", 1)
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - defensive audit output
            status[family] = f"missing:{type(exc).__name__}: {exc}"
            continue
        status[family] = f"{cls.__module__}.{cls.__name__}"
    return status


def _all_backbone_support_contract_gate() -> dict[str, Any]:
    try:
        infer = _load_infer_script_module()
        from sparsevideo._support import LIMITED_METHODS_BY_MODEL_TYPE
    except Exception as exc:  # pragma: no cover - defensive audit output
        return {
            "gate": "all_backbone_support_contract",
            "status": "fail",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
            "missing": ["scripts/infer.py and sparsevideo._support must be importable"],
        }

    missing: list[str] = []
    model_evidence: dict[str, dict[str, Any]] = {}
    for model_key, expected_family in EXPECTED_BACKBONE_MODEL_SPECS.items():
        spec = infer.MODEL_SPECS.get(model_key)
        if spec is None:
            missing.append(f"missing MODEL_SPECS[{model_key!r}]")
            continue
        sparse_methods = spec.sparse_methods
        model_evidence[model_key] = {
            "family": spec.family,
            "pipeline_class": spec.pipeline_class,
            "sparse_supported": spec.sparse_supported,
            "sparse_methods": list(sparse_methods) if sparse_methods is not None else None,
            "compatibility_label": spec.compatibility_label,
        }
        if spec.family != expected_family:
            missing.append(f"{model_key}: family={spec.family}, expected={expected_family}")
        if model_key in EXPECTED_ALL_METHOD_BACKBONES:
            if not spec.sparse_supported:
                missing.append(f"{model_key}: must be sparse_supported for all public sparse methods")
            if sparse_methods is not None:
                missing.append(
                    f"{model_key}: sparse_methods={sparse_methods}, expected=None for all public sparse methods"
                )
            if spec.family in LIMITED_METHODS_BY_MODEL_TYPE:
                missing.append(
                    f"{spec.family}: still listed in LIMITED_METHODS_BY_MODEL_TYPE="
                    f"{LIMITED_METHODS_BY_MODEL_TYPE[spec.family]}"
                )
        else:
            expected_limited = LIMITED_METHODS_BY_MODEL_TYPE.get(spec.family)
            if expected_limited is not None and sparse_methods != expected_limited:
                missing.append(
                    f"{model_key}: sparse_methods={sparse_methods}, expected={expected_limited}"
                )
        expected_label = EXPECTED_DEFERRED_LABELS.get(model_key)
        if expected_label is not None and spec.compatibility_label != expected_label:
            missing.append(
                f"{model_key}: compatibility_label={spec.compatibility_label}, expected={expected_label}"
            )
        if expected_label is not None and spec.sparse_supported:
            missing.append(f"{model_key}: deferred/non-swap backbone must not be sparse_supported")

    alias_evidence = {}
    for alias, expected_key in EXPECTED_BACKBONE_ALIASES.items():
        actual = infer.MODEL_ALIASES.get(alias)
        alias_evidence[alias] = actual
        if actual != expected_key:
            missing.append(f"alias {alias!r}: {actual!r}, expected {expected_key!r}")

    processor_status = _processor_class_import_status()
    for family, dotted_path in EXPECTED_PROCESSOR_CLASSES.items():
        if processor_status.get(family) != dotted_path:
            missing.append(f"{family}: processor import status={processor_status.get(family)!r}")

    return {
        "gate": "all_backbone_support_contract",
        "status": "pass" if not missing else "fail",
        "evidence": {
            "models": model_evidence,
            "aliases": alias_evidence,
            "limited_methods_by_model_type": {
                family: list(methods) for family, methods in LIMITED_METHODS_BY_MODEL_TYPE.items()
            },
            "processor_classes": processor_status,
        },
        "missing": missing,
    }


def _all_backbone_smoke_evidence_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        infer = _load_infer_script_module()
    except Exception as exc:  # pragma: no cover - defensive audit output
        return {
            "gate": "all_backbone_smoke_evidence",
            "status": "fail",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
            "missing": ["scripts/infer.py must be importable to normalize model aliases"],
        }

    missing: list[str] = []
    evidence: dict[str, Any] = {}
    for model_key, methods in EXPECTED_BACKBONE_SMOKE_METHODS.items():
        model_records = [
            record
            for record in records
            if model_key in _record_model_candidates(record, infer.MODEL_ALIASES)
        ]
        method_evidence: dict[str, Any] = {}
        for method in methods:
            record = _best_record(
                model_records,
                lambda item, expected_method=method: _has_backbone_smoke(
                    item,
                    expected_method,
                ),
            )
            method_evidence[method] = _summarize_record(record) if record else None
            if record is None:
                missing.append(f"{model_key}/{method}: needs >=1-step status=ok smoke record")
        evidence[model_key] = {
            "required_methods": list(methods),
            "records_seen": len(model_records),
            "methods": method_evidence,
        }

    return {
        "gate": "all_backbone_smoke_evidence",
        "status": "pass" if not missing else "fail",
        "evidence": {
            "note": (
                "Counts real status=ok execution records only; dry_run/failed records "
                "do not count. skip_decode latent runs count as smoke, not quality parity."
            ),
            "models": evidence,
        },
        "missing": missing,
    }


def _checkpoint_files(path: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt", ".ckpt", ".pth"}
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix in suffixes)


def _checkpoint_index_refs(path: Path) -> tuple[list[Path], list[str]]:
    refs: list[Path] = []
    errors: list[str] = []
    for index_path in sorted(path.rglob("*.index.json")):
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{index_path.relative_to(path)}: {type(exc).__name__}: {exc}")
            continue
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict):
            continue
        for rel_path in set(weight_map.values()):
            if isinstance(rel_path, str):
                refs.append(index_path.parent / rel_path)
    return sorted(set(refs)), errors


def _checkpoint_availability(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing_dir",
            "path": str(path),
            "index_ref_count": 0,
            "missing_index_refs": [],
            "checkpoint_file_count": 0,
            "partial_file_count": 0,
            "index_errors": [],
        }

    refs, errors = _checkpoint_index_refs(path)
    missing_refs = [ref for ref in refs if not ref.exists()]
    ckpt_files = _checkpoint_files(path)
    partial_files = [
        item
        for pattern in ("*.incomplete", "*.lock", "*.part")
        for item in path.rglob(pattern)
        if item.is_file()
    ]
    if errors:
        status = "index_error"
    elif refs:
        status = "complete" if not missing_refs else "missing_index_refs"
    else:
        status = "complete" if ckpt_files else "no_checkpoint_files"

    return {
        "status": status,
        "path": str(path),
        "index_ref_count": len(refs),
        "missing_index_refs": [str(ref.relative_to(path)) for ref in missing_refs],
        "checkpoint_file_count": len(ckpt_files),
        "partial_file_count": len(partial_files),
        "index_errors": errors,
    }


def _all_backbone_checkpoint_availability_gate(
    model_root: Path = DEFAULT_MODEL_ROOT,
) -> dict[str, Any]:
    try:
        infer = _load_infer_script_module()
    except Exception as exc:  # pragma: no cover - defensive audit output
        return {
            "gate": "all_backbone_checkpoint_availability",
            "status": "fail",
            "evidence": {"error": f"{type(exc).__name__}: {exc}"},
            "missing": ["scripts/infer.py must be importable to inspect model local_dir values"],
        }

    missing: list[str] = []
    evidence: dict[str, Any] = {}
    for model_key in EXPECTED_BACKBONE_SMOKE_METHODS:
        spec = infer.MODEL_SPECS.get(model_key)
        if spec is None or spec.local_dir is None:
            missing.append(f"{model_key}: no MODEL_SPECS local_dir")
            continue
        availability = _checkpoint_availability(model_root / spec.local_dir)
        evidence[model_key] = availability
        if availability["status"] == "complete":
            continue
        if availability["status"] == "missing_index_refs":
            missing.append(
                f"{model_key}: missing {len(availability['missing_index_refs'])}/"
                f"{availability['index_ref_count']} indexed checkpoint files"
            )
        else:
            missing.append(f"{model_key}: checkpoint status={availability['status']}")

    return {
        "gate": "all_backbone_checkpoint_availability",
        "status": "pass" if not missing else "fail",
        "evidence": {
            "model_root": str(model_root),
            "models": evidence,
        },
        "missing": missing,
    }


def _runtime_ownership_gate() -> dict[str, Any]:
    offenders = _scan_training_free_imports()
    return {
        "gate": "no_direct_training_free_runtime_imports",
        "status": "pass" if not offenders else "fail",
        "evidence": {"direct_imports": offenders},
        "missing": [] if not offenders else ["remove direct training_free runtime imports from src/sparsevideo"],
    }


def _flashomni_reference_policy_evidence() -> dict[str, Any]:
    root = REPO_ROOT / "training_free" / "FlashOmni"
    if not root.exists():
        return {
            "reference_root": str(root),
            "exists": False,
            "video_policy_candidates": [],
            "benchmark_sparse_helpers": [],
        }

    video_policy_re = re.compile(r"\b(?:hunyuan|wan|cogvideo|mochi)\b", re.IGNORECASE)
    helper_terms = ("get_qkvo_global_sparse", "sparse_info", "sparse_kv_info")
    global_random_terms = ("get_qkvo_global_sparse", "spq_Q", "spq_KV")
    score_sparse_terms = (
        "fill_flashomni_triton",
        "fill_sparse_info_triton",
        "fill_sparse_kv_info_triton",
        "pooled_score",
        "searchsorted",
    )
    candidates: list[str] = []
    helpers: list[str] = []
    global_random_helpers: list[str] = []
    score_sparse_helpers: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md"}:
            continue
        if any(part in {"3rdparty", "build", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        lowered = text.lower()
        if video_policy_re.search(text):
            candidates.append(rel)
        if any(term in text for term in helper_terms):
            helpers.append(rel)
        if any(term in text for term in global_random_terms):
            global_random_helpers.append(rel)
        if any(term in text for term in score_sparse_terms):
            score_sparse_helpers.append(rel)
    policy_candidates = [
        item for item in candidates
        if item not in {"README.md"} and not item.startswith("benchmark/")
    ]
    git_history = _flashomni_git_history_evidence(root)
    return {
        "reference_root": str(root),
        "exists": True,
        "video_policy_candidates": policy_candidates,
        "git_history": git_history,
        "benchmark_sparse_helpers": helpers,
        "benchmark_global_random_helpers": global_random_helpers,
        "benchmark_score_sparse_helpers": score_sparse_helpers,
        "public_source_status": FLASHOMNI_PUBLIC_SOURCE_STATUS,
        "note": (
            "benchmark helpers, including global-random and score-CDF sparse-info "
            "fill helpers, prove sparse-info tensor mechanics only; they are not "
            "Wan/Hunyuan video sparse-info policy evidence"
        ),
    }


def _flashomni_git_history_evidence(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {
            "available": False,
            "reason": "not a git checkout",
            "head": None,
            "branches": [],
            "tags": [],
            "policy_candidates": [],
        }

    def git(args: list[str]) -> list[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    history_names = git(["log", "--all", "--name-only", "--pretty=format:"])
    history_re = re.compile(
        r"(?:wan|hunyuan|policy|infer|pipeline|video|sparse_symbol|sparse_info)",
        re.IGNORECASE,
    )
    return {
        "available": True,
        "head": next(iter(git(["rev-parse", "HEAD"])), None),
        "branches": git(["branch", "-a", "--format=%(refname:short)"]),
        "tags": git(["tag", "-l"]),
        "policy_candidates": sorted({name for name in history_names if history_re.search(name)}),
    }


def _flashomni_method_path_evidence() -> dict[str, Any]:
    method_root = REPO_ROOT / "src" / "sparsevideo" / "methods" / "flashomni"
    processor_roots = [
        REPO_ROOT / "src" / "sparsevideo" / "processors" / "wan.py",
        REPO_ROOT / "src" / "sparsevideo" / "processors" / "hunyuan_video.py",
    ]
    native_root = REPO_ROOT / "src" / "sparsevideo" / "kernels" / "native" / "flashomni"
    native_gemm_files = [
        native_root / "flashomni" / "gemm.py",
        native_root / "csrc" / "gemm.cu",
        native_root / "csrc" / "gemm_reduction.cu",
        native_root / "csrc" / "flashomni_gemm_ops.cu",
    ]
    paper_policy_source = method_root / "policy.py"
    paper_policy_text = ""
    if paper_policy_source.exists():
        try:
            paper_policy_text = paper_policy_source.read_text(errors="replace")
        except OSError:
            paper_policy_text = ""

    runtime_files = [
        path
        for path in sorted(method_root.rglob("*.py"))
        if path.is_file()
    ] + [path for path in processor_roots if path.exists()]

    runtime_hits: list[str] = []
    update_dispatch_hits: list[str] = []
    for path in runtime_files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if "flashomni_gemm" in text or "flashomni_gemm_reduction" in text:
            runtime_hits.append(str(path.relative_to(REPO_ROOT)))
        if (
            "_FlashOmniPaperMMDiTState" in text
            and "_flashomni_paper_mmdit_schedule" in text
            and "_flashomni_apply_cached_q_blocks" in text
        ):
            update_dispatch_hits.append(str(path.relative_to(REPO_ROOT)))

    return {
        "owned_sparse_gemm_runtime": all(path.exists() for path in native_gemm_files),
        "owned_sparse_gemm_files": [
            str(path.relative_to(REPO_ROOT)) for path in native_gemm_files if path.exists()
        ],
        "method_uses_sparse_gemm": bool(runtime_hits),
        "method_sparse_gemm_hits": runtime_hits,
        "method_uses_update_dispatch_cache": bool(update_dispatch_hits),
        "method_update_dispatch_hits": update_dispatch_hits,
        "owned_paper_policy_source": paper_policy_source.exists(),
        "owned_paper_policy_path": (
            str(paper_policy_source.relative_to(REPO_ROOT)) if paper_policy_source.exists() else None
        ),
        "current_policy_sha256": _current_flashomni_policy_sha256(),
        "current_method_sha256": _current_flashomni_method_sha256(),
        "owned_score_cdf_policy": all(
            term in paper_policy_text
            for term in (
                "benchmark/test_attn_score.py",
                "vision-to-text contribution",
                "text-to-vision guidance",
                "_apply_score_cdf_feature_cache_symbols",
                "_apply_score_cdf_kv_symbols",
            )
        ),
        "owned_hunyuan_video_policy": all(
            term in paper_policy_text
            for term in (
                "flashomni_hunyuan_sparse_blocks",
                "threshold_q",
                "threshold_kv",
                "current_iter",
                "simthreshd1",
                "_flashomni_hunyuan_fill_q_sparse_info",
            )
        ),
        "owned_hunyuan_transformer_forward_taylor_cache": any(
            all(term in path.read_text(errors="replace") for term in ("flashomni_hunyuan_forward", "cal_type", "taylor_formula"))
            for path in runtime_files
            if path.is_file()
        ),
        "note": (
            "FlashOmni paper parity requires sparse-symbol update/dispatch plus "
            "GEMM-Q/GEMM-O method integration; the current method path is an "
            "attention-kernel adapter if no update-dispatch and GEMM runtime "
            "hits are listed"
        ),
    }


def _sta_hardware_evidence() -> dict[str, Any]:
    status: dict[str, Any] = {
        "torch_imported": False,
        "cuda_available": False,
        "devices": [],
        "hopper_visible": False,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive audit output
        status["torch_import_error"] = f"{type(exc).__name__}: {exc}"
        return status

    status["torch_imported"] = True
    status["cuda_available"] = bool(torch.cuda.is_available())
    if not status["cuda_available"]:
        return status
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        device = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(capability),
        }
        status["devices"].append(device)
        if capability[0] >= 9:
            status["hopper_visible"] = True
    return status


def _has_flashomni_paper_policy_dispatch(record: dict[str, Any], min_steps: int) -> bool:
    config = record.get("method_config")
    if not isinstance(config, dict) or config.get("sparse_pattern") != "paper_mmdit":
        return False
    fingerprints = record.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    if fingerprints.get("flashomni_policy_sha256") != _current_flashomni_policy_sha256():
        return False
    if fingerprints.get("flashomni_method_sha256") != _current_flashomni_method_sha256():
        return False
    backend_counts = _backend_counts(record)
    return (
        _has_quality_output(record, min_steps)
        and _is_strict(record)
        and not _preflight_errors(record)
        and _is_flashomni_hunyuan_record(record)
        and _has_flashomni_hunyuan_target_shape(record)
        and _flashomni_reported_hunyuan_config_match(record)
        and int(backend_counts.get("flashomni_explicit_upstream", 0) or 0) > 0
        and int(backend_counts.get("flashomni_full_upstream", 0) or 0) > 0
    )


def _is_flashomni_hunyuan_record(record: dict[str, Any]) -> bool:
    handle = _handle(record)
    fields = (
        handle.get("model_key"),
        handle.get("model_type"),
        record.get("model_arg"),
        record.get("model"),
    )
    return any(isinstance(value, str) and "hunyuan" in value.lower() for value in fields)


def _has_flashomni_hunyuan_target_shape(record: dict[str, Any]) -> bool:
    return (
        int(record.get("height") or 0) >= 720
        and int(record.get("width") or 0) >= 1280
        and int(record.get("num_frames") or 0) >= 129
    )


def _flashomni_reported_hunyuan_config_match(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    config = record.get("method_config")
    if not isinstance(config, dict):
        return False
    legacy_aliases = {
        "threshold_q": "tau_q",
        "threshold_kv": "tau_kv",
        "fresh_threshold": "N",
        "max_order": "D",
        "saving_threshold_q_for_taylor": "S_q",
    }
    for key, expected in FLASHOMNI_REPORTED_HUNYUAN_CONFIG.items():
        if key == "source":
            continue
        actual = config.get(key, config.get(legacy_aliases.get(key)))
        if isinstance(expected, float):
            try:
                if abs(float(actual) - expected) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def _flashomni_goal_checklist(
    reference_policy: dict[str, Any],
    method_path: dict[str, Any],
    paper_policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    public_status = reference_policy.get("public_source_status", {})
    upstream_policy_available = bool(
        reference_policy.get("video_policy_candidates")
        or public_status.get("anonymous_hunyuan_policy_files")
    )
    owned_policy_available = bool(method_path.get("owned_score_cdf_policy"))
    owned_hunyuan_policy_available = bool(method_path.get("owned_hunyuan_video_policy"))
    owned_full_hunyuan_method_path = bool(method_path.get("owned_hunyuan_transformer_forward_taylor_cache"))
    paper_policy_runtime = _summarize_record(paper_policy) if paper_policy else None
    paper_policy_matches_reported_hunyuan = _flashomni_reported_hunyuan_config_match(paper_policy)
    paper_runtime_caveat = None
    if paper_policy and not paper_policy_matches_reported_hunyuan:
        paper_runtime_caveat = (
            "strict-dispatch run does not use the paper's reported HunyuanVideo "
            "(50%, 5%, 6, 1, 30%) config"
        )
    elif paper_policy_runtime:
        artifact_qc = paper_policy_runtime.get("artifact_qc")
        warnings = artifact_qc.get("warnings") if isinstance(artifact_qc, dict) else None
        if warnings:
            paper_runtime_caveat = (
                "runtime/dispatch evidence only; artifact QC warnings require separate "
                f"visual quality acceptance: {warnings}"
            )

    return [
        {
            "requirement": "src/sparsevideo-owned FlashOmni Hunyuan sparse-symbol policy",
            "status": "pass" if owned_hunyuan_policy_available else "missing",
            "evidence": method_path.get("owned_paper_policy_path"),
            "caveat": None,
        },
        {
            "requirement": "tests align FlashOmni policy behavior against available public benchmark/anonymous helpers",
            "status": "pass" if owned_policy_available and owned_hunyuan_policy_available else "missing",
            "evidence": [
                "tests/test_flashomni_parity.py::test_flashomni_hunyuan_sparse_blocks_follow_anonymous_policy_shapes_and_schedule_factor",
                "tests/test_flashomni_parity.py::test_flashomni_paper_mmdit_schedule_matches_hunyuan_first_enhance_and_refresh",
                "tests/test_flashomni_parity.py::test_flashomni_paper_sparse_blocks_match_paper_contribution_guidance_prefix_logic",
                "tests/test_flashomni_parity.py::test_flashomni_paper_sparse_blocks_match_score_cdf_tail_logic_with_trimmed_kv_text",
                "tests/test_parity_audit.py::test_flashomni_paper_policy_record_counts_as_runtime_but_not_code_parity",
            ],
            "caveat": None if upstream_policy_available else "anonymous Hunyuan source evidence is missing",
        },
        {
            "requirement": (
                "Hunyuan 720p/129-frame/50-step real video inference using the reported "
                "Hunyuan config with flashomni_explicit_upstream dispatch"
            ),
            "status": "pass" if paper_policy else "missing",
            "evidence": paper_policy_runtime,
            "caveat": paper_runtime_caveat,
        },
        {
            "requirement": "code-level upstream Wan/Hunyuan video sparse-symbol policy evidence",
            "status": "pass" if upstream_policy_available else "missing",
            "evidence": reference_policy.get("video_policy_candidates", []) or public_status.get("anonymous_hunyuan_policy_files", []),
            "caveat": None if upstream_policy_available else public_status.get("completion_note"),
        },
        {
            "requirement": "complete Hunyuan transformer forward/Taylor-cache method path parity",
            "status": "pass" if owned_full_hunyuan_method_path else "missing",
            "evidence": method_path.get("method_update_dispatch_hits", []),
            "caveat": (
                None
                if owned_full_hunyuan_method_path
                else "anonymous Hunyuan wraps transformer forward, block forwards, cal_type, and taylor_formula; SparseVideo currently ports attention sparse-info/update dispatch only"
            ),
        },
    ]


def _method_audit(method: str, records: list[dict[str, Any]], min_steps: int) -> dict[str, Any]:
    method_records = [record for record in records if record.get("method") == method]
    strict_dispatch = _best_record(method_records, _has_strict_sparse_dispatch)
    quality = _best_record(method_records, lambda record: _has_quality_output(record, min_steps))
    quality_with_dispatch = _best_record(
        method_records,
        lambda record: _has_quality_output(record, min_steps) and _has_strict_sparse_dispatch(record),
    )

    missing: list[str] = []
    next_artifacts: list[str] = []
    deferred_artifacts: list[str] = []
    extra_evidence: dict[str, Any] = {}
    if method == "dense":
        if quality is None:
            missing.append(f"dense baseline needs a {min_steps}-step mp4 quality record")
    else:
        if strict_dispatch is None:
            missing.append("needs strict sparse native/backend dispatch evidence")
        if quality is None:
            missing.append(f"needs a {min_steps}-step mp4 quality record")
        if quality_with_dispatch is None:
            missing.append(
                f"needs one {min_steps}-step quality record that also records strict native/backend dispatch"
            )

    if method == "flashomni":
        reference_policy = _flashomni_reference_policy_evidence()
        method_path = _flashomni_method_path_evidence()
        extra_evidence["reference_policy"] = reference_policy
        extra_evidence["method_path"] = method_path
        explicit = _best_record(
            method_records,
            lambda record: (
                _has_quality_output(record, min_steps)
                and _has_strict_sparse_dispatch(record)
                and isinstance(record.get("method_config"), dict)
                and record["method_config"].get("sparse_pattern") == "explicit"
            ),
        )
        paper_policy = _best_record(
            method_records,
            lambda record: _has_flashomni_paper_policy_dispatch(record, min_steps),
        )
        extra_evidence["paper_policy_runtime"] = (
            _summarize_record(paper_policy) if paper_policy else None
        )
        extra_evidence["reported_hunyuan_config"] = FLASHOMNI_REPORTED_HUNYUAN_CONFIG
        extra_evidence["paper_threshold_schedule_status"] = FLASHOMNI_PAPER_THRESHOLD_SCHEDULE_STATUS
        extra_evidence["paper_policy_matches_reported_hunyuan_config"] = (
            _flashomni_reported_hunyuan_config_match(paper_policy)
        )
        extra_evidence["goal_checklist"] = _flashomni_goal_checklist(
            reference_policy,
            method_path,
            paper_policy,
        )
        public_status = reference_policy.get("public_source_status", {})
        upstream_policy_available = bool(
            reference_policy.get("video_policy_candidates")
            or public_status.get("anonymous_hunyuan_policy_files")
        )
        if explicit is None:
            missing.append(
                "FlashOmni needs explicit sparse_info/sparse_kv_info parity evidence; "
                "global_random is only a synthetic kernel benchmark"
            )
        if not upstream_policy_available:
            missing.append(
                "FlashOmni reference code has no reusable Wan/Hunyuan video sparse-info policy files"
            )
        if not method_path.get("owned_hunyuan_video_policy"):
            missing.append(
                "FlashOmni needs a SparseVideo-owned Hunyuan sparse-symbol policy matching "
                "anonymous example/hunyuan attention_processor.py"
            )
        if not method_path.get("owned_hunyuan_transformer_forward_taylor_cache"):
            missing.append(
                "FlashOmni still lacks the anonymous Hunyuan transformer forward/Taylor-cache "
                "method path; current SparseVideo path ports attention sparse-info/update dispatch only"
            )
        if paper_policy is None:
            missing.append(
                "FlashOmni paper/benchmark-derived policy needs real Hunyuan 720p/129-frame/50-step "
                "video inference with the reported Hunyuan config, flashomni_full_upstream update, "
                "and flashomni_explicit_upstream dispatch"
            )
        if not method_path.get("method_uses_update_dispatch_cache"):
            missing.append(
                "FlashOmni method path does not integrate sparse-symbol update/dispatch; "
                "current SparseVideo path replaces attention only"
            )
        if not method_path.get("method_uses_sparse_gemm"):
            missing.append(
                "FlashOmni method path does not integrate FlashOmni GEMM-Q/GEMM-O; "
                "current SparseVideo path still computes Q/O projections outside FlashOmni sparse GEMMs"
            )
        if not method_path.get("owned_hunyuan_video_policy"):
            next_artifacts.append(
                "SparseVideo-owned FlashOmni Hunyuan sparse-symbol policy source"
            )
        if not method_path.get("owned_hunyuan_transformer_forward_taylor_cache"):
            next_artifacts.append(
                "SparseVideo-owned FlashOmni Hunyuan transformer forward/Taylor-cache method path"
            )
        if not method_path.get("method_uses_update_dispatch_cache"):
            next_artifacts.append(
                "SparseVideo-owned FlashOmni update/dispatch method path with sparse-symbol state"
            )
        if not method_path.get("method_uses_sparse_gemm"):
            next_artifacts.append(
                "SparseVideo-owned FlashOmni GEMM-Q/GEMM-O integration in the method path"
            )
        if paper_policy is None:
            next_artifacts.append(
                "Hunyuan 720p/129-frame/"
                f"{min_steps}-step real video metrics with strict flashomni_full_upstream update "
                "and flashomni_explicit_upstream dispatch using the reported Hunyuan config"
            )

    if method == "sta":
        hardware = _sta_hardware_evidence()
        extra_evidence["hardware"] = hardware
        a100_triton = _best_record(
            method_records,
            lambda record: any(
                str(name).lower() in {"fastvideo_sta_a100_triton", "fastvideo_sta_triton"}
                for name in _backend_counts(record)
            ),
        )
        a100_quality = _best_record(
            method_records,
            lambda record: (
                _has_quality_output(record, min_steps)
                and any(
                    str(name).lower() in {"fastvideo_sta_a100_triton", "fastvideo_sta_triton"}
                    for name in _backend_counts(record)
                )
            ),
        )
        a100_named_quality = _best_record(
            method_records,
            lambda record: (
                _has_quality_output(record, min_steps)
                and any(
                    str(name).lower() == "fastvideo_sta_a100_triton"
                    for name in _backend_counts(record)
                )
            ),
        )
        legacy_triton_quality = _best_record(
            method_records,
            lambda record: (
                _has_quality_output(record, min_steps)
                and any(
                    str(name).lower() == "fastvideo_sta_triton"
                    for name in _backend_counts(record)
                )
            ),
        )
        a100_speed = _sta_speed_comparison(legacy_triton_quality, a100_named_quality)
        extra_evidence["a100_triton"] = {
            "strict_dispatch": _summarize_record(a100_triton) if a100_triton else None,
            "quality_with_dispatch": _summarize_record(a100_quality) if a100_quality else None,
            "status": "pass" if a100_triton and a100_quality else "missing",
        }
        extra_evidence["sta_a100"] = {
            "backend": "fastvideo_sta_a100_triton",
            "strict_dispatch": _summarize_record(a100_triton) if a100_triton else None,
            "quality_with_dispatch": _summarize_record(a100_quality) if a100_quality else None,
            "speed_comparison": a100_speed,
            "status": (
                "pass"
                if a100_triton and a100_quality and a100_speed["status"] == "pass"
                else "missing"
            ),
        }
        if a100_triton is None or a100_quality is None:
            missing.append(
                "STA A100 optimized Triton path needs strict dispatch and "
                f"{min_steps}-step real video quality evidence"
            )
            next_artifacts.append(
                f"{min_steps}-step A100 metrics showing fastvideo_sta_a100_triton backend dispatch"
            )
        if a100_speed["status"] != "pass":
            missing.append(
                "STA A100 optimized Triton path needs matching before/after speed evidence "
                "against the legacy FastVideo Triton fallback"
            )
            next_artifacts.append(
                f"{min_steps}-step matching STA metrics for fastvideo_sta_triton and "
                "fastvideo_sta_a100_triton"
            )
        h100 = _best_record(
            method_records,
            lambda record: any("h100" in str(name).lower() for name in _backend_counts(record)),
        )
        h100_quality = _best_record(
            method_records,
            lambda record: (
                _has_quality_output(record, min_steps)
                and any("h100" in str(name).lower() for name in _backend_counts(record))
            ),
        )
        h100_status = (
            "pass"
            if h100 and h100_quality
            else "deferred"
            if not hardware.get("hopper_visible")
            else "missing"
        )
        extra_evidence["sta_h100"] = {
            "backend": "fastvideo_sta_h100",
            "strict_dispatch": _summarize_record(h100) if h100 else None,
            "quality_with_dispatch": _summarize_record(h100_quality) if h100_quality else None,
            "status": h100_status,
            "deferred_reason": None
            if h100_status != "deferred"
            else "No Hopper/H100 GPU is visible on this machine; H100/TK dispatch is out of current hardware scope.",
        }
        h100_artifacts = [
            "Hopper/H100 inference metrics showing fastvideo_sta_h100 backend dispatch",
            f"{min_steps}-step real video quality record produced on the same H100/TK path",
        ]
        if h100_status == "missing":
            missing.append(
                "STA H100/TK C++ path has not been observed on Hopper hardware; "
                "A100 Triton fallback evidence is not H100 kernel evidence"
            )
            next_artifacts.extend(h100_artifacts)
        elif h100_status == "deferred":
            deferred_artifacts.extend(h100_artifacts)

    status = "pass" if not missing else ("partial" if strict_dispatch or quality else "missing")
    return {
        "method": method,
        "status": status,
        "record_count": len(method_records),
        "evidence": {
            "strict_dispatch": _summarize_record(strict_dispatch) if strict_dispatch else None,
            "quality": _summarize_record(quality) if quality else None,
            "quality_with_dispatch": _summarize_record(quality_with_dispatch)
            if quality_with_dispatch
            else None,
            **extra_evidence,
        },
        "missing": missing,
        "next_artifacts": next_artifacts,
        "deferred_artifacts": deferred_artifacts,
    }


def build_audit(records: list[dict[str, Any]], *, min_steps: int) -> dict[str, Any]:
    method_audits = [_method_audit(method, records, min_steps) for method in EXPECTED_METHODS]
    methods = {item["method"]: item for item in method_audits}
    sta_evidence = methods.get("sta", {}).get("evidence", {})
    checklist = [
        _public_api_and_config_gate(),
        _public_methods_gate(),
        _all_backbone_support_contract_gate(),
        _all_backbone_checkpoint_availability_gate(),
        _all_backbone_smoke_evidence_gate(records),
        _runtime_ownership_gate(),
        {
            "gate": "per_method_strict_dispatch_and_quality_evidence",
            "status": "pass" if all(item["status"] == "pass" for item in method_audits) else "fail",
            "evidence": {
                item["method"]: {
                    "status": item["status"],
                    "record_count": item["record_count"],
                }
                for item in method_audits
            },
            "missing": [
                f"{item['method']}: {', '.join(item['missing'])}"
                for item in method_audits
                if item["missing"]
            ],
        },
    ]
    overall = "complete" if all(item["status"] == "pass" for item in checklist) else "incomplete"
    return {
        "objective": "Ensure SparseVideo methods are the same implementation as the referenced training_free methods.",
        "overall_status": overall,
        "min_quality_steps": min_steps,
        "metric_record_count": len(records),
        "checklist": checklist,
        "methods": methods,
        "sta_path_status": {
            "sta_a100": sta_evidence.get("sta_a100"),
            "sta_h100": sta_evidence.get("sta_h100"),
        },
        "required_next_artifacts": {
            item["method"]: item["next_artifacts"]
            for item in method_audits
            if item["next_artifacts"]
        },
        "deferred_next_artifacts": {
            item["method"]: item["deferred_artifacts"]
            for item in method_audits
            if item.get("deferred_artifacts")
        },
    }


def _print_markdown(audit: dict[str, Any]) -> None:
    print(f"# SparseVideo Parity Audit\n")
    print(f"Overall: **{audit['overall_status']}**")
    print(f"Metric records: {audit['metric_record_count']}")
    print(f"Minimum quality steps: {audit['min_quality_steps']}\n")
    print("## Checklist")
    for item in audit["checklist"]:
        print(f"- {item['status']}: {item['gate']}")
        for missing in item["missing"]:
            print(f"  - missing: {missing}")
    print("\n## Methods")
    for method, item in audit["methods"].items():
        print(f"- {method}: {item['status']}")
        for missing in item["missing"]:
            print(f"  - missing: {missing}")
        for artifact in item["next_artifacts"]:
            print(f"  - next_artifact: {artifact}")
        for artifact in item.get("deferred_artifacts", []):
            print(f"  - deferred_artifact: {artifact}")
        evidence = item["evidence"]
        reference_policy = evidence.get("reference_policy")
        if reference_policy:
            print(
                "  - reference_policy_candidates: "
                f"{reference_policy.get('video_policy_candidates', [])}"
            )
            print(
                "  - reference_benchmark_sparse_helpers: "
                f"{reference_policy.get('benchmark_sparse_helpers', [])}"
            )
            print(
                "  - reference_benchmark_global_random_helpers: "
                f"{reference_policy.get('benchmark_global_random_helpers', [])}"
            )
            print(
                "  - reference_benchmark_score_sparse_helpers: "
                f"{reference_policy.get('benchmark_score_sparse_helpers', [])}"
            )
            git_history = reference_policy.get("git_history") or {}
            print(
                "  - reference_git_history_policy_candidates: "
                f"{git_history.get('policy_candidates', [])}"
            )
            public_status = reference_policy.get("public_source_status")
            if public_status:
                print(
                    "  - reference_public_code_url: "
                    f"{public_status.get('public_code_url')}"
                )
                print(
                    "  - reference_public_observed_dirs: "
                    f"{public_status.get('observed_public_dirs', [])}"
                )
                print(
                    "  - reference_public_observed_files: "
                    f"{public_status.get('observed_public_files', [])}"
                )
                print(
                    "  - reference_public_anonymous_repo_api_status: "
                    f"{public_status.get('anonymous_repo_api_status')}"
                )
                print(
                    "  - reference_public_anonymous_hunyuan_policy_files: "
                    f"{public_status.get('anonymous_hunyuan_policy_files', [])}"
                )
                print(
                    "  - reference_public_completion_note: "
                    f"{public_status.get('completion_note')}"
                )
                print(
                    "  - reference_public_issues_status: "
                    f"{public_status.get('github_issues_status')}"
                )
                print(
                    "  - reference_openreview_rebuttal_referenced_paths: "
                    f"{public_status.get('openreview_rebuttal_referenced_paths', [])}"
                )
                print(
                    "  - reference_openreview_rebuttal_missing_paths: "
                    f"{public_status.get('openreview_rebuttal_missing_paths')}"
                )
                print(
                    "  - reference_openreview_revision_pdf_status: "
                    f"{public_status.get('openreview_revision_pdf_status')}"
                )
                print(
                    "  - reference_openreview_revision_threshold_schedule_status: "
                    f"{public_status.get('openreview_revision_threshold_schedule_status')}"
                )
                print(
                    "  - reference_arxiv_source_status: "
                    f"{public_status.get('arxiv_source_status')}"
                )
                print(
                    "  - reference_arxiv_threshold_schedule_status: "
                    f"{public_status.get('arxiv_threshold_schedule_status')}"
                )
        method_path = evidence.get("method_path")
        if method_path:
            print(
                "  - flashomni_owned_sparse_gemm_runtime: "
                f"{method_path.get('owned_sparse_gemm_runtime')}"
            )
            print(
                "  - flashomni_method_uses_sparse_gemm: "
                f"{method_path.get('method_uses_sparse_gemm')} "
                f"{method_path.get('method_sparse_gemm_hits', [])}"
            )
            print(
                "  - flashomni_method_uses_update_dispatch_cache: "
                f"{method_path.get('method_uses_update_dispatch_cache')} "
                f"{method_path.get('method_update_dispatch_hits', [])}"
            )
            print(
                "  - flashomni_owned_paper_policy_source: "
                f"{method_path.get('owned_paper_policy_source')} "
                f"{method_path.get('owned_paper_policy_path')}"
            )
            print(
                "  - flashomni_current_policy_sha256: "
                f"{method_path.get('current_policy_sha256')}"
            )
            print(
                "  - flashomni_current_method_sha256: "
                f"{method_path.get('current_method_sha256')}"
            )
            print(
                "  - flashomni_owned_score_cdf_policy: "
                f"{method_path.get('owned_score_cdf_policy')}"
            )
            print(
                "  - flashomni_owned_hunyuan_video_policy: "
                f"{method_path.get('owned_hunyuan_video_policy')}"
            )
            print(
                "  - flashomni_owned_hunyuan_transformer_forward_taylor_cache: "
                f"{method_path.get('owned_hunyuan_transformer_forward_taylor_cache')}"
            )
        paper_policy_runtime = evidence.get("paper_policy_runtime")
        if paper_policy_runtime:
            print(
                "  - flashomni_paper_policy_runtime: "
                f"{paper_policy_runtime['source']} "
                f"{paper_policy_runtime['model']} "
                f"{paper_policy_runtime['steps']} steps "
                f"generate_sec={paper_policy_runtime['generate_sec']}"
            )
            artifact_qc = paper_policy_runtime.get("artifact_qc") or {}
            if artifact_qc:
                print(f"  - flashomni_paper_policy_artifact_qc: {artifact_qc}")
            print(
                "  - flashomni_reported_hunyuan_config: "
                f"{evidence.get('reported_hunyuan_config')}"
            )
            print(
                "  - flashomni_paper_threshold_schedule_status: "
                f"{evidence.get('paper_threshold_schedule_status')}"
            )
            print(
                "  - flashomni_paper_policy_matches_reported_hunyuan_config: "
                f"{evidence.get('paper_policy_matches_reported_hunyuan_config')}"
            )
        goal_checklist = evidence.get("goal_checklist")
        if goal_checklist:
            print("  - flashomni_goal_checklist:")
            for check in goal_checklist:
                print(
                    "    - "
                    f"{check.get('status')}: {check.get('requirement')}"
                )
                if check.get("caveat"):
                    print(f"      caveat: {check.get('caveat')}")
        hardware = evidence.get("hardware")
        if hardware:
            devices = [
                f"{device.get('index')}:{device.get('name')} cc={device.get('capability')}"
                for device in hardware.get("devices", [])
            ]
            print(f"  - hardware_hopper_visible: {hardware.get('hopper_visible')}")
            print(f"  - hardware_devices: {devices}")
        for key in ("sta_a100", "sta_h100"):
            path_status = evidence.get(key)
            if path_status:
                print(f"  - {key}: {path_status['status']} backend={path_status.get('backend')}")
                comparison = path_status.get("speed_comparison")
                if comparison:
                    print(f"  - {key}_speed_comparison: {comparison['status']}")
                    if comparison.get("status") == "pass":
                        baseline = comparison["baseline"]
                        optimized = comparison["optimized"]
                        print(
                            f"  - {key}_speedup: "
                            f"{comparison['generate_sec_speedup']:.3f}x "
                            f"({baseline.get('generate_sec'):.3f}s -> "
                            f"{optimized.get('generate_sec'):.3f}s generate)"
                        )
                record = path_status.get("quality_with_dispatch")
                if record:
                    print(
                        f"  - {key}_quality_with_dispatch: "
                        f"{record['source']} {record['model']} {record['steps']} steps"
                    )
        for key in ("strict_dispatch", "quality", "quality_with_dispatch"):
            record = evidence.get(key)
            if record:
                timing = ""
                if record.get("generate_sec") is not None:
                    timing = f" generate_sec={record['generate_sec']}"
                print(
                    f"  - {key}: {record['source']} "
                    f"{record['model']} {record['steps']} steps{timing}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        action="append",
        default=[],
        type=Path,
        help="Additional metrics JSONL path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--metrics-glob",
        action="append",
        default=None,
        help=(
            "Repo-relative glob for metrics JSONL files. Default: "
            f"{', '.join(DEFAULT_METRICS_GLOBS)}"
        ),
    )
    parser.add_argument("--min-steps", type=int, default=50)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with status 1 when the audit is incomplete.",
    )
    args = parser.parse_args()

    records = _collect_metrics(args.metrics, args.metrics_glob or list(DEFAULT_METRICS_GLOBS))
    audit = build_audit(records, min_steps=args.min_steps)
    if args.format == "markdown":
        _print_markdown(audit)
    else:
        print(_json_dumps(audit))
    if args.fail_on_incomplete and audit["overall_status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
