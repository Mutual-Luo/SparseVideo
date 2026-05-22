#!/usr/bin/env python3
"""Run a small DiffSynth-Studio SparseVideo method sweep.

This is a smoke/dispatch check, not quality evidence. It loads DiffSynth
pipelines, applies each requested SparseVideo method, optionally runs a tiny
generation, records dispatch/backend counts, saves outputs, restores, then
moves on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparsevideo import apply_sparse_attention
from _diffsynth_models import (
    DEFAULT_MODEL_ROOT,
    diffsynth_output_audio_sample_rate,
    diffsynth_model_list_lines,
    get_diffsynth_model_spec,
    list_diffsynth_model_specs,
    load_diffsynth_pipeline,
    resolve_diffsynth_model_paths,
    save_diffsynth_output,
)


PUBLIC_SPARSE_METHODS = (
    "svg1",
    "svg2",
    "spargeattn",
    "radial",
    "sta",
    "draft",
    "adacluster",
    "flashomni",
    "svoo",
)
ALL_METHODS = ("dense", *PUBLIC_SPARSE_METHODS)
DEFAULT_SMOKE_NUM_FRAMES = 5
LTX2_SMOKE_NUM_FRAMES = 9


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for method in PUBLIC_SPARSE_METHODS:
            print(method)
        return 0
    if args.list_models:
        for line in diffsynth_model_list_lines():
            print(line)
        return 0

    try:
        methods = _parse_methods(args.methods)
        models = _parse_models(args.models or args.model)
    except Exception as exc:
        payload = {
            "backend": "diffsynth",
            "model": args.models or args.model,
            "models": [],
            "methods": args.methods,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit_payload(args, payload)
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or (output_dir / "metrics.jsonl")
    _prepare_metrics_file(metrics_file, append=args.append_metrics)

    payload: Dict[str, Any] = {
        "backend": "diffsynth",
        "model": models[0] if len(models) == 1 else "multiple",
        "models": models,
        "methods": methods,
        "status": "running",
        "results": [],
    }
    failed = False

    for model in models:
        model_output_dir = output_dir / model if len(models) > 1 else output_dir
        model_output_dir.mkdir(parents=True, exist_ok=True)
        resolved_model = None
        try:
            resolved = resolve_diffsynth_model_paths(model, model_root=args.model_root)
            resolved_model = resolved.as_dict()
            payload.setdefault("resolved_models", {})[model] = resolved_model
        except Exception as exc:
            record = {
                "backend": "diffsynth",
                "model": model,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if resolved_model is not None:
                record["resolved_model"] = resolved_model
            payload["results"].append(record)
            _append_jsonl(metrics_file, record)
            _print_model_line(record)
            failed = True
            continue

        generation_error = _generation_smoke_error(model, args) or _generation_shape_error(model, args)
        if generation_error is not None:
            for method in methods:
                method_config = _smoke_method_config(method, args.num_inference_steps)
                record = {
                    "backend": "diffsynth",
                    "model": model,
                    "method": method,
                    "status": "failed",
                    "mode": "generate",
                    "error_type": "ValueError",
                    "error": generation_error,
                    "method_config": method_config,
                }
                if resolved_model is not None:
                    record["resolved_model"] = resolved_model
                payload["results"].append(record)
                _append_jsonl(metrics_file, record)
                _print_method_line(record)
            failed = True
            continue

        if not resolved.complete:
            record = {
                "backend": "diffsynth",
                "model": model,
                "status": "failed",
                "error_type": "FileNotFoundError",
                "error": (
                    f"DiffSynth model '{resolved.spec.key}' is incomplete under {resolved.model_root}. "
                    "Run scripts/download_diffsynth_models.sh for the missing native DiffSynth files."
                ),
                "resolved_model": resolved_model,
            }
            payload["results"].append(record)
            _append_jsonl(metrics_file, record)
            _print_model_line(record)
            failed = True
            continue

        try:
            torch_dtype = _torch_dtype(args.dtype)
            pipe, resolved = load_diffsynth_pipeline(
                model,
                model_root=args.model_root,
                torch_dtype=torch_dtype,
                device=args.device,
                offload_device=args.offload_device,
                vram_limit=args.vram_limit,
                enable_vram_management=not args.no_vram_management,
            )
        except Exception as exc:
            record = {
                "backend": "diffsynth",
                "model": model,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if resolved_model is not None:
                record["resolved_model"] = resolved_model
            payload["results"].append(record)
            _append_jsonl(metrics_file, record)
            _print_model_line(record)
            failed = True
            continue

        try:
            for method in methods:
                record = _run_method_smoke(
                    pipe,
                    args,
                    model,
                    method,
                    model_output_dir,
                    resolved_model=resolved_model,
                )
                payload["results"].append(record)
                _append_jsonl(metrics_file, record)
                _print_method_line(record)
                failed = failed or record["status"] != "ok"
        finally:
            _cleanup_after_model(pipe, args.device)

    payload["status"] = "failed" if failed else "ok"
    payload["metrics_file"] = str(metrics_file)
    _emit_payload(args, payload)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", default="wan21-t2v-1.3b")
    parser.add_argument("--models", help="Comma-separated DiffSynth model list or 'all'. Overrides --model.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--methods", default="all", help="Comma-separated method list, 'all', or 'sparse'.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "diffsynth_method_smoke")
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument(
        "--append-metrics",
        action="store_true",
        help="Append to an existing metrics JSONL file instead of replacing it at run start.",
    )
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--apply-only", action="store_true", help="Apply and restore methods without generation.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload-device", default="cpu")
    parser.add_argument("--vram-limit", type=float)
    parser.add_argument("--no-vram-management", action="store_true")

    parser.add_argument("--prompt", default="A calm lake.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Frame count for generation smoke. Defaults to 5, or 9 for LTX2 to satisfy its 8n+1 frame rule.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--video-quality", type=int, default=5)
    return parser


def _run_method_smoke(
    pipe,
    args,
    model: str,
    method: str,
    output_dir: Path,
    *,
    resolved_model: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    _reset_cuda_memory(args.device)
    started = time.perf_counter()
    handle = None
    record: Dict[str, Any] = {
        "backend": "diffsynth",
        "model": get_diffsynth_model_spec(model).key,
        "method": method,
        "status": None,
    }
    if resolved_model is not None:
        record["resolved_model"] = dict(resolved_model)
    method_config = _smoke_method_config(method, args.num_inference_steps)
    call_steps = _smoke_call_steps(method, args.num_inference_steps)
    try:
        handle = apply_sparse_attention(pipe, method=method, config=method_config)
        record["apply_summary"] = handle.summary()
        _validate_apply_summary(method, record["apply_summary"])
        if args.apply_only:
            record.update(
                {
                    "status": "ok",
                    "mode": "apply_only",
                    "method_config": method_config,
                }
            )
        else:
            call_kwargs = _call_kwargs(args, model, call_steps)
            generate_started = time.perf_counter()
            video = pipe(**call_kwargs)
            generate_sec = time.perf_counter() - generate_started
            record["generate_summary"] = handle.summary()
            _validate_generate_summary(method, record["generate_summary"])
            fps = args.fps or get_diffsynth_model_spec(model).default_fps
            output_metadata = save_diffsynth_output(
                video,
                output_dir / f"{method}.mp4",
                fps=fps,
                quality=args.video_quality,
                audio_sample_rate=diffsynth_output_audio_sample_rate(pipe),
            )
            record.update(
                {
                    "status": "ok",
                    "mode": "generate",
                    "generate_kwargs": _jsonable(call_kwargs),
                    "method_config": method_config,
                    "timings": {"generate_sec": generate_sec},
                }
            )
            record.update(output_metadata)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "method_config": method_config,
            }
        )
    finally:
        if handle is not None:
            try:
                handle.restore()
                record["restore_summary"] = handle.summary()
                if record["status"] == "ok":
                    _validate_restore_summary(record["restore_summary"])
            except Exception as exc:
                record["restore_error"] = f"{type(exc).__name__}: {exc}"
                record["status"] = "failed"
                record.setdefault("error_type", type(exc).__name__)
                record.setdefault("error", str(exc))
        elapsed_sec = time.perf_counter() - started
        record["elapsed_sec"] = elapsed_sec
        timings = record.setdefault("timings", {})
        timings.setdefault("total_sec", elapsed_sec)
        record["cuda"] = _cuda_memory(args.device)
    return record


def _validate_apply_summary(method: str, summary: Mapping[str, Any]) -> None:
    if summary.get("pipeline_backend") != "diffsynth":
        raise RuntimeError(f"DiffSynth smoke expected pipeline_backend='diffsynth', got {summary.get('pipeline_backend')!r}")
    layer_count = int(summary.get("num_self_attn_layers") or 0)
    if layer_count <= 0:
        raise RuntimeError("DiffSynth smoke did not report any discovered attention modules")
    if method == "dense":
        patched_count = int(summary.get("patched_attention_count") or 0)
        patched_paths = summary.get("patched_attention_paths")
        if patched_count != 0 or patched_paths not in (None, []):
            raise RuntimeError(f"DiffSynth dense smoke unexpectedly patched {patched_count} attention modules")
        if not summary.get("diffsynth_version"):
            raise RuntimeError("DiffSynth smoke did not record the installed diffsynth version")
        return
    patched_count = int(summary.get("patched_attention_count") or 0)
    if patched_count <= 0:
        raise RuntimeError("DiffSynth smoke did not patch any attention modules")
    if patched_count != layer_count:
        raise RuntimeError(
            "DiffSynth smoke patched only part of the discovered attention modules: "
            f"patched_attention_count={patched_count}, num_self_attn_layers={layer_count}"
        )
    patched_paths = summary.get("patched_attention_paths")
    if not isinstance(patched_paths, list) or len(patched_paths) != patched_count:
        raise RuntimeError(
            "DiffSynth smoke patch path count does not match patched_attention_count: "
            f"patched_attention_count={patched_count}, patched_attention_paths={patched_paths!r}"
        )
    if not summary.get("diffsynth_version"):
        raise RuntimeError("DiffSynth smoke did not record the installed diffsynth version")


def _validate_generate_summary(method: str, summary: Mapping[str, Any]) -> None:
    _validate_apply_summary(method, summary)
    if method == "dense":
        return
    runtime = summary.get("method_runtime") or {}
    dispatch_counts = runtime.get("dispatch_counts") or {}
    sparse_count = int(dispatch_counts.get("sparse") or 0)
    if sparse_count <= 0:
        raise RuntimeError(f"DiffSynth smoke generated without sparse dispatch: dispatch_counts={dispatch_counts}")


def _validate_restore_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("restored") is not True:
        raise RuntimeError("DiffSynth smoke did not restore the original attention path")


def _smoke_method_config(method: str, num_inference_steps: int) -> Dict[str, Any]:
    if method == "dense":
        return {}
    steps = max(1, int(num_inference_steps))
    config: Dict[str, Any] = {
        "dense_warmup_step_ratio": 0,
        "dense_warmup_layer_ratio": 0,
    }
    if method == "draft":
        config["allow_triton_fallback"] = True
    elif method == "flashomni":
        config["num_inference_steps"] = steps
        config["sparse_pattern"] = "global_random"
    return config


def _smoke_call_steps(method: str, num_inference_steps: int) -> int:
    steps = max(1, int(num_inference_steps))
    if method == "draft":
        return max(2, steps)
    return steps


def _generation_smoke_error(model: str, args) -> str | None:
    if args.apply_only:
        return None
    spec = get_diffsynth_model_spec(model)
    if not spec.required_inputs:
        return None
    required = ", ".join(f"--{name.replace('_', '-')}" for name in spec.required_inputs)
    return (
        f"DiffSynth smoke generation for '{spec.key}' requires media input(s) {required}. "
        "Use --apply-only for bundle/patch validation, or scripts/infer_diffsynth.py for a media-specific generation run."
    )


def _generation_shape_error(model: str, args) -> str | None:
    if args.apply_only:
        return None
    spec = get_diffsynth_model_spec(model)
    height = args.height
    width = args.width
    num_frames = _smoke_num_frames(args, spec)
    if spec.pipeline == "LTX2AudioVideoPipeline":
        return _shape_error(
            spec.key,
            height=height,
            width=width,
            num_frames=num_frames,
            spatial_multiple=32,
            time_factor=8,
            time_remainder=1,
        )
    if spec.pipeline in {"WanVideoPipeline", "MovaAudioVideoPipeline"}:
        return _shape_error(
            spec.key,
            height=height,
            width=width,
            num_frames=num_frames,
            spatial_multiple=16,
            time_factor=4,
            time_remainder=1,
        )
    return None


def _shape_error(
    model_key: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    spatial_multiple: int,
    time_factor: int,
    time_remainder: int,
) -> str | None:
    if (
        height % spatial_multiple == 0
        and width % spatial_multiple == 0
        and num_frames % time_factor == time_remainder
    ):
        return None
    return (
        f"DiffSynth model '{model_key}' requires height/width multiples of {spatial_multiple} "
        f"and num_frames % {time_factor} == {time_remainder}; got "
        f"height={height}, width={width}, num_frames={num_frames}."
    )


def _call_kwargs(args, model: str, num_inference_steps: int) -> Dict[str, Any]:
    spec = get_diffsynth_model_spec(model)
    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": _smoke_num_frames(args, spec),
        "num_inference_steps": num_inference_steps,
        "cfg_scale": spec.default_cfg_scale,
        "tiled": True,
    }
    if spec.pipeline == "LTX2AudioVideoPipeline":
        kwargs["frame_rate"] = args.fps or spec.default_fps
    else:
        kwargs["sigma_shift"] = spec.default_sigma_shift
    if spec.pipeline == "MovaAudioVideoPipeline":
        kwargs["frame_rate"] = args.fps or spec.default_fps
        kwargs["switch_DiT_boundary"] = spec.default_switch_dit_boundary
    elif spec.pipeline != "LTX2AudioVideoPipeline":
        kwargs["switch_DiT_boundary"] = spec.default_switch_dit_boundary
    return kwargs


def _smoke_num_frames(args, spec) -> int:
    if args.num_frames is not None:
        return args.num_frames
    if spec.pipeline == "LTX2AudioVideoPipeline":
        return LTX2_SMOKE_NUM_FRAMES
    return DEFAULT_SMOKE_NUM_FRAMES


def _parse_models(value: str) -> list[str]:
    if value == "all":
        return [spec.key for spec in list_diffsynth_model_specs()]
    models = [get_diffsynth_model_spec(item.strip()).key for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("No DiffSynth models selected")
    return models


def _parse_methods(value: str) -> list[str]:
    if value == "all":
        return list(ALL_METHODS)
    if value == "sparse":
        return list(PUBLIC_SPARSE_METHODS)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    supported = set(ALL_METHODS)
    unknown = [method for method in methods if method not in supported]
    if unknown:
        raise ValueError(f"Unsupported DiffSynth smoke method(s): {unknown}")
    return methods


def _print_method_line(record: Mapping[str, Any]) -> None:
    runtime = (record.get("generate_summary") or {}).get("method_runtime") or {}
    print(
        " ".join(
            [
                f"model={record['model']}",
                f"method={record['method']}",
                f"status={record['status']}",
                f"dispatch={runtime.get('dispatch_counts')}",
                f"backend={runtime.get('backend_counts')}",
            ]
        ),
        flush=True,
    )


def _print_model_line(record: Mapping[str, Any]) -> None:
    print(
        " ".join(
            [
                f"model={record['model']}",
                f"status={record['status']}",
                f"error={record.get('error_type')}",
            ]
        ),
        flush=True,
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _prepare_metrics_file(path: Path, *, append: bool) -> None:
    if append:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_after_model(pipe, device: str) -> None:
    try:
        load_models_to_device = getattr(pipe, "load_models_to_device", None)
        if callable(load_models_to_device):
            load_models_to_device([])
    except Exception:
        pass
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _reset_cuda_memory(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(_cuda_device_arg(str(device)))
    except Exception:
        return


def _cuda_memory(device: str) -> Dict[str, Any]:
    if not str(device).startswith("cuda"):
        return {"device": device, "available": False}
    try:
        import torch

        return {
            "device": device,
            "available": torch.cuda.is_available(),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(_cuda_device_arg(str(device))) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(_cuda_device_arg(str(device))) / 1024**3,
        }
    except Exception as exc:
        return {"device": device, "available": False, "error": str(exc)}


def _cuda_device_arg(device: str) -> str | None:
    return None if device == "cuda" else device


def _emit_payload(args, payload: Dict[str, Any]) -> None:
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={payload['status']} backend=diffsynth model={payload['model']}")
        if payload.get("metrics_file"):
            print(f"metrics_file={payload['metrics_file']}")


def _torch_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _jsonable(data: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(data)


if __name__ == "__main__":
    raise SystemExit(main())
