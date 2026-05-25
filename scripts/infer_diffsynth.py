#!/usr/bin/env python3
"""SparseVideo DiffSynth-Studio inference entrypoint.

Diffusers inference stays in scripts/infer.py. This script uses DiffSynth-native
ModelConfig(path=...) loading so flat local model directories can be reused
without starting a new download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (SCRIPT_DIR, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sparsevideo import apply_sparse_attention, default_method_config, normalize_method_config
from _infer_diffsynth.models import (
    DEFAULT_MODEL_ROOT,
    diffsynth_output_audio_sample_rate,
    diffsynth_model_list_lines,
    get_diffsynth_model_spec,
    load_diffsynth_pipeline,
    resolve_diffsynth_model_paths,
    save_diffsynth_output,
)


DEFAULT_PROMPT = "A serene mountain lake at sunrise, cinematic, natural motion."
DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, low quality, blurry, distorted, static, watermark, subtitles, text"
)
EXPECTED_SPARSE_BACKENDS = {
    "adacluster": {"adacluster_flashinfer", "triton_cluster_sparse_attn", "triton_cluster_sparse_attn_topk"},
    "draft": {"mit_block_sparse"},
    "flashomni": {"flashomni_explicit_upstream"},
    "radial": {"flashinfer", "sage_block_sparse"},
    "spargeattn": {"spas_sage", "spas_sage_block_sparse", "spas_sage_cdfthreshd", "spas_sage_topk", "spas_sage_tuned"},
    "sta": {"fastvideo_sta_h100", "fastvideo_sta_a100_block_sparse_cuda"},
    "svg1": {"flex_attention"},
    "svg2": {"flashinfer"},
    "svoo": {"svoo_flashinfer"},
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for line in diffsynth_model_list_lines():
            print(line)
        return 0

    payload: Dict[str, Any] = {
        "backend": "diffsynth",
        "model": args.model,
        "method": args.method,
    }

    try:
        spec = get_diffsynth_model_spec(args.model)
        payload["model"] = spec.key
        payload.update(
            {
                "model_arg": args.model,
                "height": args.height or spec.default_height,
                "width": args.width or spec.default_width,
                "num_frames": args.num_frames or spec.default_num_frames,
                "fps": args.fps or spec.default_fps,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "device": args.device,
                "dtype": args.dtype,
            }
        )
        if args.dry_run:
            resolved = resolve_diffsynth_model_paths(args.model, model_root=args.model_root)
            payload.update(
                {
                    "status": "dry_run",
                    "resolved_model": resolved.as_dict(),
                }
            )
            return 0 if resolved.complete else 1

        if not args.apply_only:
            _validate_generation_inputs(args, check_paths=True)

        resolved = resolve_diffsynth_model_paths(args.model, model_root=args.model_root)
        payload["resolved_model"] = resolved.as_dict()
        if not resolved.complete:
            missing = "\n  - ".join(resolved.missing)
            raise FileNotFoundError(
                f"DiffSynth model '{resolved.spec.key}' is incomplete under {resolved.model_root}.\n"
                f"  - {missing}\n"
                "Run scripts/download_diffsynth_models.sh for the missing native DiffSynth files."
            )

        started = time.perf_counter()
        _configure_method_runtime_env(args.method)
        torch_dtype = _torch_dtype(args.dtype)
        _reset_cuda_memory(args.device)
        pipe, resolved = load_diffsynth_pipeline(
            args.model,
            model_root=args.model_root,
            torch_dtype=torch_dtype,
            device=args.device,
            offload_device=args.offload_device,
            vram_limit=args.vram_limit,
            use_usp=args.use_usp,
            enable_vram_management=not args.no_vram_management,
        )

        handle = None
        try:
            method_config = _build_method_config(args, spec)
            payload["method_config"] = method_config
            handle = apply_sparse_attention(pipe, method=args.method, config=method_config)
            apply_summary = handle.summary()
            payload["apply_summary"] = apply_summary
            payload["sparse_attention_handle"] = apply_summary
            payload["method_config"] = apply_summary.get("method_config") or method_config
            _validate_sparse_apply_summary(args.method, apply_summary)

            if args.apply_only:
                payload.update(
                    {
                        "status": "ok",
                        "mode": "apply_only",
                        "timings": {
                            "load_apply_sec": time.perf_counter() - started,
                        },
                        "cuda": _cuda_memory(args.device),
                    }
                )
                return 0

            prompt = _read_prompt(args)
            call_kwargs = _build_call_kwargs(args, prompt)
            output_file = Path(args.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            generate_started = time.perf_counter()
            video = pipe(**call_kwargs)
            generate_sec = time.perf_counter() - generate_started
            payload["generate_summary"] = handle.summary()
            payload["sparse_attention_handle"] = payload["generate_summary"]
            _validate_sparse_generate_summary(args.method, payload["generate_summary"])
            _validate_sparse_backend_summary(args.method, payload["generate_summary"])

            save_fps = args.fps or spec.default_fps
            output_metadata = save_diffsynth_output(
                video,
                output_file,
                fps=save_fps,
                quality=args.video_quality,
                audio_sample_rate=diffsynth_output_audio_sample_rate(pipe),
            )
            payload.update(
                {
                    "status": "ok",
                    "mode": "generate",
                    "method_config": payload["generate_summary"].get("method_config") or method_config,
                    "fps": save_fps,
                    "generate_kwargs": _jsonable(call_kwargs),
                    "timings": {
                        "load_apply_generate_save_sec": time.perf_counter() - started,
                        "generate_sec": generate_sec,
                    },
                    "cuda": _cuda_memory(args.device),
                }
            )
            payload.update(output_metadata)
            return 0
        finally:
            if handle is not None:
                handle.restore()
                payload["restore_summary"] = handle.summary()
                payload["sparse_attention_handle_after_restore"] = payload["restore_summary"]
                _validate_restore_summary(payload["restore_summary"])
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return 1
    finally:
        _emit_payload(args, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", help="List DiffSynth model keys and exit.")
    parser.add_argument("--model", default="wan21-t2v-1.3b", help="DiffSynth model key.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--method", default="dense", help="SparseVideo method name.")
    parser.add_argument("--method-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Resolve local files without loading DiffSynth.")
    parser.add_argument("--apply-only", action="store_true", help="Load pipeline, apply/restore SparseVideo, skip generation.")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON payload.")
    parser.add_argument("--metrics-file", type=Path, help="Append JSON metrics payload to this file.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--offload-device", default="cpu")
    parser.add_argument("--vram-limit", type=float)
    parser.add_argument("--no-vram-management", action="store_true")
    parser.add_argument("--use-usp", action="store_true", help="Use DiffSynth USP. Sparse methods currently reject this.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--input-image", type=Path, help="Optional image input for TI2V/I2V-style DiffSynth runs.")
    parser.add_argument("--end-image", type=Path, help="Optional end image for FLF2V-style DiffSynth runs.")
    parser.add_argument("--input-video", type=Path, help="Optional video file or image folder for video-to-video runs.")
    parser.add_argument("--control-video", type=Path, help="Optional video file or image folder for control runs.")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--camera-control-direction")
    parser.add_argument("--camera-control-speed", type=float)
    parser.add_argument("--camera-control-origin", type=_parse_float_tuple)
    parser.add_argument("--vace-video", type=Path, help="Optional video file or image folder for VACE.")
    parser.add_argument("--vace-video-mask", type=Path, help="Optional image mask for VACE.")
    parser.add_argument("--vace-reference-image", type=Path)
    parser.add_argument("--vace-scale", type=float)
    parser.add_argument("--animate-pose-video", type=Path, help="Optional video file or image folder for Wan Animate.")
    parser.add_argument("--animate-face-video", type=Path, help="Optional video file or image folder for Wan Animate.")
    parser.add_argument("--animate-inpaint-video", type=Path, help="Optional video file or image folder for Wan Animate.")
    parser.add_argument("--animate-mask-video", type=Path, help="Optional video file or image folder for Wan Animate.")
    parser.add_argument("--vap-video", type=Path, help="Video file or image folder for Video-as-Prompt Wan.")
    parser.add_argument("--vap-prompt")
    parser.add_argument("--negative-vap-prompt")
    parser.add_argument("--longcat-video", type=Path, help="Video file or image folder for LongCat-Video.")
    parser.add_argument("--input-audio", type=Path, help="Optional audio file for S2V/audio-video DiffSynth runs.")
    parser.add_argument("--s2v-pose-video", type=Path, help="Optional pose video file or image folder for S2V.")
    parser.add_argument("--motion-video", type=Path, help="Optional motion video file or image folder for S2V.")
    parser.add_argument("--wantodance-music-path", type=Path)
    parser.add_argument("--wantodance-reference-image", type=Path)
    parser.add_argument("--wantodance-fps", type=float)
    parser.add_argument("--wantodance-keyframes", type=Path, help="Optional image folder or video for WanToDance keyframes.")
    parser.add_argument("--wantodance-keyframes-mask", type=_parse_int_list)
    parser.add_argument("--in-context-video", action="append", type=Path, default=[], help="LTX2 in-context video; repeat for multiple videos.")
    parser.add_argument("--in-context-downsample-factor", type=int)
    parser.add_argument("--retake-video-regions", type=_parse_regions)
    parser.add_argument("--retake-audio-regions", type=_parse_regions)
    parser.add_argument("--audio-start-time", type=float, default=0.0)
    parser.add_argument("--audio-duration", type=float)
    parser.add_argument("--output-file", type=Path, default=REPO_ROOT / "outputs" / "diffsynth.mp4")
    parser.add_argument("--video-quality", type=int, default=5)

    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rand-device")
    parser.add_argument("--denoising-strength", type=float)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--cfg-merge", action="store_true", help="Use DiffSynth Wan cfg_merge path.")
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--switch-dit-boundary", type=float)
    parser.add_argument("--motion-bucket-id", type=int)
    parser.add_argument("--sliding-window-size", type=int)
    parser.add_argument("--sliding-window-stride", type=int)
    parser.add_argument("--tea-cache-l1-thresh", type=float)
    parser.add_argument("--tea-cache-model-id")
    parser.add_argument("--framewise-decoding", action="store_true")
    parser.add_argument("--output-type", choices=("quantized", "floatpoint"), default="quantized")
    parser.add_argument("--tile-size", type=_parse_int_pair)
    parser.add_argument("--tile-stride", type=_parse_int_pair)
    parser.add_argument("--input-images-indexes", type=_parse_int_list)
    parser.add_argument("--input-images-strength", type=float)
    parser.add_argument("--tile-size-in-pixels", type=int)
    parser.add_argument("--tile-overlap-in-pixels", type=int)
    parser.add_argument("--tile-size-in-frames", type=int)
    parser.add_argument("--tile-overlap-in-frames", type=int)
    parser.add_argument("--use-two-stage-pipeline", action="store_true")
    parser.add_argument("--stage2-spatial-upsample-factor", type=int)
    parser.add_argument("--clear-lora-before-stage-two", action="store_true")
    parser.add_argument("--use-distilled-pipeline", action="store_true")
    parser.add_argument("--no-tiled", action="store_true")
    return parser


def _read_prompt(args) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


_PRELOAD_INPUT_PATH_ARGS = (
    "prompt_file",
    "input_image",
    "end_image",
    "input_video",
    "control_video",
    "reference_image",
    "vace_video",
    "vace_video_mask",
    "vace_reference_image",
    "animate_pose_video",
    "animate_face_video",
    "animate_inpaint_video",
    "animate_mask_video",
    "vap_video",
    "longcat_video",
    "input_audio",
    "s2v_pose_video",
    "motion_video",
    "wantodance_music_path",
    "wantodance_reference_image",
    "wantodance_keyframes",
    "in_context_video",
)


def _validate_generation_inputs(args, *, check_paths: bool = False) -> None:
    spec = get_diffsynth_model_spec(args.model)
    missing_inputs = [
        name for name in spec.required_inputs if getattr(args, name, None) is None
    ]
    if missing_inputs:
        missing = ", ".join(f"--{name.replace('_', '-')}" for name in missing_inputs)
        raise ValueError(f"DiffSynth model '{spec.key}' requires {missing} for generation.")
    if check_paths:
        missing_paths = _missing_preload_input_paths(args)
        if missing_paths:
            missing = ", ".join(missing_paths)
            raise ValueError(
                "DiffSynth generation input path(s) do not exist before model loading: "
                f"{missing}"
            )
    shape_error = _generation_shape_error(spec, args)
    if shape_error is not None:
        raise ValueError(shape_error)


def _missing_preload_input_paths(args) -> list[str]:
    missing: list[str] = []
    for name in _PRELOAD_INPUT_PATH_ARGS:
        value = getattr(args, name, None)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for path in values:
            if path is None:
                continue
            if not Path(path).exists():
                missing.append(f"--{name.replace('_', '-')}={path}")
    return missing


def _generation_shape_error(spec, args) -> str | None:
    height = args.height or spec.default_height
    width = args.width or spec.default_width
    num_frames = args.num_frames or spec.default_num_frames
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


def _validate_sparse_apply_summary(method: str, summary: Mapping[str, Any]) -> None:
    if summary.get("pipeline_backend") != "diffsynth":
        raise RuntimeError(
            f"DiffSynth inference expected pipeline_backend='diffsynth', got {summary.get('pipeline_backend')!r}"
        )
    patched_count = int(summary.get("patched_attention_count") or 0)
    layer_count = int(summary.get("num_self_attn_layers") or 0)
    if layer_count <= 0:
        raise RuntimeError("DiffSynth inference did not report any discovered attention modules.")
    if method == "dense":
        patched_paths = summary.get("patched_attention_paths")
        if patched_count != 0 or patched_paths not in (None, []):
            raise RuntimeError(f"DiffSynth dense inference unexpectedly patched {patched_count} attention modules.")
        if not summary.get("diffsynth_version"):
            raise RuntimeError("DiffSynth inference did not record the installed diffsynth version.")
        return
    if patched_count <= 0:
        raise RuntimeError(f"DiffSynth method '{method}' did not patch any attention modules.")
    if patched_count != layer_count:
        raise RuntimeError(
            f"DiffSynth method '{method}' patched only part of the discovered attention modules: "
            f"patched_attention_count={patched_count}, num_self_attn_layers={layer_count}"
        )
    patched_paths = summary.get("patched_attention_paths")
    if not isinstance(patched_paths, list) or len(patched_paths) != patched_count:
        raise RuntimeError(
            f"DiffSynth method '{method}' patch path count does not match patched_attention_count: "
            f"patched_attention_count={patched_count}, patched_attention_paths={patched_paths!r}"
        )
    if not summary.get("diffsynth_version"):
        raise RuntimeError("DiffSynth inference did not record the installed diffsynth version.")


def _validate_sparse_generate_summary(method: str, summary: Mapping[str, Any]) -> None:
    _validate_sparse_apply_summary(method, summary)
    if method == "dense" or summary.get("pipeline_backend") != "diffsynth":
        return
    runtime = summary.get("method_runtime") or {}
    dispatch_counts = runtime.get("dispatch_counts") or {}
    sparse_count = int(dispatch_counts.get("sparse") or 0)
    if sparse_count <= 0:
        raise RuntimeError(
            f"DiffSynth method '{method}' generated without sparse dispatch: "
            f"dispatch_counts={dispatch_counts}"
        )


def _validate_sparse_backend_summary(
    method: str,
    summary: Mapping[str, Any],
) -> None:
    if method == "dense":
        return
    runtime = summary.get("method_runtime") or {}
    backend_counts = runtime.get("backend_counts") or {}
    observed_backends = {
        str(name)
        for name, count in backend_counts.items()
        if _runtime_count(count) > 0
    }
    if not observed_backends:
        raise RuntimeError(
            f"DiffSynth method '{method}' generated without backend evidence: "
            f"backend_counts={backend_counts}"
        )
    debug_backends = sorted(name for name in observed_backends if "debug_fallback" in name)
    if debug_backends:
        raise RuntimeError(
            f"DiffSynth method '{method}' used debug fallback backend(s) in strict mode: "
            f"{debug_backends}"
        )
    expected = EXPECTED_SPARSE_BACKENDS.get(method, set())
    if expected and not (observed_backends & expected):
        raise RuntimeError(
            f"DiffSynth method '{method}' used unexpected sparse backend(s): "
            f"observed={sorted(observed_backends)}, expected={sorted(expected)}"
        )


def _validate_restore_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("restored") is not True:
        raise RuntimeError("DiffSynth inference did not restore the original attention path.")


def _build_call_kwargs(args, prompt: str) -> Dict[str, Any]:
    spec = get_diffsynth_model_spec(args.model)
    _validate_generation_inputs(args)

    height = args.height or spec.default_height
    width = args.width or spec.default_width
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": height,
        "width": width,
        "num_frames": args.num_frames or spec.default_num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale if args.cfg_scale is not None else spec.default_cfg_scale,
        "tiled": not args.no_tiled,
    }
    if args.rand_device is not None:
        kwargs["rand_device"] = args.rand_device
    is_wan_pipeline = spec.pipeline == "WanVideoPipeline"
    if spec.pipeline == "LTX2AudioVideoPipeline":
        kwargs["frame_rate"] = args.fps or spec.default_fps
    else:
        kwargs["sigma_shift"] = args.sigma_shift if args.sigma_shift is not None else spec.default_sigma_shift
    if spec.pipeline == "MovaAudioVideoPipeline":
        kwargs["frame_rate"] = args.fps or spec.default_fps
        kwargs["switch_DiT_boundary"] = (
            args.switch_dit_boundary
            if args.switch_dit_boundary is not None
            else spec.default_switch_dit_boundary
        )
    elif spec.pipeline != "LTX2AudioVideoPipeline":
        kwargs["switch_DiT_boundary"] = (
            args.switch_dit_boundary
            if args.switch_dit_boundary is not None
            else spec.default_switch_dit_boundary
        )
    if args.denoising_strength is not None:
        kwargs["denoising_strength"] = args.denoising_strength
    if args.input_image:
        if spec.pipeline == "LTX2AudioVideoPipeline":
            kwargs["input_images"] = [_load_image(args.input_image)]
        else:
            kwargs["input_image"] = _load_image(args.input_image)
    if args.end_image:
        kwargs["end_image"] = _load_image(args.end_image)
    if args.reference_image:
        kwargs["reference_image"] = _load_image(args.reference_image)
    if args.wantodance_reference_image:
        kwargs["wantodance_reference_image"] = _load_image(args.wantodance_reference_image)
    if args.vace_reference_image:
        kwargs["vace_reference_image"] = _load_image(args.vace_reference_image)
    if args.vace_video_mask:
        kwargs["vace_video_mask"] = _load_image(args.vace_video_mask)

    for name in (
        "input_video",
        "control_video",
        "vace_video",
        "animate_pose_video",
        "animate_face_video",
        "animate_inpaint_video",
        "animate_mask_video",
        "vap_video",
        "longcat_video",
        "s2v_pose_video",
        "motion_video",
        "wantodance_keyframes",
    ):
        path = getattr(args, name, None)
        if path is not None:
            target_name = "retake_video" if spec.pipeline == "LTX2AudioVideoPipeline" and name == "input_video" else name
            kwargs[target_name] = _load_video_frames(path, height=height, width=width)
    if args.in_context_video:
        kwargs["in_context_videos"] = [
            _load_video_frames(path, height=height, width=width)
            for path in args.in_context_video
        ]

    if args.input_audio:
        input_audio, audio_sample_rate = _load_audio_input(
            args.input_audio,
            start_time=args.audio_start_time,
            duration=args.audio_duration,
            as_numpy=spec.pipeline != "LTX2AudioVideoPipeline",
        )
        audio_name = "retake_audio" if spec.pipeline == "LTX2AudioVideoPipeline" else "input_audio"
        kwargs[audio_name] = input_audio
        kwargs["audio_sample_rate"] = audio_sample_rate

    if args.vace_scale is not None:
        kwargs["vace_scale"] = args.vace_scale
    if args.vap_prompt is not None:
        kwargs["vap_prompt"] = args.vap_prompt
    if args.negative_vap_prompt is not None:
        kwargs["negative_vap_prompt"] = args.negative_vap_prompt
    if is_wan_pipeline:
        kwargs["cfg_merge"] = args.cfg_merge
        kwargs["framewise_decoding"] = args.framewise_decoding
        kwargs["output_type"] = args.output_type
        if args.tile_size is not None:
            kwargs["tile_size"] = args.tile_size
        if args.tile_stride is not None:
            kwargs["tile_stride"] = args.tile_stride
        if args.camera_control_direction is not None:
            kwargs["camera_control_direction"] = args.camera_control_direction
        if args.camera_control_speed is not None:
            kwargs["camera_control_speed"] = args.camera_control_speed
        if args.camera_control_origin is not None:
            kwargs["camera_control_origin"] = args.camera_control_origin
        if args.motion_bucket_id is not None:
            kwargs["motion_bucket_id"] = args.motion_bucket_id
        if args.sliding_window_size is not None:
            kwargs["sliding_window_size"] = args.sliding_window_size
        if args.sliding_window_stride is not None:
            kwargs["sliding_window_stride"] = args.sliding_window_stride
        if args.tea_cache_l1_thresh is not None:
            kwargs["tea_cache_l1_thresh"] = args.tea_cache_l1_thresh
        if args.tea_cache_model_id is not None:
            kwargs["tea_cache_model_id"] = args.tea_cache_model_id
        if args.wantodance_music_path is not None:
            kwargs["wantodance_music_path"] = str(args.wantodance_music_path)
        if args.wantodance_fps is not None:
            kwargs["wantodance_fps"] = args.wantodance_fps
        if args.wantodance_keyframes_mask is not None:
            kwargs["wantodance_keyframes_mask"] = args.wantodance_keyframes_mask
    if spec.pipeline == "MovaAudioVideoPipeline":
        if args.tile_size is not None:
            kwargs["tile_size"] = args.tile_size
        if args.tile_stride is not None:
            kwargs["tile_stride"] = args.tile_stride
    if spec.pipeline == "LTX2AudioVideoPipeline":
        if args.input_images_indexes is not None:
            kwargs["input_images_indexes"] = args.input_images_indexes
        if args.input_images_strength is not None:
            kwargs["input_images_strength"] = args.input_images_strength
        if args.in_context_downsample_factor is not None:
            kwargs["in_context_downsample_factor"] = args.in_context_downsample_factor
        if args.retake_video_regions is not None:
            kwargs["retake_video_regions"] = args.retake_video_regions
        if args.retake_audio_regions is not None:
            kwargs["retake_audio_regions"] = args.retake_audio_regions
        if args.tile_size_in_pixels is not None:
            kwargs["tile_size_in_pixels"] = args.tile_size_in_pixels
        if args.tile_overlap_in_pixels is not None:
            kwargs["tile_overlap_in_pixels"] = args.tile_overlap_in_pixels
        if args.tile_size_in_frames is not None:
            kwargs["tile_size_in_frames"] = args.tile_size_in_frames
        if args.tile_overlap_in_frames is not None:
            kwargs["tile_overlap_in_frames"] = args.tile_overlap_in_frames
        if args.use_two_stage_pipeline:
            kwargs["use_two_stage_pipeline"] = True
        if args.stage2_spatial_upsample_factor is not None:
            kwargs["stage2_spatial_upsample_factor"] = args.stage2_spatial_upsample_factor
        if args.clear_lora_before_stage_two:
            kwargs["clear_lora_before_state_two"] = True
        if args.use_distilled_pipeline:
            kwargs["use_distilled_pipeline"] = True
    return kwargs


def _load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _load_video_frames(path: Path, *, height: int, width: int):
    from diffsynth.utils.data import VideoData

    path = Path(path)
    if path.is_dir():
        data = VideoData(image_folder=str(path), height=height, width=width)
    else:
        data = VideoData(video_file=str(path), height=height, width=width)
    return data.raw_data()


def _load_audio_input(
    path: Path,
    *,
    start_time: float = 0.0,
    duration: float | None = None,
    as_numpy: bool = True,
):
    from diffsynth.utils.data.audio import read_audio

    try:
        waveform, sample_rate = read_audio(
            str(path),
            start_time=start_time,
            duration=duration,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torchcodec":
            raise
        waveform, sample_rate = _load_audio_with_torchaudio(path, start_time=start_time, duration=duration)
    except ImportError as exc:
        if "torchcodec" not in str(exc):
            raise
        waveform, sample_rate = _load_audio_with_torchaudio(path, start_time=start_time, duration=duration)
    if as_numpy and hasattr(waveform, "detach"):
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim > 2:
            waveform = waveform.reshape(-1)
        waveform = waveform.detach().cpu().float().numpy()
    return waveform, int(sample_rate)


def _load_audio_with_torchaudio(
    path: Path,
    *,
    start_time: float = 0.0,
    duration: float | None = None,
):
    import torchaudio

    info = torchaudio.info(str(path))
    sample_rate = int(info.sample_rate)
    frame_offset = max(0, int(start_time * sample_rate))
    num_frames = -1 if duration is None else max(0, int(duration * sample_rate))
    return torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames)


def _parse_method_config(items: list[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method-config '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        config[key] = _parse_value(value)
    return config


def _build_method_config(args, spec) -> Dict[str, Any]:
    user_config = _parse_method_config(args.method_config)
    method_config = default_method_config(
        args.method,
        num_inference_steps=args.num_inference_steps,
        model_family=spec.family,
        model_key=spec.key,
    )
    method_config.update(normalize_method_config(args.method, user_config))
    return method_config


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_int_pair(value: str) -> tuple[int, int]:
    parsed = _parse_sequence_value(value)
    if len(parsed) != 2:
        raise argparse.ArgumentTypeError(f"expected two integers, got {value!r}")
    try:
        return int(parsed[0]), int(parsed[1])
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected two integers, got {value!r}") from exc


def _parse_int_list(value: str) -> list[int]:
    parsed = _parse_sequence_value(value)
    try:
        return [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected integer list, got {value!r}") from exc


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = _parse_sequence_value(value)
    try:
        return tuple(float(item) for item in parsed)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected float tuple, got {value!r}") from exc


def _parse_regions(value: str) -> list[tuple[float, float]]:
    parsed = _parse_value(value)
    if isinstance(parsed, str):
        parsed = [
            _parse_sequence_value(part)
            for part in parsed.split(";")
            if part.strip()
        ]
    if (
        isinstance(parsed, (list, tuple))
        and len(parsed) == 2
        and not isinstance(parsed[0], (list, tuple))
    ):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError(f"expected region list, got {value!r}")

    regions = []
    for region in parsed:
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise argparse.ArgumentTypeError(f"expected region pairs, got {value!r}")
        try:
            regions.append((float(region[0]), float(region[1])))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"expected numeric region pairs, got {value!r}") from exc
    return regions


def _parse_sequence_value(value: str) -> list[Any]:
    parsed = _parse_value(value)
    if isinstance(parsed, str):
        text = parsed.replace("x", ",")
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, tuple):
        return list(parsed)
    if isinstance(parsed, list):
        return parsed
    raise argparse.ArgumentTypeError(f"expected comma-separated or JSON list value, got {value!r}")


def _torch_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _configure_method_runtime_env(method: str) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python_prefix = Path(sys.prefix)
    nvcc = python_prefix / "bin" / "nvcc"
    if "CUDA_HOME" not in os.environ and nvcc.exists():
        os.environ["CUDA_HOME"] = str(python_prefix)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_path = Path(cuda_home)
        os.environ.setdefault("CUDA_PATH", str(cuda_path))
        nvcc_path = cuda_path / "bin" / "nvcc"
        if nvcc_path.exists():
            os.environ.setdefault("CUDACXX", str(nvcc_path))
        bin_path = str(cuda_path / "bin")
        if bin_path not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        lib_paths = [
            str(cuda_path / "lib"),
            str(cuda_path / "targets" / "x86_64-linux" / "lib"),
        ]
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        for lib_path in reversed(lib_paths):
            if lib_path not in ld_library_path.split(os.pathsep):
                ld_library_path = lib_path + os.pathsep + ld_library_path
        os.environ["LD_LIBRARY_PATH"] = ld_library_path.rstrip(os.pathsep)
        os.environ.setdefault(
            "FLASHINFER_EXTRA_LDFLAGS",
            "-L{0}/lib -L{0}/targets/x86_64-linux/lib "
            "-L{0}/lib/stubs -L{0}/targets/x86_64-linux/lib/stubs".format(cuda_home),
        )

    if method != "svoo":
        return
    cache_root = Path(os.environ.get("SVOO_CACHE_ROOT", REPO_ROOT / ".triton_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache_root / "flashinfer"))
    os.environ.setdefault("SVOO_ENABLE_MEM_SAVE", "1")


def _reset_cuda_memory(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(_cuda_device_arg(device))
    except Exception:
        return


def _cuda_memory(device: str) -> Dict[str, Any]:
    if not device.startswith("cuda"):
        return {"device": device, "available": False}
    try:
        import torch

        return {
            "device": device,
            "available": torch.cuda.is_available(),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(_cuda_device_arg(device)) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(_cuda_device_arg(device)) / 1024**3,
        }
    except Exception as exc:
        return {"device": device, "available": False, "error": str(exc)}


def _cuda_device_arg(device: str) -> str | None:
    return None if device == "cuda" else device


def _runtime_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _jsonable(data: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        result[key] = _jsonable_value(key, value)
    return result


def _jsonable_value(key: str, value: Any) -> Any:
    if key.endswith("_image") and hasattr(value, "size"):
        return "<PIL.Image>"
    if isinstance(value, list) and value and hasattr(value[0], "size"):
        return f"<video frames={len(value)}>"
    if key.endswith("_video") and isinstance(value, list):
        return f"<video frames={len(value)}>"
    if key in {"input_audio", "retake_audio"}:
        try:
            samples = len(value)
        except TypeError:
            samples = "unknown"
        return f"<audio samples={samples}>"
    try:
        json.dumps(value)
    except TypeError:
        return f"<{type(value).__module__}.{type(value).__name__}>"
    return value


def _emit_payload(args, payload: Dict[str, Any]) -> None:
    if args.metrics_file:
        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    status = payload.get("status", "unknown")
    model = payload.get("model")
    method = payload.get("method")
    print(f"status={status} backend=diffsynth model={model} method={method}")
    if payload.get("error"):
        print(f"error={payload['error']}")
    if payload.get("output_file"):
        print(f"output_file={payload['output_file']}")


if __name__ == "__main__":
    raise SystemExit(main())
