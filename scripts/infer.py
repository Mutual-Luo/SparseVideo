#!/usr/bin/env python3
#
# SparseVideo inference entrypoint.
#
# This is the only inference script you need. Select the model and sparse
# attention method by command-line arguments:
#
#   python scripts/infer.py --model wan1.3b --method dense
#   python scripts/infer.py --model wan1.3b --method svoo
#   python scripts/infer.py --model hunyuan --method sta
#
# Supported models:
#   wan1.3b    Wan2.1 T2V 1.3B Diffusers
#   wan14b     Wan2.1 T2V 14B Diffusers
#   wan22      Wan2.2 T2V A14B Diffusers
#   hunyuan    HunyuanVideo T2V
#   cogvideox  CogVideoX dense baseline only
#
# Supported methods:
#   dense       Original dense attention baseline.
#   svg1        training_free/Sparse-VideoGen SVG method.
#   svg2        training_free/Sparse-VideoGen second method.
#   spargeattn  training_free/SpargeAttn.
#   radial      training_free/radial-attention.
#   sta         training_free/FastVideo Sliding Tile Attention.
#   draft       training_free/draft-attention.
#   adacluster  training_free/Adacluster.
#   flashomni   training_free/FlashOmni.
#   svoo        training_free/SVOO.
#
# Current sparse support:
#   Wan and Hunyuan support the sparse methods above.
#   CogVideoX is included only for the dense baseline until processors are added.
#
# Common options:
#
#   --dry-run                 Show resolved settings without loading model.
#   --num-frames 81           Override exact frame count.
#   --num-inference-steps 10  Run fewer steps for a smoke test.
#   --prompt "..."            Override prompt from the command line.
#   --prompt-file prompt.txt  Read prompt from a file.
#   --cpu-offload             Use pipeline CPU offload if available.
#
# Defaults:
#   Wan uses 81 frames at 16 fps.
#   Hunyuan uses 129 frames at 24 fps.
#   Outputs go to result/inference/<model>/<method>/.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


METHODS = (
    "dense",
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


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    pipeline_class: str
    hf_id: str
    local_dir: Optional[str]
    fps: int
    default_frames: int
    default_steps: int
    guidance_scale: float
    output_type: str
    sparse_supported: bool = True


MODEL_SPECS: Dict[str, ModelSpec] = {
    "wan21-t2v-1.3b": ModelSpec(
        key="wan21-t2v-1.3b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        local_dir="Wan2.1-T2V-1.3B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan21-t2v-14b": ModelSpec(
        key="wan21-t2v-14b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        local_dir="Wan2.1-T2V-14B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan22-t2v-a14b": ModelSpec(
        key="wan22-t2v-a14b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        local_dir="Wan2.2-T2V-A14B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=40,
        guidance_scale=5.0,
        output_type="np",
    ),
    "hunyuan-t2v": ModelSpec(
        key="hunyuan-t2v",
        family="hunyuan_video",
        pipeline_class="HunyuanVideoPipeline",
        hf_id="tencent/HunyuanVideo",
        local_dir="HunyuanVideo",
        fps=24,
        default_frames=129,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
    ),
    "cogvideox-t2v": ModelSpec(
        key="cogvideox-t2v",
        family="cogvideox",
        pipeline_class="CogVideoXPipeline",
        hf_id="THUDM/CogVideoX-5b",
        local_dir="CogVideoX-5b",
        fps=8,
        default_frames=49,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
        sparse_supported=False,
    ),
}


MODEL_ALIASES = {
    "wan1.3b": "wan21-t2v-1.3b",
    "wan21-1.3b": "wan21-t2v-1.3b",
    "wan21-t2v-1.3b": "wan21-t2v-1.3b",
    "wan14b": "wan21-t2v-14b",
    "wan21-14b": "wan21-t2v-14b",
    "wan21-t2v-14b": "wan21-t2v-14b",
    "wan22": "wan22-t2v-a14b",
    "wan22-a14b": "wan22-t2v-a14b",
    "wan22-t2v-a14b": "wan22-t2v-a14b",
    "hunyuan": "hunyuan-t2v",
    "hunyuan-t2v": "hunyuan-t2v",
    "cog": "cogvideox-t2v",
    "cogvideox": "cogvideox-t2v",
    "cogvideox-t2v": "cogvideox-t2v",
}


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, static, blurred details, subtitles, low quality, worst quality, "
    "jpeg artifacts, deformed, disfigured, extra fingers, bad hands, bad anatomy"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one SparseVideo inference job and write timing metrics.",
        epilog=(
            "Examples:\n"
            "  python scripts/infer.py --model wan1.3b --method dense\n"
            "  python scripts/infer.py --model wan1.3b --method svoo --num-inference-steps 10\n"
            "  python scripts/infer.py --model hunyuan --method radial --prompt-file prompt.txt\n"
            "  python scripts/infer.py --model wan1.3b --method sta --num-frames 81\n"
            "\n"
            "Models: wan1.3b, wan14b, wan22, hunyuan, cogvideox\n"
            "Methods: dense, svg1, svg2, spargeattn, radial, sta, draft, "
            "adacluster, flashomni, svoo\n"
            "Note: sparse methods currently support Wan and Hunyuan; CogVideoX is dense-only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument("--model-root", type=Path, default=Path("/home/dataset-assist-0/luojy/models"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a red sports car driving along a coastal road at sunset, detailed, realistic")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "result" / "inference")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=REPO_ROOT / "result" / "inference" / "metrics.jsonl")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--method-config-json", type=str, default=None)
    parser.add_argument(
        "--method-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra sparse method config. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_method_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if args.method_config_json:
        loaded = json.loads(args.method_config_json)
        if not isinstance(loaded, dict):
            raise ValueError("--method-config-json must decode to an object")
        config.update(loaded)
    for item in args.method_config:
        if "=" not in item:
            raise ValueError(f"Invalid --method-config {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        config[key] = parse_json_value(value)
    return config


def default_num_frames(duration_seconds: float, fps: int) -> int:
    frames = max(1, int(round(duration_seconds * fps)) + 1)
    remainder = (frames - 1) % 4
    if remainder:
        frames += 4 - remainder
    return frames


def resolve_model_id(spec: ModelSpec, model_root: Path, model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    if spec.local_dir:
        local_path = model_root / spec.local_dir
        if local_path.exists():
            return str(local_path.resolve())
    return spec.hf_id


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


def parse_dtype(torch_module, dtype: str):
    if dtype == "bf16":
        return torch_module.bfloat16
    if dtype == "fp16":
        return torch_module.float16
    return torch_module.float32


def load_pipeline(spec: ModelSpec, model_id: str, torch_dtype, local_files_only: bool):
    if spec.pipeline_class == "WanPipeline":
        from diffusers import WanPipeline

        cls = WanPipeline
    elif spec.pipeline_class == "HunyuanVideoPipeline":
        from diffusers import HunyuanVideoPipeline

        cls = HunyuanVideoPipeline
    elif spec.pipeline_class == "CogVideoXPipeline":
        from diffusers import CogVideoXPipeline

        cls = CogVideoXPipeline
    else:
        raise ValueError(f"Unknown pipeline class: {spec.pipeline_class}")

    kwargs = {"torch_dtype": torch_dtype}
    if local_files_only:
        kwargs["local_files_only"] = True
    if spec.pipeline_class == "HunyuanVideoPipeline" and model_id == "tencent/HunyuanVideo":
        kwargs["revision"] = "refs/pr/18"
    return cls.from_pretrained(model_id, **kwargs)


def prepare_pipeline(pipe, device: str, cpu_offload: bool, vae_tiling: bool):
    if vae_tiling and hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    if cpu_offload:
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError("This pipeline does not expose enable_model_cpu_offload()")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)


def build_call_kwargs(
    args: argparse.Namespace,
    spec: ModelSpec,
    prompt: str,
    negative_prompt: str,
    generator,
    num_frames: int,
    fps: int,
) -> Dict[str, Any]:
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else spec.guidance_scale
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": spec.output_type,
    }
    if spec.key == "wan22-t2v-a14b":
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    if spec.family == "hunyuan_video":
        kwargs["true_cfg_scale"] = args.true_cfg_scale
    if spec.family == "cogvideox":
        kwargs["use_dynamic_cfg"] = False
    return kwargs


def make_output_file(args: argparse.Namespace, model: str, method: str, num_frames: int) -> Path:
    if args.output_file is not None:
        return args.output_file
    filename = f"seed{args.seed}_{args.height}x{args.width}_{num_frames}f.mp4"
    return args.output_dir / model / method / filename


def append_metrics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def cuda_memory_gb(torch_module) -> Dict[str, float]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024**3),
        "cuda_peak_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024**3),
    }


def sync_if_cuda(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def run(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[MODEL_ALIASES[args.model]]
    fps = args.fps if args.fps is not None else spec.fps
    if args.num_frames is not None:
        num_frames = args.num_frames
    elif args.duration_seconds is not None:
        num_frames = default_num_frames(args.duration_seconds, fps)
    else:
        num_frames = spec.default_frames
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    method_config = parse_method_config(args)
    model_id = resolve_model_id(spec, args.model_root, args.model_path)
    output_file = make_output_file(args, spec.key, args.method, num_frames)

    base_metrics: Dict[str, Any] = {
        "model": spec.key,
        "model_arg": args.model,
        "model_id": model_id,
        "method": args.method,
        "method_config": method_config,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
        "requested_duration_seconds": args.duration_seconds,
        "num_inference_steps": steps,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "seed": args.seed,
        "output_file": str(output_file),
    }

    unsupported = args.method != "dense" and not spec.sparse_supported

    if args.dry_run:
        base_metrics.update(status="unsupported_dry_run" if unsupported else "dry_run")
        if unsupported:
            base_metrics["error"] = (
                f"{args.method} is not implemented for {spec.family}; "
                "only dense baseline is available."
            )
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 0

    if unsupported:
        base_metrics.update(
            status="unsupported",
            error=f"{args.method} is not implemented for {spec.family}; only dense baseline is available.",
        )
        append_metrics(args.metrics_file, base_metrics)
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 2

    stage = "start"
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    try:
        stage = "import"
        import torch
        from diffusers.utils import export_to_video
        import sparsevideo

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt = read_prompt(args)
        torch_dtype = parse_dtype(torch, args.dtype)

        stage = "load_pipeline"
        t0 = time.perf_counter()
        pipe = load_pipeline(spec, model_id, torch_dtype, args.local_files_only)
        prepare_pipeline(pipe, args.device, args.cpu_offload, args.vae_tiling)
        sync_if_cuda(torch, args.device)
        timings["load_pipeline_sec"] = time.perf_counter() - t0

        stage = "apply_sparse_attention"
        t0 = time.perf_counter()
        handle = sparsevideo.apply_sparse_attention(pipe, method=args.method, config=method_config)
        sync_if_cuda(torch, args.device)
        timings["apply_sparse_attention_sec"] = time.perf_counter() - t0

        stage = "generate"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and output_file.exists():
            base_metrics.update(status="skipped_existing", timings=timings)
            append_metrics(args.metrics_file, base_metrics)
            print(json.dumps(base_metrics, indent=2, sort_keys=True))
            handle.restore()
            return 0

        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        call_kwargs = build_call_kwargs(
            args=args,
            spec=spec,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_frames=num_frames,
            fps=fps,
        )

        t0 = time.perf_counter()
        result = pipe(**call_kwargs)
        sync_if_cuda(torch, args.device)
        timings["generate_sec"] = time.perf_counter() - t0

        stage = "export_video"
        t0 = time.perf_counter()
        export_to_video(result.frames[0], str(output_file), fps=fps)
        timings["export_video_sec"] = time.perf_counter() - t0
        handle.restore()

        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="ok",
            timings=timings,
            seconds_per_frame=timings["generate_sec"] / max(num_frames, 1),
            **cuda_memory_gb(torch),
        )
        append_metrics(args.metrics_file, base_metrics)
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="failed",
            failed_stage=stage,
            timings=timings,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        append_metrics(args.metrics_file, base_metrics)
        traceback.print_exc()
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
