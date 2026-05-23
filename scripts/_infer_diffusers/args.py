from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .models import DEFAULT_HEIGHT, DEFAULT_SEED, DEFAULT_WIDTH, MODEL_ALIASES

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one SparseVideo inference job and write timing metrics.",
        epilog=(
            "Examples:\n"
            "  python scripts/infer.py --model wan1.3b --method dense\n"
            "  python scripts/infer.py --model wan1.3b --method svoo --num-inference-steps 10\n"
            "  python scripts/infer.py --model hunyuan --method radial --prompt-file prompt.txt\n"
            "  python scripts/infer.py --model wan14b --method sta --profile upstream\n"
            "  python scripts/infer.py --model wan14b-i2v --method svoo --image input.jpg\n"
            "  python scripts/infer.py --model hunyuan-i2v --method radial --image input.png\n"
            "\n"
            "Models: wan1.3b, wan14b, wan22, wan14b-i2v, wan22-i2v, "
            "skyreels-v2, skyreels-v2-i2v, wananimate, wan-vace, hunyuan, "
            "hunyuan-i2v, cogvideox, cogvideox-i2v, ltx, ltx-i2v, allegro, mochi, "
            "easyanimate, motif-video, ltx-video-2, sana-video, kandinsky5\n"
            "Methods: dense, svg1, svg2, spargeattn, radial, sta, draft, "
            "adacluster, flashomni, svoo\n"
            "Note: sparse methods support Wan-family/SkyReels and Hunyuan (T2V and I2V); "
            "CogVideoX T2V/I2V, LTX, Allegro, Mochi, and EasyAnimate currently use the guarded "
            "validated matrix from sparsevideo._support. "
            "MotifVideo and LTX Video 2 are unknown in this Diffusers install; "
            "SanaVideo is incompatible; Kandinsky5 is native-N/A.\n"
            "I2V models require --image <path>."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("SPARSEVIDEO_MODEL_ROOT", "/home/dataset-assist-0/public-models")),
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a red sports car driving along a coastal road at sunset, detailed, realistic")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--image", type=str, default=None, help="Input image path. Required for *-i2v models and WanAnimate (reference character).")
    parser.add_argument("--pose-video", type=str, default=None, help="Pose video path. Required for WanAnimate.")
    parser.add_argument("--face-video", type=str, default=None, help="Face video path. Required for WanAnimate.")
    parser.add_argument("--reference-video", type=str, default=None, help="Reference/input video path for WanVACE (optional).")
    parser.add_argument("--mask-video", type=str, default=None, help="Mask video path for WanVACE (optional).")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument(
        "--profile",
        choices=("default", "upstream"),
        default="default",
        help="default keeps SparseVideo's normal 720p comparison shape; upstream uses method benchmark shapes and fails if none is defined.",
    )
    parser.add_argument(
        "--upstream-profile",
        action="store_const",
        dest="profile",
        const="upstream",
        help="Alias for --profile upstream.",
    )
    parser.add_argument(
        "--profile-for-method",
        choices=METHODS,
        default=None,
        help=(
            "Resolve --profile using another method's upstream benchmark profile. "
            "Use this for dense baselines, e.g. --method dense --profile upstream "
            "--profile-for-method draft. Sparse method_config from that profile is "
            "not applied unless it matches --method."
        ),
    )
    parser.add_argument("--height", type=int, default=None, help=f"Output height. Default: {DEFAULT_HEIGHT}.")
    parser.add_argument("--width", type=int, default=None, help=f"Output width. Default: {DEFAULT_WIDTH}.")
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=None, help="Wan/Hunyuan scheduler flow_shift. Wan defaults to 5.0 for 720p, 3.0 below 720p.")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--vae-dtype",
        choices=("bf16", "fp16", "fp32"),
        default=None,
        help="Wan VAE dtype. Default: fp32; upstream profiles may override it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed. Default: {DEFAULT_SEED}; --profile upstream may set the upstream example seed.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use pipeline CPU offload. Default: false, unless --profile upstream sets an upstream offload policy.",
    )
    parser.add_argument(
        "--cpu-offload-mode",
        choices=("model", "sequential"),
        default=None,
        help="CPU offload API to use when CPU offload is enabled. Default: model, unless the profile sets sequential.",
    )
    parser.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable VAE tiling. Default: false, unless --profile upstream sets the reference script policy.",
    )
    parser.add_argument(
        "--vae-slicing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable VAE slicing. Default: false, unless --profile upstream sets the reference script policy.",
    )
    parser.add_argument("--vae-decoder-chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "result" / "inference")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=REPO_ROOT / "result" / "inference" / "metrics.jsonl")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full metrics/config/runtime JSON to stdout. By default stdout is a concise summary.",
    )
    parser.add_argument(
        "--verbose-compile-logs",
        action="store_true",
        help="Show TorchInductor/Triton compile and autotune diagnostics. Default suppresses non-fatal kernel-choice noise.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help=(
            "Run with output_type=latent and skip mp4 export. Use only for "
            "backend/kernel dispatch smoke; it is not visual quality evidence."
        ),
    )
    parser.add_argument(
        "--strict-kernels",
        action="store_true",
        help="Keep strict fallback checks enabled. This is the default and overrides --allow-debug-fallbacks.",
    )
    parser.add_argument(
        "--allow-debug-fallbacks",
        action="store_true",
        help="Allow explicitly labeled non-parity fallback kernel paths for smoke/debug runs only.",
    )
    parser.add_argument("--method-config-json", type=str, default=None)
    parser.add_argument(
        "--method-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra sparse method config, e.g. top_p_kmeans=0.9 for SVOO. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate resolved settings without loading model. Add --print-json to show the full resolved payload.",
    )
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


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt
