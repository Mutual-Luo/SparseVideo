#!/usr/bin/env python3
"""Run the official Wan2.2 Animate preprocessing pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WAN2_REPO_URL = "https://github.com/Wan-Video/Wan2.2.git"
WAN_ANIMATE_REPO_ID = "Wan-AI/Wan2.2-Animate-14B"
PREPROCESS_REL = Path("wan/modules/animate/preprocess/preprocess_data.py")

DEFAULT_IMAGE = REPO_ROOT / "example" / "animate" / "1.jpeg"
DEFAULT_VIDEO = REPO_ROOT / "example" / "animate" / "1.mp4"
DEFAULT_OUTPUT = REPO_ROOT / "example" / "animate" / "process_results"
DEFAULT_CKPT = Path("/home/dataset-assist-0/luojy/models/Wan2.2-Animate-14B/process_checkpoint")
DEFAULT_SOURCE_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "sparsevideo" / "Wan2.2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare WanAnimate inputs for Diffusers. The animation-mode outputs "
            "are src_ref.png, src_pose.mp4, and src_face.mp4."
        )
    )
    parser.add_argument("--image", "--refer-path", dest="image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--video", "--video-path", dest="video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", "--save-path", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="Download the required official process_checkpoint files before running.",
    )
    parser.add_argument(
        "--wan2-dir",
        type=Path,
        default=None,
        help="Path to a Wan2.2 source checkout. Defaults to $WAN2_DIR, common local paths, or a cache clone.",
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=DEFAULT_SOURCE_CACHE,
        help="Cache path used when the Wan2.2 source checkout is not found.",
    )
    parser.add_argument(
        "--no-auto-source",
        dest="auto_source",
        action="store_false",
        help="Do not clone Wan2.2 source automatically when it is missing.",
    )
    parser.set_defaults(auto_source=True)

    parser.add_argument("--mode", choices=("animation", "replacement"), default="animation")
    parser.add_argument("--replace", dest="mode", action="store_const", const="replacement")
    parser.add_argument("--resolution-area", type=int, nargs=2, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--w-len", type=int, default=1)
    parser.add_argument("--h-len", type=int, default=1)

    parser.set_defaults(retarget=True, use_flux=True)
    parser.add_argument("--no-retarget", dest="retarget", action="store_false")
    parser.add_argument("--use-flux", dest="use_flux", action="store_true")
    parser.add_argument("--no-use-flux", dest="use_flux", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Print the official command without running it.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _has_preprocess(wan2_dir: Path) -> bool:
    return (wan2_dir / PREPROCESS_REL).is_file()


def _source_candidates(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.wan2_dir is not None:
        candidates.append(args.wan2_dir)
    env_wan2 = os.environ.get("WAN2_DIR")
    if env_wan2:
        candidates.append(Path(env_wan2))
    candidates.extend(
        [
            REPO_ROOT / "third_party" / "Wan2.2",
            REPO_ROOT.parent / "Wan2.2",
            REPO_ROOT.parent.parent / "Wan2.2",
            Path("/home/dataset-assist-0/luojy/Wan2.2"),
            args.source_cache,
        ]
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = _resolve(candidate)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _ensure_wan2_source(args: argparse.Namespace) -> Path:
    for candidate in _source_candidates(args):
        if _has_preprocess(candidate):
            return candidate

    cache_dir = _resolve(args.source_cache)
    if args.dry_run:
        return cache_dir

    if args.auto_source and not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning Wan2.2 source into {cache_dir}", file=sys.stderr)
        subprocess.run(["git", "clone", "--depth", "1", WAN2_REPO_URL, str(cache_dir)], check=True)
        if _has_preprocess(cache_dir):
            return cache_dir

    checked = "\n  ".join(str(path) for path in _source_candidates(args))
    raise SystemExit(
        "Wan2.2 preprocessing source was not found.\n"
        f"Checked:\n  {checked}\n"
        "Pass --wan2-dir /path/to/Wan2.2, set WAN2_DIR, or allow the default source cache clone."
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def _required_checkpoint_paths(ckpt_path: Path, args: argparse.Namespace) -> list[Path]:
    required = [
        ckpt_path / "pose2d" / "vitpose_h_wholebody.onnx",
        ckpt_path / "det" / "yolov10m.onnx",
    ]
    if args.mode == "replacement":
        required.append(ckpt_path / "sam2" / "sam2_hiera_large.pt")
    if args.mode == "animation" and args.use_flux:
        required.append(ckpt_path / "FLUX.1-Kontext-dev")
    return required


def _missing_checkpoint_paths(ckpt_path: Path, args: argparse.Namespace) -> list[Path]:
    return [path for path in _required_checkpoint_paths(ckpt_path, args) if not path.exists()]


def _download_checkpoint(args: argparse.Namespace) -> None:
    if args.ckpt_path.name != "process_checkpoint":
        raise SystemExit("--download-checkpoint expects --ckpt-path to end with process_checkpoint.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface_hub; install it or download process_checkpoint manually.") from exc

    patterns = ["process_checkpoint/pose2d/*", "process_checkpoint/det/*"]
    if args.mode == "replacement":
        patterns.append("process_checkpoint/sam2/*")
    if args.mode == "animation" and args.use_flux:
        patterns.append("process_checkpoint/FLUX.1-Kontext-dev/**")

    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WanAnimate process checkpoint files into {args.ckpt_path.parent}")
    snapshot_download(
        repo_id=WAN_ANIMATE_REPO_ID,
        local_dir=str(args.ckpt_path.parent),
        allow_patterns=patterns,
    )


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.mode == "animation" and args.use_flux and not args.retarget:
        raise SystemExit("--use-flux requires retargeting; remove --no-retarget or pass --no-use-flux.")

    _require_file(args.image, "reference image")
    _require_file(args.video, "driving video")

    missing = _missing_checkpoint_paths(args.ckpt_path, args)
    if args.download_checkpoint and missing and not args.dry_run:
        _download_checkpoint(args)
        missing = _missing_checkpoint_paths(args.ckpt_path, args)

    if not args.ckpt_path.is_dir():
        if args.dry_run:
            print(f"Warning: missing WanAnimate process checkpoint directory: {args.ckpt_path}", file=sys.stderr)
            return
        raise SystemExit(
            f"Missing WanAnimate process checkpoint directory: {args.ckpt_path}\n"
            "Expected the official Wan2.2 Animate process_checkpoint directory. "
            "Use --download-checkpoint to fetch the required files."
        )

    if missing:
        details = "\n  ".join(str(path) for path in missing)
        hint = ""
        if args.mode == "animation" and args.use_flux:
            hint = "\nFor a non-FLUX local animation preprocess, rerun with --no-use-flux."
        if args.dry_run:
            print(f"Warning: missing WanAnimate checkpoint files:\n  {details}{hint}", file=sys.stderr)
            return
        raise SystemExit(f"Missing WanAnimate checkpoint files:\n  {details}{hint}\nUse --download-checkpoint to fetch them.")


def _build_command(args: argparse.Namespace, preprocess_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(preprocess_path),
        "--ckpt_path",
        str(args.ckpt_path),
        "--video_path",
        str(args.video),
        "--refer_path",
        str(args.image),
        "--save_path",
        str(args.output),
        "--resolution_area",
        str(args.resolution_area[0]),
        str(args.resolution_area[1]),
        "--fps",
        str(args.fps),
    ]
    if args.mode == "replacement":
        command.extend(
            [
                "--replace_flag",
                "--iterations",
                str(args.iterations),
                "--k",
                str(args.k),
                "--w_len",
                str(args.w_len),
                "--h_len",
                str(args.h_len),
            ]
        )
    else:
        if args.retarget:
            command.append("--retarget_flag")
        if args.use_flux:
            command.append("--use_flux")
    return command


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _expected_outputs(output: Path, mode: str) -> list[Path]:
    outputs = [output / "src_ref.png", output / "src_pose.mp4", output / "src_face.mp4"]
    if mode == "replacement":
        outputs.extend([output / "src_bg.mp4", output / "src_mask.mp4"])
    return outputs


def main() -> None:
    args = _parse_args()
    args.image = _resolve(args.image)
    args.video = _resolve(args.video)
    args.output = _resolve(args.output)
    args.ckpt_path = _resolve(args.ckpt_path)

    wan2_dir = _ensure_wan2_source(args)
    preprocess_path = wan2_dir / PREPROCESS_REL
    preprocess_dir = preprocess_path.parent
    _validate_inputs(args)

    command = _build_command(args, preprocess_path)
    print(f"Wan2.2 source: {wan2_dir}")
    print(f"Working directory: {preprocess_dir}")
    print(f"Command: {_format_command(command)}")
    print("Expected outputs:")
    for path in _expected_outputs(args.output, args.mode):
        print(f"  {path}")
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    py_paths = [str(preprocess_dir), str(wan2_dir)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    subprocess.run(command, cwd=preprocess_dir, env=env, check=True)
    missing = [path for path in _expected_outputs(args.output, args.mode) if not path.exists()]
    if missing:
        details = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"WanAnimate preprocess finished, but expected outputs are missing:\n  {details}")


if __name__ == "__main__":
    main()
