#!/usr/bin/env python3
"""Compare two generated videos with lightweight, reproducible diagnostics.

This is intended for SparseVideo dense-vs-sparse quality checks. It does not
replace human inspection, but it records objective evidence next to inference
metrics: video metadata, sampled-frame PSNR/SSIM/MAE, and an optional
side-by-side contact sheet.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency in some envs.
    structural_similarity = None


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def ffprobe(path: Path) -> dict[str, Any]:
    data = run_json(
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
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": _int_or_none(stream.get("nb_frames")),
        "duration": _float_or_none(stream.get("duration")),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "bit_rate": _int_or_none(stream.get("bit_rate")),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_indices(frame_count: int, samples: int) -> list[int]:
    if frame_count <= 0:
        raise RuntimeError("Cannot sample a video with unknown or zero frame count")
    samples = max(1, min(int(samples), frame_count))
    if samples == 1:
        return [0]
    return sorted({round(i * (frame_count - 1) / (samples - 1)) for i in range(samples)})


def read_sampled_frames(path: Path, metadata: dict[str, Any], indices: list[int], width: int, height: int) -> np.ndarray:
    if not indices:
        raise ValueError("indices must not be empty")
    select_terms = "+".join(f"eq(n\\,{index})" for index in indices)
    vf = f"select='{select_terms}',scale={width}:{height}"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    if len(result.stdout) % frame_bytes != 0:
        raise RuntimeError(f"ffmpeg returned an incomplete rawvideo buffer for {path}")
    count = len(result.stdout) // frame_bytes
    if count != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} sampled frames from {path}, got {count}. "
            f"metadata={metadata}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, height, width, 3)


def frame_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr: float | str = "inf" if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    ssim = None
    if structural_similarity is not None:
        ssim = float(structural_similarity(reference, candidate, channel_axis=2, data_range=255))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def summarize(values: list[float | None]) -> dict[str, float | None]:
    if values and all(value == "inf" for value in values):
        return {"mean": "inf", "min": "inf", "max": "inf"}
    clean = [
        float(value)
        for value in values
        if value not in (None, "inf") and math.isfinite(float(value))
    ]
    if not clean:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def write_contact_sheet(
    path: Path,
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    indices: list[int],
    *,
    max_rows: int,
    thumb_width: int,
) -> None:
    rows = min(max_rows, len(indices))
    ref_frames = reference_frames[:rows]
    cand_frames = candidate_frames[:rows]
    height, width = ref_frames.shape[1:3]
    thumb_height = max(1, round(height * thumb_width / width))
    label_height = 22
    columns = 3
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(rows):
        ref = Image.fromarray(ref_frames[row]).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        cand = Image.fromarray(cand_frames[row]).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        diff_arr = np.abs(ref_frames[row].astype(np.int16) - cand_frames[row].astype(np.int16))
        diff_arr = np.clip(diff_arr * 4, 0, 255).astype(np.uint8)
        diff = Image.fromarray(diff_arr).resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        y = row * (thumb_height + label_height)
        sheet.paste(ref, (0, y + label_height))
        sheet.paste(cand, (thumb_width, y + label_height))
        sheet.paste(diff, (2 * thumb_width, y + label_height))
        draw.text((4, y + 4), f"frame {indices[row]} dense", fill=(0, 0, 0))
        draw.text((thumb_width + 4, y + 4), "sparse", fill=(0, 0, 0))
        draw.text((2 * thumb_width + 4, y + 4), "abs diff x4", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    reference = Path(args.reference)
    candidate = Path(args.candidate)
    if not reference.exists():
        raise FileNotFoundError(reference)
    if not candidate.exists():
        raise FileNotFoundError(candidate)

    ref_meta = ffprobe(reference)
    cand_meta = ffprobe(candidate)
    frame_count = min(ref_meta.get("nb_frames") or 0, cand_meta.get("nb_frames") or 0)
    width = min(ref_meta["width"], cand_meta["width"])
    height = min(ref_meta["height"], cand_meta["height"])
    indices = sample_indices(frame_count, args.samples)

    ref_frames = read_sampled_frames(reference, ref_meta, indices, width, height)
    cand_frames = read_sampled_frames(candidate, cand_meta, indices, width, height)
    per_frame = [
        {"frame": index, **frame_metrics(ref_frame, cand_frame)}
        for index, ref_frame, cand_frame in zip(indices, ref_frames, cand_frames)
    ]
    metrics = {
        "mse": summarize([item["mse"] for item in per_frame]),
        "mae": summarize([item["mae"] for item in per_frame]),
        "psnr": summarize([item["psnr"] for item in per_frame]),
        "ssim": summarize([item["ssim"] for item in per_frame]),
    }

    result = {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_metadata": ref_meta,
        "candidate_metadata": cand_meta,
        "sampled_frames": indices,
        "metrics": metrics,
        "per_frame": per_frame,
        "notes": [
            "PSNR/SSIM compare sampled pixels and do not prove semantic or aesthetic quality by themselves.",
            "Use the contact sheet and mp4 artifacts for human review before claiming no visible degradation.",
        ],
    }
    if args.contact_sheet:
        write_contact_sheet(
            Path(args.contact_sheet),
            ref_frames,
            cand_frames,
            indices,
            max_rows=args.sheet_rows,
            thumb_width=args.thumb_width,
        )
        result["contact_sheet"] = str(args.contact_sheet)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Dense baseline mp4")
    parser.add_argument("--candidate", required=True, help="Sparse candidate mp4")
    parser.add_argument("--samples", type=int, default=16, help="Number of frames to sample uniformly")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--sheet-rows", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=320)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = compare(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
