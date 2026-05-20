#!/usr/bin/env python3
#
# Generate test inputs for SparseVideo backbone/method testing.
#
# Sources used:
#   - portrait.jpg  (I2V):  training_free/FastVideo/assets/girl.png  (real face photo)
#   - character.jpg (WanAnimate): synthetically drawn standing full-body figure
#   - video_ref.mp4:  training_free/FastVideo/assets/motorcycle.mp4  (real footage)
#   - video_mask.mp4: oval sweep mask
#   - pose_video.mp4: DWPose-style COCO-17 standing sway (upper-body motion only,
#                     compatible with a standing full-body reference)
#   - face_video.mp4: face crop from portrait + slight zoom
#
# WanAnimate requires a FULL-BODY STANDING character image.  Using a close-up
# portrait (like the café girl) with a standing-walk pose video is semantically
# wrong; the model cannot reconcile them.  character.jpg is a simple drawn
# standing figure that correctly matches the pose video.
#
# Usage:
#   python scripts/make_test_inputs.py
#   python scripts/make_test_inputs.py --output-dir /path/to/dir
#
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths to real source files
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_PORTRAIT  = _REPO_ROOT / "training_free/FastVideo/assets/girl.png"
_REAL_VIDEO     = _REPO_ROOT / "training_free/FastVideo/assets/motorcycle.mp4"


# ---------------------------------------------------------------------------
# DWPose / COCO-17 skeleton
# ---------------------------------------------------------------------------
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

_JOINT_COLORS = [
    (255, 0, 0),    # 0  nose
    (255, 85, 0),   # 1  l-eye
    (255, 170, 0),  # 2  r-eye
    (255, 255, 0),  # 3  l-ear
    (170, 255, 0),  # 4  r-ear
    (85, 255, 0),   # 5  l-shoulder
    (0, 255, 0),    # 6  r-shoulder
    (0, 255, 85),   # 7  l-elbow
    (0, 255, 170),  # 8  r-elbow
    (0, 255, 255),  # 9  l-wrist
    (0, 170, 255),  # 10 r-wrist
    (0, 85, 255),   # 11 l-hip
    (0, 0, 255),    # 12 r-hip
    (85, 0, 255),   # 13 l-knee
    (170, 0, 255),  # 14 r-knee
    (255, 0, 255),  # 15 l-ankle
    (255, 0, 170),  # 16 r-ankle
]

_BONE_COLOR = (128, 128, 128)


def _get_coco17_keypoints_sway(h: int, w: int, phase: float) -> np.ndarray:
    """
    Standing figure with gentle upper-body sway and arm swing.
    Hips and legs stay nearly still — compatible with a standing character portrait.
    """
    cx = w * 0.5
    # Anchor points (fractions of frame)
    nose_y    = h * 0.10
    sh_y      = h * 0.25
    sh_dx     = w * 0.10
    elbow_y   = h * 0.40
    wrist_y   = h * 0.54
    hip_y     = h * 0.55
    hip_dx    = w * 0.07
    knee_y    = h * 0.73
    ankle_y   = h * 0.92

    sway      = w * 0.015 * math.sin(phase)          # gentle torso sway
    arm_swing = w * 0.04  * math.sin(phase)           # arms swing opposite to sway
    head_tilt = w * 0.008 * math.sin(phase * 0.7)

    kp = np.zeros((17, 2), dtype=np.float32)
    kp[0]  = [cx + head_tilt,              nose_y]
    kp[1]  = [cx - w*0.02  + head_tilt,   nose_y - h*0.02]
    kp[2]  = [cx + w*0.02  + head_tilt,   nose_y - h*0.02]
    kp[3]  = [cx - w*0.04  + head_tilt,   nose_y]
    kp[4]  = [cx + w*0.04  + head_tilt,   nose_y]
    kp[5]  = [cx - sh_dx   + sway,        sh_y]
    kp[6]  = [cx + sh_dx   + sway,        sh_y]
    kp[7]  = [cx - sh_dx   - arm_swing,   elbow_y]
    kp[8]  = [cx + sh_dx   + arm_swing,   elbow_y]
    kp[9]  = [cx - sh_dx   - arm_swing * 1.4, wrist_y]
    kp[10] = [cx + sh_dx   + arm_swing * 1.4, wrist_y]
    kp[11] = [cx - hip_dx,                hip_y]
    kp[12] = [cx + hip_dx,                hip_y]
    kp[13] = [cx - hip_dx  + w*0.01,      knee_y]
    kp[14] = [cx + hip_dx  - w*0.01,      knee_y]
    kp[15] = [cx - hip_dx,                ankle_y]
    kp[16] = [cx + hip_dx,                ankle_y]
    return kp


def _draw_pose_frame(h: int, w: int, phase: float) -> np.ndarray:
    img  = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    kp   = _get_coco17_keypoints_sway(h, w, phase)
    lw   = max(1, w // 120)

    for i, j in _COCO_SKELETON:
        x1, y1 = float(kp[i][0]), float(kp[i][1])
        x2, y2 = float(kp[j][0]), float(kp[j][1])
        draw.line([(x1, y1), (x2, y2)], fill=_BONE_COLOR, width=lw)

    r = max(2, w // 80)
    for idx, (x, y) in enumerate(kp):
        x, y = float(x), float(y)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_JOINT_COLORS[idx])

    return np.array(img)


# ---------------------------------------------------------------------------
# Standing full-body character illustration  (for WanAnimate)
# ---------------------------------------------------------------------------

def _draw_standing_character(size: int = 512) -> np.ndarray:
    """
    Simple drawn full-body standing person on a plain background.
    Proportions match the COCO-17 keypoint layout in the pose video.
    """
    w, h = size, size
    img  = Image.new("RGB", (w, h), (240, 235, 230))   # warm off-white
    draw = ImageDraw.Draw(img)

    cx   = w // 2
    skin = (220, 180, 150)
    hair = (60, 40, 30)
    cloth_top    = (80, 100, 160)
    cloth_bottom = (50, 60, 100)
    shoe_col     = (40, 35, 30)

    # Proportions (matching _get_coco17_keypoints_sway with phase=0)
    nose_y  = int(h * 0.10)
    sh_y    = int(h * 0.25)
    sh_dx   = int(w * 0.10)
    hip_y   = int(h * 0.55)
    hip_dx  = int(w * 0.07)
    knee_y  = int(h * 0.73)
    ankle_y = int(h * 0.92)
    head_r  = int(h * 0.09)

    # Legs
    draw.rectangle([cx - hip_dx - 8, hip_y, cx - hip_dx + 8, ankle_y], fill=cloth_bottom)
    draw.rectangle([cx + hip_dx - 8, hip_y, cx + hip_dx + 8, ankle_y], fill=cloth_bottom)
    # Shoes
    draw.ellipse([cx - hip_dx - 14, ankle_y - 6, cx - hip_dx + 18, ankle_y + 10], fill=shoe_col)
    draw.ellipse([cx + hip_dx - 18, ankle_y - 6, cx + hip_dx + 14, ankle_y + 10], fill=shoe_col)
    # Torso
    draw.polygon(
        [(cx - sh_dx, sh_y), (cx + sh_dx, sh_y),
         (cx + hip_dx + 4, hip_y), (cx - hip_dx - 4, hip_y)],
        fill=cloth_top,
    )
    # Arms (hanging straight at phase=0)
    elbow_y = int(h * 0.40)
    wrist_y = int(h * 0.54)
    draw.rectangle([cx - sh_dx - 7, sh_y, cx - sh_dx + 1, wrist_y], fill=cloth_top)
    draw.rectangle([cx + sh_dx - 1, sh_y, cx + sh_dx + 7, wrist_y], fill=cloth_top)
    # Hands
    draw.ellipse([cx - sh_dx - 10, wrist_y, cx - sh_dx + 4, wrist_y + 14], fill=skin)
    draw.ellipse([cx + sh_dx - 4,  wrist_y, cx + sh_dx + 10, wrist_y + 14], fill=skin)
    # Neck
    draw.rectangle([cx - 7, int(h * 0.17), cx + 7, sh_y], fill=skin)
    # Head circle
    draw.ellipse(
        [cx - head_r, nose_y - head_r, cx + head_r, nose_y + head_r],
        fill=skin,
    )
    # Hair
    draw.chord(
        [cx - head_r, nose_y - head_r, cx + head_r, nose_y + head_r * 0.3],
        start=180, end=0, fill=hair,
    )
    # Eyes
    ey = nose_y - head_r // 4
    draw.ellipse([cx - head_r // 3 - 4, ey - 3, cx - head_r // 3 + 4, ey + 3], fill=(30, 20, 15))
    draw.ellipse([cx + head_r // 3 - 4, ey - 3, cx + head_r // 3 + 4, ey + 3], fill=(30, 20, 15))
    # Smile
    draw.arc([cx - 10, nose_y + 2, cx + 10, nose_y + 12], start=0, end=180, fill=(160, 80, 70), width=2)

    return np.array(img)


# ---------------------------------------------------------------------------
# Mask video
# ---------------------------------------------------------------------------

def _mask_frame(h: int, w: int, t: int, n: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx = int(w * 0.15 + w * 0.7 * t / max(n - 1, 1))
    cy = h // 2
    rx, ry = w // 6, h // 3
    ys, xs = np.ogrid[:h, :w]
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
    frame[mask] = 255
    return frame


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 16) -> None:
    import imageio.v3 as iio
    iio.imwrite(str(path), frames, fps=fps)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def make_portrait(output_dir: Path, size: int = 512) -> Path:
    """Real face photo — for I2V models."""
    path = output_dir / "portrait.jpg"
    if _REAL_PORTRAIT.exists():
        img = Image.open(_REAL_PORTRAIT).convert("RGB").resize((size, size), Image.LANCZOS)
        img.save(path, quality=95)
        print(f"  portrait.jpg         (from {_REAL_PORTRAIT.name}, for I2V)")
    else:
        arr = np.full((size, size, 3), [200, 160, 120], dtype=np.uint8)
        Image.fromarray(arr).save(path, quality=95)
        print(f"  portrait.jpg         (flat fallback, real photo unavailable)")
    return path


def make_character(output_dir: Path, size: int = 512) -> Path:
    """Full-body standing figure — for WanAnimate (must show whole body)."""
    path = output_dir / "character.jpg"
    arr = _draw_standing_character(size)
    Image.fromarray(arr).save(path, quality=95)
    print(f"  character.jpg        (drawn full-body figure, for WanAnimate)")
    return path


def make_reference_video(
    output_dir: Path, frames: int, h: int, w: int, fps: int
) -> Path:
    path = output_dir / "video_ref.mp4"
    if _REAL_VIDEO.exists():
        _sample_video_frames(_REAL_VIDEO, path, frames, h, w, fps)
        print(f"  video_ref.mp4        ({frames} frames from {_REAL_VIDEO.name})")
    else:
        data = []
        for t in range(frames):
            frac  = t / max(frames - 1, 1)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:int(h * 0.6)] = [135, int(180 - 40 * frac), 210]
            frame[int(h * 0.6):] = [80, 100, 60]
            data.append(frame)
        write_mp4(path, data, fps)
        print(f"  video_ref.mp4        (synthetic fallback)")
    return path


def _sample_video_frames(
    src: Path, dst: Path, n_out: int, h: int, w: int, fps: int
) -> None:
    import subprocess, tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True,
        )
        total  = int(probe.stdout.strip())
        indices = [round(i * (total - 1) / max(n_out - 1, 1)) for i in range(n_out)]

        frames = []
        for idx in indices:
            out_path = os.path.join(tmp, f"f{idx:06d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(src),
                 "-vf", f"select='eq(n\\,{idx})',scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                 "-vframes", "1", out_path],
                check=True,
            )
            frames.append(np.array(Image.open(out_path).convert("RGB")))

    write_mp4(dst, frames, fps)


def make_mask_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    data = [_mask_frame(h, w, t, frames) for t in range(frames)]
    path = output_dir / "video_mask.mp4"
    write_mp4(path, data, fps)
    print(f"  video_mask.mp4       ({frames} frames)")
    return path


def make_pose_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    data = []
    for t in range(frames):
        phase = 2 * math.pi * t / max(frames, 1) * 3   # ~3 sway cycles
        data.append(_draw_pose_frame(h, w, phase))
    path = output_dir / "pose_video.mp4"
    write_mp4(path, data, fps)
    print(f"  pose_video.mp4       ({frames} frames, standing sway, COCO-17)")
    return path


def make_face_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    path = output_dir / "face_video.mp4"
    if _REAL_PORTRAIT.exists():
        src  = Image.open(_REAL_PORTRAIT).convert("RGB")
        sw, sh = src.size
        crop = src.crop((sw // 8, 0, sw * 7 // 8, sh // 2))
        data = []
        for t in range(frames):
            zoom = 0.05 * t / max(frames - 1, 1)
            dw   = int(crop.width  * zoom / 2)
            dh   = int(crop.height * zoom / 2)
            zoomed = crop.crop((dw, dh, crop.width - dw, crop.height - dh))
            data.append(np.array(zoomed.resize((w, h), Image.LANCZOS)))
        write_mp4(path, data, fps)
        print(f"  face_video.mp4       ({frames} frames, face crop from {_REAL_PORTRAIT.name})")
    else:
        data = [np.full((h, w, 3), [220, 180, 150], dtype=np.uint8) for _ in range(frames)]
        write_mp4(path, data, fps)
        print(f"  face_video.mp4       ({frames} frames, synthetic fallback)")
    return path


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

README_TEMPLATE = """\
Test inputs generated by scripts/make_test_inputs.py
=====================================================

Files
-----
  portrait.jpg      Real face photo (girl.png) — for I2V models.
  character.jpg     Drawn full-body standing figure — for WanAnimate ONLY.
                    WanAnimate requires a full-body character (not a close-up portrait).
  video_ref.mp4     Short clip sampled from real motorcycle footage — for WanVACE.
  video_mask.mp4    Oval-sweep mask — for WanVACE inpainting.
  pose_video.mp4    DWPose-style COCO-17 standing-sway skeleton — for WanAnimate.
                    Motion: gentle upper-body sway + arm swing (standing, not walking).
  face_video.mp4    Face crop from portrait with slow zoom — for WanAnimate.

WanAnimate note
---------------
  pose_video.mp4 shows a STANDING figure with upper-body sway.
  character.jpg shows a STANDING figure that matches.
  Do NOT use portrait.jpg (café close-up) with this pose video — they are incompatible.

  For high-quality WanAnimate output you need real DWPose-extracted pose videos.
  These synthetic inputs are for pipeline smoke testing (end-to-end run check) only.

Example commands (--num-inference-steps 5 for smoke tests)
----------------------------------------------------------

T2V models:
  python scripts/infer.py --model wan1.3b      --method dense      --num-inference-steps 5
  python scripts/infer.py --model wan14b       --method svoo       --num-inference-steps 5
  python scripts/infer.py --model wan22        --method svg2       --num-inference-steps 5
  python scripts/infer.py --model hunyuan      --method adacluster --num-inference-steps 5
  python scripts/infer.py --model skyreels-v2  --method radial     --num-inference-steps 5
  python scripts/infer.py --model cogvideox    --method dense      --num-inference-steps 5
  python scripts/infer.py --model ltx          --method dense      --num-inference-steps 5
  python scripts/infer.py --model allegro      --method dense      --num-inference-steps 5
  python scripts/infer.py --model mochi        --method dense      --num-inference-steps 5
  python scripts/infer.py --model easyanimate  --method dense      --num-inference-steps 5

I2V models (portrait.jpg = real face photo, good starting frame):
  python scripts/infer.py --model wan14b-i2v    --method svoo  --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model wan22-i2v     --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model hunyuan-i2v   --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model skyreels-i2v  --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model cogvideox-i2v --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model ltx-i2v       --method dense --image test_inputs/portrait.jpg --num-inference-steps 5

WanVACE:
  python scripts/infer.py --model wan-vace --method dense \\
      --reference-video test_inputs/video_ref.mp4 \\
      --mask-video test_inputs/video_mask.mp4 --num-inference-steps 5

WanAnimate (uses character.jpg, not portrait.jpg):
  python scripts/infer.py --model wananimate --method dense \\
      --image test_inputs/character.jpg \\
      --pose-video test_inputs/pose_video.mp4 \\
      --face-video test_inputs/face_video.mp4 \\
      --num-inference-steps 5
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate test inputs for SparseVideo.")
    p.add_argument("--output-dir",    type=Path, default=Path("test_inputs"))
    p.add_argument("--img-size",      type=int,  default=512)
    p.add_argument("--video-size",    type=str,  default="480x272",
                   help="Video WxH, both must be divisible by 16 (default 480x272).")
    p.add_argument("--video-frames",  type=int,  default=77,
                   help="Frame count for WanAnimate videos (default 77).")
    p.add_argument("--short-frames",  type=int,  default=17,
                   help="Frame count for WanVACE short videos (default 17).")
    p.add_argument("--fps",           type=int,  default=16)
    return p


def main() -> None:
    args = build_parser().parse_args()
    w_str, h_str = args.video_size.split("x")
    vw, vh = int(w_str), int(h_str)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing test inputs to {out.resolve()}/")

    make_portrait(out, size=args.img_size)
    make_character(out, size=args.img_size)
    make_reference_video(out, args.short_frames, vh, vw, args.fps)
    make_mask_video(out, args.short_frames, vh, vw, args.fps)
    make_pose_video(out, args.video_frames, vh, vw, args.fps)
    make_face_video(out, args.video_frames, vh, vw, args.fps)

    readme = out / "README.txt"
    readme.write_text(README_TEMPLATE)
    print(f"  README.txt")
    print()
    print("Done. See test_inputs/README.txt for example inference commands.")


if __name__ == "__main__":
    main()
