#!/usr/bin/env python3
#
# Generate test inputs for SparseVideo backbone/method testing.
#
# ─────────────────────────────────────────────────────────────────────────────
# File purposes and consistency requirements
# ─────────────────────────────────────────────────────────────────────────────
#
#  portrait.jpg     → I2V models (--image)
#                     The STARTING FRAME the model extends into a video.
#                     Any real scene works.  Uses girl.png (real café photo).
#
#  character.jpg    → WanAnimate (--image)
#                     The CHARACTER WHOSE APPEARANCE is animated.
#                     Must be a FULL-BODY STANDING person so the model has all
#                     body parts to animate.  We draw a standing figure.
#
#  pose_video.mp4   → WanAnimate (--pose-video)
#                     DWPose-style COCO-17 skeleton frames showing the motion
#                     to apply.  Must be:
#                       1. From the same "driver" as face_video (temporally
#                          aligned frame-for-frame).
#                       2. Full-body standing motion (matching character.jpg).
#                     We derive it from the same animation timeline as the
#                     other WanAnimate files.
#
#  face_video.mp4   → WanAnimate (--face-video)
#                     Face-identity video of the DRIVER (not the character).
#                     The model maps the driver's face expressions onto the
#                     character.  Must be frame-for-frame aligned with
#                     pose_video so the face matches the body motion at each t.
#                     We render the face region of the same drawn figure at
#                     the exact same animation phases as pose_video.
#
#  video_ref.mp4    → WanVACE (--reference-video)
#                     The SOURCE SCENE to edit/continue.  Uses real motorcycle
#                     footage (17 frames sampled uniformly).
#
#  video_mask.mp4   → WanVACE (--mask-video)
#                     Binary mask: WHITE = regenerate, BLACK = keep.
#                     Must have the same resolution and frame count as
#                     video_ref.mp4.  An oval that sweeps left→right defines
#                     a moving inpaint region over the scene.
#
# ─────────────────────────────────────────────────────────────────────────────
# WanAnimate note
# ─────────────────────────────────────────────────────────────────────────────
# For high-quality output you need a REAL driving video of a standing person +
# DWPose extraction.  No such video exists on this machine, so all three files
# (character / pose / face) are derived from the same synthetic standing figure
# at the same animation phases, making them mutually consistent for a pipeline
# smoke test.  Replace with real inputs for quality evaluation.
#
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_REAL_PORTRAIT = _REPO_ROOT / "training_free/FastVideo/assets/girl.png"
_REAL_VIDEO    = _REPO_ROOT / "training_free/FastVideo/assets/motorcycle.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# Shared animation timeline
# ─────────────────────────────────────────────────────────────────────────────
# All WanAnimate files (character / pose / face) derive from the same set of
# animation phases.  At frame t the phase is phases[t].  character.jpg uses
# phase=0 (neutral standing).

def _make_phases(n_frames: int, n_cycles: float = 3.0) -> list[float]:
    return [2 * math.pi * t / max(n_frames, 1) * n_cycles for t in range(n_frames)]


# ─────────────────────────────────────────────────────────────────────────────
# Body geometry  (all coordinates as fractions of frame W / H)
# ─────────────────────────────────────────────────────────────────────────────

def _body_at_phase(phase: float) -> dict:
    """Return a dict of landmark positions (fractions of W, H) for a given animation phase."""
    sway      = 0.015 * math.sin(phase)
    arm_swing = 0.04  * math.sin(phase)
    head_tilt = 0.008 * math.sin(phase * 0.7)
    chin_bob  = 0.005 * abs(math.sin(phase))     # very subtle vertical bob

    return dict(
        # face/head
        nose_xf   = 0.50 + head_tilt,
        nose_yf   = 0.10 + chin_bob,
        head_rf   = 0.09,
        # shoulders
        ls_xf = 0.50 - 0.10 + sway,   ls_yf = 0.25,
        rs_xf = 0.50 + 0.10 + sway,   rs_yf = 0.25,
        # elbows
        le_xf = 0.50 - 0.10 - arm_swing,   le_yf = 0.40,
        re_xf = 0.50 + 0.10 + arm_swing,   re_yf = 0.40,
        # wrists
        lw_xf = 0.50 - 0.10 - arm_swing * 1.4,  lw_yf = 0.54,
        rw_xf = 0.50 + 0.10 + arm_swing * 1.4,  rw_yf = 0.54,
        # hips (stable)
        lh_xf = 0.50 - 0.07,  lh_yf = 0.55,
        rh_xf = 0.50 + 0.07,  rh_yf = 0.55,
        # knees / ankles (stable)
        lk_xf = 0.49,  lk_yf = 0.73,
        rk_xf = 0.51,  rk_yf = 0.73,
        la_xf = 0.49,  la_yf = 0.92,
        ra_xf = 0.51,  ra_yf = 0.92,
    )


def _px(frac_x: float, frac_y: float, w: int, h: int):
    return (int(frac_x * w), int(frac_y * h))


# ─────────────────────────────────────────────────────────────────────────────
# DWPose-style skeleton frame (black bg + COCO-17 joints + bones)
# ─────────────────────────────────────────────────────────────────────────────

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
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),
    (85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),
    (0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),
    (255,0,255),(255,0,170),
]

_BONE_COLOR = (128, 128, 128)


def _keypoints_from_body(b: dict, w: int, h: int) -> list[tuple[int, int]]:
    """17 COCO keypoints in pixel space from a body-at-phase dict."""
    cx = int(0.50 * w)
    nose  = _px(b["nose_xf"], b["nose_yf"], w, h)
    l_eye = (cx - int(0.02 * w) + int(b["head_tilt"] if "head_tilt" in b else 0),
             int((b["nose_yf"] - 0.02) * h))
    # recompute simply from the dict
    nx, ny = int(b["nose_xf"] * w), int(b["nose_yf"] * h)
    hr     = int(b["head_rf"] * h)
    return [
        (nx, ny),                                # 0  nose
        (nx - int(0.02*w), ny - int(0.02*h)),   # 1  l-eye
        (nx + int(0.02*w), ny - int(0.02*h)),   # 2  r-eye
        (nx - int(0.04*w), ny),                  # 3  l-ear
        (nx + int(0.04*w), ny),                  # 4  r-ear
        _px(b["ls_xf"], b["ls_yf"], w, h),       # 5  l-shoulder
        _px(b["rs_xf"], b["rs_yf"], w, h),       # 6  r-shoulder
        _px(b["le_xf"], b["le_yf"], w, h),       # 7  l-elbow
        _px(b["re_xf"], b["re_yf"], w, h),       # 8  r-elbow
        _px(b["lw_xf"], b["lw_yf"], w, h),       # 9  l-wrist
        _px(b["rw_xf"], b["rw_yf"], w, h),       # 10 r-wrist
        _px(b["lh_xf"], b["lh_yf"], w, h),       # 11 l-hip
        _px(b["rh_xf"], b["rh_yf"], w, h),       # 12 r-hip
        _px(b["lk_xf"], b["lk_yf"], w, h),       # 13 l-knee
        _px(b["rk_xf"], b["rk_yf"], w, h),       # 14 r-knee
        _px(b["la_xf"], b["la_yf"], w, h),       # 15 l-ankle
        _px(b["ra_xf"], b["ra_yf"], w, h),       # 16 r-ankle
    ]


def _draw_pose_frame(w: int, h: int, phase: float) -> np.ndarray:
    b   = _body_at_phase(phase)
    kp  = _keypoints_from_body(b, w, h)
    img  = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    lw   = max(1, w // 120)
    for i, j in _COCO_SKELETON:
        draw.line([kp[i], kp[j]], fill=_BONE_COLOR, width=lw)
    r = max(2, w // 80)
    for idx, (x, y) in enumerate(kp):
        draw.ellipse([x-r, y-r, x+r, y+r], fill=_JOINT_COLORS[idx])
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# Full-body character illustration  (same geometry as pose video at phase=0)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_character(size: int = 512, phase: float = 0.0) -> np.ndarray:
    w, h = size, size
    img  = Image.new("RGB", (w, h), (235, 230, 225))
    draw = ImageDraw.Draw(img)
    b    = _body_at_phase(phase)

    skin  = (220, 180, 150)
    hair  = (55, 38, 28)
    shirt = (75, 95, 160)
    pants = (45, 55, 95)
    shoes = (38, 32, 28)

    ls = _px(b["ls_xf"], b["ls_yf"], w, h)
    rs = _px(b["rs_xf"], b["rs_yf"], w, h)
    le = _px(b["le_xf"], b["le_yf"], w, h)
    re = _px(b["re_xf"], b["re_yf"], w, h)
    lw_pt = _px(b["lw_xf"], b["lw_yf"], w, h)
    rw_pt = _px(b["rw_xf"], b["rw_yf"], w, h)
    # Hips wider apart, knees and ankles also separate left/right properly
    lh = _px(b["lh_xf"], b["lh_yf"], w, h)
    rh = _px(b["rh_xf"], b["rh_yf"], w, h)
    lk = (lh[0] + int(w * 0.01), _px(b["lk_xf"], b["lk_yf"], w, h)[1])
    rk = (rh[0] - int(w * 0.01), _px(b["rk_xf"], b["rk_yf"], w, h)[1])
    la = (lh[0], _px(b["la_xf"], b["la_yf"], w, h)[1])
    ra = (rh[0], _px(b["ra_xf"], b["ra_yf"], w, h)[1])
    nx = int(b["nose_xf"] * w)
    ny = int(b["nose_yf"] * h)
    hr = int(b["head_rf"] * h)

    aw = max(7, w // 50)

    # Legs (each leg stays under its hip)
    draw.line([lh, lk, la], fill=pants, width=aw)
    draw.line([rh, rk, ra], fill=pants, width=aw)
    # Shoes
    draw.ellipse([la[0]-14, la[1]-4, la[0]+18, la[1]+10], fill=shoes)
    draw.ellipse([ra[0]-18, ra[1]-4, ra[0]+14, ra[1]+10], fill=shoes)
    # Torso
    draw.polygon([ls, rs, rh, lh], fill=shirt)
    # Arms
    draw.line([ls, le, lw_pt], fill=shirt, width=aw - 1)
    draw.line([rs, re, rw_pt], fill=shirt, width=aw - 1)
    # Hands
    draw.ellipse([lw_pt[0]-9, lw_pt[1], lw_pt[0]+5, lw_pt[1]+14], fill=skin)
    draw.ellipse([rw_pt[0]-5, rw_pt[1], rw_pt[0]+9, rw_pt[1]+14], fill=skin)
    # Neck (centered column, not a V)
    nk_x0, nk_x1 = nx - 6, nx + 6
    nk_y0 = ny + hr - 4
    nk_y1 = ls[1]
    draw.rectangle([nk_x0, nk_y0, nk_x1, nk_y1], fill=skin)
    # Head
    draw.ellipse([nx-hr, ny-hr, nx+hr, ny+hr], fill=skin)
    # Hair
    draw.chord([nx-hr, ny-hr, nx+hr, ny+int(hr*0.2)], start=180, end=0, fill=hair)
    # Eyes (small dots, not sunglasses)
    ey  = ny - hr // 5
    er  = max(2, hr // 8)
    draw.ellipse([nx-hr//3-er, ey-er, nx-hr//3+er, ey+er], fill=(28, 18, 12))
    draw.ellipse([nx+hr//3-er, ey-er, nx+hr//3+er, ey+er], fill=(28, 18, 12))
    # Smile
    draw.arc([nx-9, ny + hr//6, nx+9, ny + hr//2], start=10, end=170, fill=(150, 75, 65), width=2)

    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# Face frame: crop + slight head-tilt motion (in sync with phase)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_face_frame(size_w: int, size_h: int, phase: float,
                     char_size: int = 512) -> np.ndarray:
    """
    Render the character's head region at this phase and resize to (size_w, size_h).
    The head position changes with phase, so the face video is temporally aligned
    with the pose video.
    """
    b  = _body_at_phase(phase)
    nx = int(b["nose_xf"] * char_size)
    ny = int(b["nose_yf"] * char_size)
    hr = int(b["head_rf"] * char_size)

    char_arr = _draw_character(char_size, phase)
    char_img = Image.fromarray(char_arr)

    pad = int(hr * 1.6)
    x0  = max(0, nx - pad)
    y0  = max(0, ny - pad)
    x1  = min(char_size, nx + pad)
    y1  = min(char_size, ny + pad)
    face_crop = char_img.crop((x0, y0, x1, y1))
    return np.array(face_crop.resize((size_w, size_h), Image.LANCZOS))


# ─────────────────────────────────────────────────────────────────────────────
# Mask video (for WanVACE)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_frame(h: int, w: int, t: int, n: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cx = int(w * 0.15 + w * 0.7 * t / max(n - 1, 1))
    cy = h // 2
    ys, xs = np.ogrid[:h, :w]
    mask = ((xs - cx) / (w // 6)) ** 2 + ((ys - cy) / (h // 3)) ** 2 <= 1
    frame[mask] = 255
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Video writer
# ─────────────────────────────────────────────────────────────────────────────

def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 16) -> None:
    import imageio.v3 as iio
    iio.imwrite(str(path), frames, fps=fps)


# ─────────────────────────────────────────────────────────────────────────────
# Public outputs
# ─────────────────────────────────────────────────────────────────────────────

def make_portrait(output_dir: Path, size: int = 512) -> Path:
    """Real face photo for I2V models — just the starting frame, can be any scene."""
    path = output_dir / "portrait.jpg"
    if _REAL_PORTRAIT.exists():
        img = Image.open(_REAL_PORTRAIT).convert("RGB").resize((size, size), Image.LANCZOS)
        img.save(path, quality=95)
        print(f"  portrait.jpg    real photo ({_REAL_PORTRAIT.name}) — for I2V models")
    else:
        Image.fromarray(np.full((size, size, 3), [200, 160, 120], dtype=np.uint8)).save(path, quality=95)
        print(f"  portrait.jpg    flat fallback — for I2V models")
    return path


def make_character(output_dir: Path, size: int = 512) -> Path:
    """Full-body standing figure for WanAnimate — same geometry as pose_video at phase=0."""
    path = output_dir / "character.jpg"
    Image.fromarray(_draw_character(size, phase=0.0)).save(path, quality=95)
    print(f"  character.jpg   full-body standing figure (phase=0) — for WanAnimate --image")
    return path


def make_reference_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    """Real footage for WanVACE — the source scene to edit."""
    path = output_dir / "video_ref.mp4"
    if _REAL_VIDEO.exists():
        _sample_video_frames(_REAL_VIDEO, path, frames, h, w, fps)
        print(f"  video_ref.mp4   {frames} frames from {_REAL_VIDEO.name} — for WanVACE --reference-video")
    else:
        data = []
        for t in range(frames):
            frac  = t / max(frames - 1, 1)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:int(h*0.6)] = [135, int(180-40*frac), 210]
            frame[int(h*0.6):] = [80, 100, 60]
            data.append(frame)
        write_mp4(path, data, fps)
        print(f"  video_ref.mp4   synthetic fallback — for WanVACE --reference-video")
    return path


def _sample_video_frames(src: Path, dst: Path, n_out: int, h: int, w: int, fps: int) -> None:
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        probe  = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True,
        )
        total   = int(probe.stdout.strip())
        indices = [round(i * (total-1) / max(n_out-1, 1)) for i in range(n_out)]
        frames  = []
        for idx in indices:
            out_p = os.path.join(tmp, f"f{idx:06d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-vf", f"select='eq(n\\,{idx})',scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                 "-vframes", "1", out_p],
                check=True,
            )
            frames.append(np.array(Image.open(out_p).convert("RGB")))
    write_mp4(dst, frames, fps)


def make_mask_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    """Oval-sweep mask for WanVACE — same resolution and frame count as video_ref."""
    data = [_mask_frame(h, w, t, frames) for t in range(frames)]
    path = output_dir / "video_mask.mp4"
    write_mp4(path, data, fps)
    print(f"  video_mask.mp4  {frames} frames, oval sweep — for WanVACE --mask-video (pairs with video_ref)")
    return path


def make_pose_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    """
    DWPose-style skeleton animation — for WanAnimate --pose-video.
    Uses the same animation phases as face_video so they are temporally aligned.
    Frame t shows the skeleton at phases[t], which matches face_video frame t.
    """
    phases = _make_phases(frames)
    data   = [_draw_pose_frame(w, h, p) for p in phases]
    path   = output_dir / "pose_video.mp4"
    write_mp4(path, data, fps)
    print(f"  pose_video.mp4  {frames} frames, COCO-17 sway — for WanAnimate --pose-video (aligned with face_video)")
    return path


def make_face_video(output_dir: Path, frames: int, h: int, w: int, fps: int) -> Path:
    """
    Face of the same figure used in pose_video and character.jpg.
    Frame t = face crop of character at phases[t], so it is temporally aligned
    with pose_video: when the skeleton sways, the face sways too.
    """
    phases = _make_phases(frames)
    data   = [_draw_face_frame(w, h, p) for p in phases]
    path   = output_dir / "face_video.mp4"
    write_mp4(path, data, fps)
    print(f"  face_video.mp4  {frames} frames, face at same phases as pose — for WanAnimate --face-video (aligned with pose_video)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# README
# ─────────────────────────────────────────────────────────────────────────────

README_TEMPLATE = """\
Test inputs  —  scripts/make_test_inputs.py
===========================================

FILE PURPOSES AND CONSISTENCY
------------------------------

portrait.jpg
  Used by:  I2V models  (--image)
  Role:     STARTING FRAME that the model extends into a video.
  Content:  Real photo (girl.png) — any interesting image works here.
  Pairs with: nothing specific; just needs to be a real scene.

character.jpg
  Used by:  WanAnimate  (--image)
  Role:     CHARACTER APPEARANCE — the person to animate.
  Content:  Full-body STANDING figure drawn at animation phase=0.
  Must match: pose_video.mp4 — same body geometry and standing pose.
  NOTE: I2V portrait (girl at café) must NOT be used here; she is seated
        and cannot perform the standing skeleton poses in pose_video.

pose_video.mp4
  Used by:  WanAnimate  (--pose-video)
  Role:     MOTION DRIVER — skeleton of the driving person's movements.
  Content:  DWPose COCO-17 skeleton, standing sway, 77 frames.
  Must match: face_video.mp4 — SAME animation phases, frame-for-frame.
              character.jpg — standing body geometry matches.

face_video.mp4
  Used by:  WanAnimate  (--face-video)
  Role:     FACE IDENTITY of the driver — whose expressions to project.
  Content:  Face crop of the same drawn figure at same animation phases.
  Must match: pose_video.mp4 — SAME phases, so face moves with the body.
  NOTE: Uses the drawn character's face, not the real girl from portrait.jpg.

video_ref.mp4
  Used by:  WanVACE  (--reference-video)
  Role:     SOURCE SCENE to edit.
  Content:  17 frames sampled from real motorcycle footage.
  Pairs with: video_mask.mp4 — same resolution (480×272) and frame count (17).

video_mask.mp4
  Used by:  WanVACE  (--mask-video)
  Role:     EDIT REGION — white=regenerate, black=keep.
  Content:  Oval sweeping left→right defines a moving inpaint region.
  Pairs with: video_ref.mp4 — same resolution and frame count.

KNOWN LIMITATION
----------------
For high-quality WanAnimate output you need a REAL driving video of a standing
person, DWPose-extracted pose_video, and a real face track.  No suitable real
driving video is available on this machine.  The three WanAnimate files are
internally consistent (all from the same drawn figure / same phases) but are
synthetic — suitable for pipeline smoke testing only.

EXAMPLE COMMANDS  (--num-inference-steps 5 = smoke test)
---------------------------------------------------------

T2V:
  python scripts/infer.py --model wan1.3b     --method dense      --num-inference-steps 5
  python scripts/infer.py --model wan14b      --method svoo       --num-inference-steps 5
  python scripts/infer.py --model wan22       --method svg2       --num-inference-steps 5
  python scripts/infer.py --model hunyuan     --method adacluster --num-inference-steps 5
  python scripts/infer.py --model skyreels-v2 --method radial     --num-inference-steps 5
  python scripts/infer.py --model cogvideox   --method dense      --num-inference-steps 5
  python scripts/infer.py --model ltx         --method dense      --num-inference-steps 5
  python scripts/infer.py --model allegro     --method dense      --num-inference-steps 5
  python scripts/infer.py --model mochi       --method dense      --num-inference-steps 5
  python scripts/infer.py --model easyanimate --method dense      --num-inference-steps 5

I2V  (portrait.jpg = real starting frame):
  python scripts/infer.py --model wan14b-i2v    --method svoo  --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model wan22-i2v     --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model hunyuan-i2v   --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model skyreels-i2v  --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model cogvideox-i2v --method dense --image test_inputs/portrait.jpg --num-inference-steps 5
  python scripts/infer.py --model ltx-i2v       --method dense --image test_inputs/portrait.jpg --num-inference-steps 5

WanVACE  (video_ref + video_mask are paired):
  python scripts/infer.py --model wan-vace --method dense \\
      --reference-video test_inputs/video_ref.mp4 \\
      --mask-video test_inputs/video_mask.mp4 --num-inference-steps 5

WanAnimate  (character.jpg + pose_video + face_video are all aligned):
  python scripts/infer.py --model wananimate --method dense \\
      --image test_inputs/character.jpg \\
      --pose-video test_inputs/pose_video.mp4 \\
      --face-video test_inputs/face_video.mp4 \\
      --num-inference-steps 5
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate test inputs for SparseVideo.")
    p.add_argument("--output-dir",   type=Path, default=Path("test_inputs"))
    p.add_argument("--img-size",     type=int,  default=512)
    p.add_argument("--video-size",   type=str,  default="480x272",
                   help="WxH, both divisible by 16 (default 480x272).")
    p.add_argument("--video-frames", type=int,  default=77,
                   help="Frames for WanAnimate files (default 77).")
    p.add_argument("--short-frames", type=int,  default=17,
                   help="Frames for WanVACE files (default 17).")
    p.add_argument("--fps",          type=int,  default=16)
    return p


def main() -> None:
    args = build_parser().parse_args()
    w_str, h_str = args.video_size.split("x")
    vw, vh = int(w_str), int(h_str)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing test inputs to {out.resolve()}/\n")

    make_portrait(out, size=args.img_size)
    make_character(out, size=args.img_size)
    make_reference_video(out, args.short_frames, vh, vw, args.fps)
    make_mask_video(out, args.short_frames, vh, vw, args.fps)
    make_pose_video(out, args.video_frames, vh, vw, args.fps)
    make_face_video(out, args.video_frames, vh, vw, args.fps)

    (out / "README.txt").write_text(README_TEMPLATE)
    print(f"\n  README.txt")
    print("\nDone. See test_inputs/README.txt for exact file purposes and example commands.")


if __name__ == "__main__":
    main()
