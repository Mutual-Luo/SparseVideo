#!/usr/bin/env bash
# SparseVideo DiffSynth-Studio inference commands.
# Mirrors inference_diffusers.sh for the DiffSynth backend.
#
# Supported methods (10):
# dense, svg1, svg2, spargeattn, radial, sta, draft, adacluster, flashomni, svoo
#
# Core DiffSynth models covered here:
# wan21-t2v-1.3b, wan21-t2v-14b, wan22-t2v-a14b,
# wan21-i2v-14b-720p, wan22-i2v-a14b,
# wan22-animate-14b, wan21-vace-1.3b, wan21-vace-14b

PROMPT='A cinematic shot of a red sports car driving along a coastal road at sunset, detailed, realistic'

# ── wan21-t2v-1.3b ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method dense      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svg1       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svg2       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method spargeattn --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method radial     --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method sta        --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method draft      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method adacluster --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method flashomni  --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svoo       --prompt "$PROMPT"

# ── wan21-t2v-14b ────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method dense      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svg1       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svg2       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method spargeattn --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method radial     --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method sta        --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method draft      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method adacluster --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method flashomni  --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svoo       --prompt "$PROMPT"

# ── wan22-t2v-a14b ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method dense      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svg1       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svg2       --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method spargeattn --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method radial     --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method sta        --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method draft      --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method adacluster --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method flashomni  --prompt "$PROMPT"
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svoo       --prompt "$PROMPT"

# ── wan21-i2v-14b-720p (image-to-video) ─────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method dense      --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svg1       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svg2       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method spargeattn --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method radial     --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method sta        --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method draft      --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method adacluster --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method flashomni  --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svoo       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg

# ── wan22-i2v-a14b (image-to-video) ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method dense      --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svg1       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svg2       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method spargeattn --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method radial     --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method sta        --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method draft      --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method adacluster --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method flashomni  --prompt "$PROMPT" --input-image test_inputs/portrait.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svoo       --prompt "$PROMPT" --input-image test_inputs/portrait.jpg

# ── wan22-animate-14b ────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-animate-14b --method dense      --prompt "$PROMPT" --animate-pose-video test_inputs/pose.mp4 --animate-face-video test_inputs/face.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-animate-14b --method svg2       --prompt "$PROMPT" --animate-pose-video test_inputs/pose.mp4 --animate-face-video test_inputs/face.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-animate-14b --method svoo       --prompt "$PROMPT" --animate-pose-video test_inputs/pose.mp4 --animate-face-video test_inputs/face.mp4

# ── wan21-vace-1.3b ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method dense --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method svg2  --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method svoo  --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4

# ── wan21-vace-14b ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-14b --method dense --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-14b --method svg2  --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-14b --method svoo  --prompt "$PROMPT" --vace-video test_inputs/reference.mp4 --vace-video-mask test_inputs/mask.mp4
