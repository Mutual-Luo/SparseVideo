#!/usr/bin/env bash
# SparseVideo DiffSynth-Studio inference commands.
# Mirrors inference_diffusers.sh for the DiffSynth backend.
#
# Supported methods (10):
# dense, svg1, svg2, spargeattn, radial, sta, draft, adacluster, flashomni, svoo
#
# DiffSynth models (26 x 10 = 260 commands):
# wan21-t2v-1.3b, wan21-t2v-14b, wan22-t2v-a14b,
# wan21-i2v-14b-720p, wan21-i2v-14b-480p, wan21-flf2v-14b-720p, wan22-i2v-a14b,
# wan22-animate-14b, wan21-vace-1.3b, wan21-vace-14b,
# wan21-speedcontrol-1.3b, wan22-ti2v-5b, wan22-s2v-14b, krea-realtime-video,
# wan21-fun-1.3b-control, wan21-fun-1.3b-inp, wan21-fun-14b-control, wan21-fun-14b-inp,
# wan21-fun-v11-1.3b-control, wan21-fun-v11-1.3b-control-camera,
# wan21-fun-v11-14b-control, wan21-fun-v11-14b-control-camera,
# wan22-fun-a14b-control, wan22-fun-a14b-control-camera,
# longcat-video, video-as-prompt-wan21-14b
#
# Input conventions (matching inference_diffusers.sh):
#   T2V:          --prompt-file example/t2v/1.txt
#   I2V:          --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
#   FLF2V/Inp:    --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
#   Control:      --prompt-file example/control/1.txt --control-video example/control/1.mp4
#   V11-Ctrl:     above + --reference-image example/control/1_ref.png
#   Camera:       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
#   VACE:         --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
#   Animate:      --prompt-file example/animate/1.txt --animate-pose-video example/animate/process_results/src_pose.mp4 --animate-face-video example/animate/process_results/src_face.mp4
#   S2V:          --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
#   LongCat/VAP:  --prompt-file example/t2v/1.txt --longcat-video/--vap-video example/animate/1.mp4

# ── wan21-t2v-1.3b ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svoo       --prompt-file example/t2v/1.txt

# ── wan21-t2v-14b ────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-t2v-14b --method svoo       --prompt-file example/t2v/1.txt

# ── wan22-t2v-a14b ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method dense      --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svg1       --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svg2       --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method spargeattn --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method radial     --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method sta        --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method draft      --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method adacluster --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method flashomni  --prompt-file example/t2v/1.txt --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svoo       --prompt-file example/t2v/1.txt --vram-limit 60

# ── wan21-i2v-14b-720p ───────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-720p --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg

# ── wan22-i2v-a14b ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-i2v-a14b --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --vram-limit 60

# ── wan22-animate-14b ────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-animate-14b --method dense      --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-animate-14b --method svg1       --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-animate-14b --method svg2       --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-animate-14b --method spargeattn --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-animate-14b --method radial     --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-animate-14b --method sta        --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-animate-14b --method draft      --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-animate-14b --method adacluster --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-animate-14b --method flashomni  --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-animate-14b --method svoo       --prompt-file example/animate/1.txt --input-image example/animate/official/official_image.png --animate-pose-video example/animate/official/official_pose.mp4 --animate-face-video example/animate/official/official_face.mp4

# ── wan21-vace-1.3b ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method dense      --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method svg1       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method svg2       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method spargeattn --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method radial     --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method sta        --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method draft      --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method adacluster --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method flashomni  --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-1.3b --method svoo       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4

# ── wan21-vace-14b ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-14b --method dense      --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-vace-14b --method svg1       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-14b --method svg2       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-14b --method spargeattn --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-14b --method radial     --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-vace-14b --method sta        --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-vace-14b --method draft      --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-vace-14b --method adacluster --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-vace-14b --method flashomni  --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-vace-14b --method svoo       --prompt-file example/inpainting/prompt.txt --vace-video example/inpainting/src_video.mp4 --vace-video-mask example/inpainting/src_mask.mp4

# ── wan21-speedcontrol-1.3b ───────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-speedcontrol-1.3b --method svoo       --prompt-file example/t2v/1.txt

# ── wan21-i2v-14b-480p ───────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-i2v-14b-480p --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg

# ── wan21-flf2v-14b-720p ─────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method dense      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method svg1       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method svg2       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method spargeattn --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method radial     --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method sta        --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method draft      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method adacluster --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method flashomni  --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-flf2v-14b-720p --method svoo       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png

# ── wan21-fun-1.3b-control ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method dense      --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method svg1       --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method svg2       --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method spargeattn --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method radial     --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method sta        --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method draft      --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method adacluster --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method flashomni  --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-control --method svoo       --prompt-file example/control/1.txt --control-video example/control/1.mp4

# ── wan21-fun-1.3b-inp ────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method dense      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method svg1       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method svg2       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method spargeattn --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method radial     --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method sta        --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method draft      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method adacluster --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method flashomni  --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-1.3b-inp --method svoo       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png

# ── wan21-fun-14b-control ─────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method dense      --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method svg1       --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method svg2       --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method spargeattn --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method radial     --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method sta        --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method draft      --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method adacluster --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method flashomni  --prompt-file example/control/1.txt --control-video example/control/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-14b-control --method svoo       --prompt-file example/control/1.txt --control-video example/control/1.mp4

# ── wan21-fun-14b-inp ─────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method dense      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method svg1       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method svg2       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method spargeattn --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method radial     --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method sta        --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method draft      --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method adacluster --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method flashomni  --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-14b-inp --method svoo       --prompt-file example/flf2v/1.txt --input-image example/flf2v/1_first.png --end-image example/flf2v/1_last.png

# ── wan21-fun-v11-1.3b-control ────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method dense      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method svg1       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method svg2       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method spargeattn --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method radial     --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method sta        --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method draft      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method adacluster --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method flashomni  --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control --method svoo       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png

# ── wan21-fun-v11-1.3b-control-camera ────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-1.3b-control-camera --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in

# ── wan21-fun-v11-14b-control ─────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method dense      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method svg1       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method svg2       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method spargeattn --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method radial     --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method sta        --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method draft      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method adacluster --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method flashomni  --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control --method svoo       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png

# ── wan21-fun-v11-14b-control-camera ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan21-fun-v11-14b-control-camera --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in

# ── wan22-ti2v-5b ────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-ti2v-5b --method svoo       --prompt-file example/t2v/1.txt

# ── wan22-s2v-14b ─────────────────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method dense      --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method svg1       --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method svg2       --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method spargeattn --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method radial     --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method sta        --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method draft      --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method adacluster --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method flashomni  --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-s2v-14b --method svoo       --prompt-file example/s2v/1.txt --input-image example/s2v/1.png --s2v-pose-video example/s2v/pose.mp4 --input-audio example/s2v/1.wav

# ── wan22-fun-a14b-control ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method dense      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method svg1       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method svg2       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method spargeattn --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method radial     --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method sta        --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method draft      --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method adacluster --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method flashomni  --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control --method svoo       --prompt-file example/control/1.txt --control-video example/control/1.mp4 --reference-image example/control/1_ref.png

# ── wan22-fun-a14b-control-camera ─────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method dense      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method svg1       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method svg2       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method spargeattn --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method radial     --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method sta        --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method draft      --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method adacluster --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method flashomni  --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model wan22-fun-a14b-control-camera --method svoo       --prompt-file example/i2v/1.txt --input-image example/i2v/1.jpg --camera-control-direction zoom_in

# ── longcat-video ─────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model longcat-video --method dense      --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model longcat-video --method svg1       --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model longcat-video --method svg2       --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model longcat-video --method spargeattn --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model longcat-video --method radial     --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model longcat-video --method sta        --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model longcat-video --method draft      --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model longcat-video --method adacluster --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model longcat-video --method flashomni  --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model longcat-video --method svoo       --prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4

# ── video-as-prompt-wan21-14b ─────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method dense      --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method svg1       --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method svg2       --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method spargeattn --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method radial     --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method sta        --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method draft      --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method adacluster --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method flashomni  --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model video-as-prompt-wan21-14b --method svoo       --prompt-file example/t2v/1.txt --input-image example/vap/input_image.jpg --vap-video example/vap/ref.mp4 --vap-prompt "A man stands with his back to the camera on a dirt path overlooking sun-drenched rolling green tea plantations. He wears a blue and green plaid shirt dark pants and white shoes. As he turns to face the camera and spreads his arms a brief magical burst of sparkling golden light particles envelops him. Through this shimmer he seamlessly transforms into a Labubu toy character."

# ── krea-realtime-video ───────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model krea-realtime-video --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model krea-realtime-video --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model krea-realtime-video --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model krea-realtime-video --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model krea-realtime-video --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model krea-realtime-video --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffsynth.py --model krea-realtime-video --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffsynth.py --model krea-realtime-video --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffsynth.py --model krea-realtime-video --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffsynth.py --model krea-realtime-video --method svoo       --prompt-file example/t2v/1.txt
