# Supported methods (10):
# dense, svg1, svg2, spargeattn, radial, sta, draft, adacluster, flashomni, svoo
#
# Supported backbones (19):
# wan1.3b, wan14b, wan22, hunyuan, wan21-i2v-14b, wan22-i2v, hunyuan-i2v,
# skyreels-v2, skyreels-v2-i2v, wananimate, wan-vace, wan21-vace-14b,
# cogvideox, cogvideox-i2v, ltx, ltx-i2v, allegro, mochi, easyanimate
#
# Run after activating the sparsevideo environment:
source /home/dataset-assist-0/luojy/miniconda3/bin/activate
conda activate sparsevideo
#
# Total command lines: 190 = 19 backbones x 10 methods.

# ---------------------------------------
# wan 1.3b t2v (model: wan1.3b)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method dense      --prompt-file example/t2v/1.txt # [ok] 12:20
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method svg1       --prompt-file example/t2v/1.txt # [ok] 07:29
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wan1.3b           --method svg2       --prompt-file example/t2v/1.txt # [ok] 06:56
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan1.3b           --method spargeattn --prompt-file example/t2v/1.txt # [ok] 08:14
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan1.3b           --method radial     --prompt-file example/t2v/1.txt # [ok] 07:26
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan1.3b           --method sta        --prompt-file example/t2v/1.txt # [x] bad performance
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method draft      --prompt-file example/t2v/1.txt # [ok] 06:14
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method adacluster --prompt-file example/t2v/1.txt # [ok] 06:14
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method flashomni  --prompt-file example/t2v/1.txt # [ok] 07:31
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan1.3b           --method svoo       --prompt-file example/t2v/1.txt # [ok] 06:03

# ---------------------------------------
# wan 14b t2v (model: wan14b)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan14b            --method dense      --prompt-file example/t2v/1.txt # 60:19
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan14b            --method svg1       --prompt-file example/t2v/1.txt # 37:04
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan14b            --method svg2       --prompt-file example/t2v/1.txt # 41:59
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan14b            --method spargeattn --prompt-file example/t2v/1.txt # 40:38
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan14b            --method radial     --prompt-file example/t2v/1.txt # 36:32
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan14b            --method sta        --prompt-file example/t2v/1.txt # [x] 36:10
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan14b            --method draft      --prompt-file example/t2v/1.txt # 32:00
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan14b            --method adacluster --prompt-file example/t2v/1.txt # 32:53
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan14b            --method flashomni  --prompt-file example/t2v/1.txt # 41:26
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan14b            --method svoo       --prompt-file example/t2v/1.txt # 38:11

# ---------------------------------------
# wan 2.1 i2v 14B (model: wan21-i2v-14b)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 60:41
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 36:33
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 41:31
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 40:53
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 36:45
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 32:02
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 33:15
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 40:19
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan21-i2v-14b     --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 38:17

# ---------------------------------------
# wan 2.2 t2v A14B (model: wan22)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan22             --method dense      --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 49:43
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan22             --method svg1       --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 29:56
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan22             --method svg2       --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 35:30
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan22             --method spargeattn --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 33:20
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan22             --method radial     --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 37:08 [needs to recheck later]
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan22             --method sta        --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wan22             --method draft      --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 26:05
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan22             --method adacluster --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 28:19
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wan22             --method flashomni  --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 33:37
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan22             --method svoo       --prompt-file example/t2v/1.txt --cpu-offload --cpu-offload-mode model # 32:13

# ---------------------------------------
# wan 2.2 i2v A14B (model: wan22-i2v)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan22-i2v         --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 49:08
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan22-i2v         --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 30:16
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan22-i2v         --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 34:12
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan22-i2v         --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 33:29
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wan22-i2v         --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan22-i2v         --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 29:53
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan22-i2v         --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 26:29
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan22-i2v         --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 27:03
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan22-i2v         --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 35:17
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan22-i2v         --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --cpu-offload --cpu-offload-mode model # 31:15

# ---------------------------------------
# hunyuanvideo t2v (model: hunyuan)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model hunyuan           --method dense      --prompt-file example/t2v/1.txt # 112:22
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model hunyuan           --method svg1       --prompt-file example/t2v/1.txt # 29:57
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model hunyuan           --method svg2       --prompt-file example/t2v/1.txt # 34:43
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model hunyuan           --method spargeattn --prompt-file example/t2v/1.txt # 36:33
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model hunyuan           --method radial     --prompt-file example/t2v/1.txt # 35:45
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model hunyuan           --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model hunyuan           --method draft      --prompt-file example/t2v/1.txt # 20:21
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model hunyuan           --method adacluster --prompt-file example/t2v/1.txt # 20:22
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model hunyuan           --method flashomni  --prompt-file example/t2v/1.txt # 33:39
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model hunyuan           --method svoo       --prompt-file example/t2v/1.txt # 29:05

# ---------------------------------------
# hunyuanvideo i2v (model: hunyuan-i2v)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model hunyuan-i2v       --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 112:19
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model hunyuan-i2v       --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 29:41
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model hunyuan-i2v       --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 35:07
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model hunyuan-i2v       --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 36:37
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model hunyuan-i2v       --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 34:04
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model hunyuan-i2v       --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model hunyuan-i2v       --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 19:31 (needs to balance again)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model hunyuan-i2v       --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 29:43
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model hunyuan-i2v       --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 29:34
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model hunyuan-i2v       --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 29:34

# ---------------------------------------
# skyreels-v2 t2v (model: skyreels-v2)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model skyreels-v2       --method dense      --prompt-file example/t2v/1.txt # 80:50
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model skyreels-v2       --method svg1       --prompt-file example/t2v/1.txt # 56:15
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model skyreels-v2       --method svg2       --prompt-file example/t2v/1.txt # 59:41
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model skyreels-v2       --method spargeattn --prompt-file example/t2v/1.txt # 60:26
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model skyreels-v2       --method radial     --prompt-file example/t2v/1.txt # 57:12 [needs to recheck later]
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model skyreels-v2       --method sta        --prompt-file example/t2v/1.txt 
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model skyreels-v2       --method draft      --prompt-file example/t2v/1.txt # 47:32 
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model skyreels-v2       --method adacluster --prompt-file example/t2v/1.txt # 48:35
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model skyreels-v2       --method flashomni  --prompt-file example/t2v/1.txt # 64:58
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model skyreels-v2       --method svoo       --prompt-file example/t2v/1.txt # 56:58

# ---------------------------------------
# skyreels-v2 i2v (model: skyreels-v2-i2v)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 81:03
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 55:04 (not so good)
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 58:04
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 60:41
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 56:53 [needs to recheck later]
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 48:35
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 48:20
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 63:43
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model skyreels-v2-i2v   --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg # 55:45

# ---------------------------------------
# wananimate (model: wananimate)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wananimate        --method dense      --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wananimate        --method svg1       --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wananimate        --method svg2       --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wananimate        --method spargeattn --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wananimate        --method radial     --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wananimate        --method sta        --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wananimate        --method draft      --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wananimate        --method adacluster --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wananimate        --method flashomni  --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wananimate        --method svoo       --prompt-file example/t2v/1.txt --image example/i2v/1.jpg --pose-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --face-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4

# ---------------------------------------
# wan vace 1.3b (model: wan-vace)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wan-vace          --method dense      --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan-vace          --method svg1       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wan-vace          --method svg2       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan-vace          --method spargeattn --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan-vace          --method radial     --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan-vace          --method sta        --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan-vace          --method draft      --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan-vace          --method adacluster --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wan-vace          --method flashomni  --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --method-config sparse_pattern=paper_mmdit --method-config use_sparse_gemm=false --method-config taylor_cache_device=cpu
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan-vace          --method svoo       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4

# ---------------------------------------
# wan vace 14b (model: wan21-vace-14b)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wan21-vace-14b    --method dense      --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan21-vace-14b    --method svg1       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model wan21-vace-14b    --method svg2       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model wan21-vace-14b    --method spargeattn --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model wan21-vace-14b    --method radial     --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model wan21-vace-14b    --method sta        --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model wan21-vace-14b    --method draft      --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model wan21-vace-14b    --method adacluster --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model wan21-vace-14b    --method flashomni  --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --method-config sparse_pattern=paper_mmdit --method-config use_sparse_gemm=false --method-config taylor_cache_device=cpu
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model wan21-vace-14b    --method svoo       --prompt-file example/t2v/1.txt --reference-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4 --mask-video result/inference/wan21-t2v-1.3b/dense/seed0_720x1280_81f.mp4

# ---------------------------------------
# cogvideox t2v (model: cogvideox)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model cogvideox         --method dense      --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:26
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model cogvideox         --method svg1       --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:17
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model cogvideox         --method svg2       --prompt-file example/t2v/1.txt --height 480 --width 720 # 06:29
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model cogvideox         --method spargeattn --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:05
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model cogvideox         --method radial     --prompt-file example/t2v/1.txt --height 480 --width 720 # 04:49
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model cogvideox         --method sta        --prompt-file example/t2v/1.txt --height 480 --width 720
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model cogvideox         --method draft      --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:48
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model cogvideox         --method adacluster --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:41
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model cogvideox         --method flashomni  --prompt-file example/t2v/1.txt --height 480 --width 720 # 03:11 [not so good]
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model cogvideox         --method svoo       --prompt-file example/t2v/1.txt --height 480 --width 720 # 04:00

# ---------------------------------------
# cogvideox i2v (model: cogvideox-i2v)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model cogvideox-i2v     --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720 # 03:26
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model cogvideox-i2v     --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720 # 03:07
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model cogvideox-i2v     --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720 #
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model cogvideox-i2v     --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720 # 03:06
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model cogvideox-i2v     --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model cogvideox-i2v     --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model cogvideox-i2v     --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model cogvideox-i2v     --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model cogvideox-i2v     --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model cogvideox-i2v     --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 480 --width 720

# ---------------------------------------
# ltx t2v (model: ltx)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model ltx               --method dense      --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model ltx               --method svg1       --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model ltx               --method svg2       --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model ltx               --method spargeattn --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model ltx               --method radial     --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model ltx               --method sta        --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model ltx               --method draft      --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model ltx               --method adacluster --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model ltx               --method flashomni  --prompt-file example/t2v/1.txt --height 704 --width 1280
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model ltx               --method svoo       --prompt-file example/t2v/1.txt --height 704 --width 1280

# ---------------------------------------
# ltx i2v (model: ltx-i2v)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model ltx-i2v           --method dense      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model ltx-i2v           --method svg1       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model ltx-i2v           --method svg2       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model ltx-i2v           --method spargeattn --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model ltx-i2v           --method radial     --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model ltx-i2v           --method sta        --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model ltx-i2v           --method draft      --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model ltx-i2v           --method adacluster --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model ltx-i2v           --method flashomni  --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model ltx-i2v           --method svoo       --prompt-file example/i2v/1.txt --image example/i2v/1.jpg --height 704 --width 1280

# ---------------------------------------
# allegro t2v (model: allegro)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model allegro           --method dense      --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model allegro           --method svg1       --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model allegro           --method svg2       --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model allegro           --method spargeattn --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model allegro           --method radial     --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model allegro           --method sta        --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model allegro           --method draft      --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model allegro           --method adacluster --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model allegro           --method flashomni  --prompt-file example/t2v/1.txt --vae-tiling
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model allegro           --method svoo       --prompt-file example/t2v/1.txt --vae-tiling

# ---------------------------------------
# mochi t2v (model: mochi)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model mochi             --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model mochi             --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model mochi             --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model mochi             --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model mochi             --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model mochi             --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model mochi             --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model mochi             --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model mochi             --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model mochi             --method svoo       --prompt-file example/t2v/1.txt

# ---------------------------------------
# easyanimate t2v (model: easyanimate)
# ---------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model easyanimate       --method dense      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model easyanimate       --method svg1       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=2 python scripts/infer_diffusers.py --model easyanimate       --method svg2       --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=3 python scripts/infer_diffusers.py --model easyanimate       --method spargeattn --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=4 python scripts/infer_diffusers.py --model easyanimate       --method radial     --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=5 python scripts/infer_diffusers.py --model easyanimate       --method sta        --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=6 python scripts/infer_diffusers.py --model easyanimate       --method draft      --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=7 python scripts/infer_diffusers.py --model easyanimate       --method adacluster --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=0 python scripts/infer_diffusers.py --model easyanimate       --method flashomni  --prompt-file example/t2v/1.txt
CUDA_VISIBLE_DEVICES=1 python scripts/infer_diffusers.py --model easyanimate       --method svoo       --prompt-file example/t2v/1.txt
