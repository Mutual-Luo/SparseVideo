### STA mask strategy search
cd /home/dataset-assist-0/luojy/efficiency/sparseformer
export PY=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
export RUN_ID=wan13_sta_native_search_20260524_a
export OUT=result/sta_mask_search/$RUN_ID
mkdir -p "$OUT/search" "$OUT/logs"
PROMPTS=(
example/t2v/1.txt
example/t2v/2.txt
example/t2v/3.txt
)

GPUS=(4 5 6)

COMMON=(
--model wan1.3b
--method sta
--local-files-only
--height 720
--width 1280
--num-frames 81
--num-inference-steps 50
--guidance-scale 5.0
--seed 12345
--skip-decode
--print-json
--metrics-file "$OUT/metrics_$(hostname).jsonl"
--method-config STA_mode=STA_searching
--method-config "mask_search_output_dir=$OUT/search"
--method-config 'window_size=[4,6,10]'
--method-config 'mask_candidates=[[3,1,10],[1,5,7],[3,3,3],[1,6,5],[1,3,10],[3,6,1]]'
)

for i in "${!GPUS[@]}"; do
gpu="${GPUS[$i]}"
prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/infer_diffusers.py \
    "${COMMON[@]}" \
    --prompt-file "$prompt" \
    --method-config "mask_search_prompt_id=$(hostname)_g${gpu}_p${i}" \
    > "$OUT/logs/search_$(hostname)_g${gpu}_p${i}.log" 2>&1 &
done

wait