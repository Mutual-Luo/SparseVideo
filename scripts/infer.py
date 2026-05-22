#!/usr/bin/env python3
#
# SparseVideo inference entrypoint.
#
# This is the only inference script you need. Select the model and sparse
# attention method by command-line arguments:
#
#   python scripts/infer.py --model wan1.3b --method dense
#   python scripts/infer.py --model wan1.3b --method svoo
#   python scripts/infer.py --model wan14b --method sta --profile upstream
#   python scripts/infer.py --model wan14b-i2v --method svoo --image input.jpg
#   python scripts/infer.py --model hunyuan-i2v --method radial --image input.png
#
# Supported models:
#   wan1.3b    Wan2.1 T2V 1.3B Diffusers
#   wan14b     Wan2.1 T2V 14B Diffusers
#   wan22      Wan2.2 T2V A14B Diffusers
#   wan14b-i2v Wan2.1 I2V 14B 720P Diffusers
#   wan22-i2v  Wan2.2 I2V A14B Diffusers
#   skyreels-v2 SkyReels V2 T2V 14B 720P Diffusers
#   skyreels-v2-i2v SkyReels V2 I2V 14B 720P Diffusers
#   hunyuan    HunyuanVideo T2V
#   hunyuan-i2v HunyuanVideo I2V
#   cogvideox  CogVideoX T2V, dense plus public sparse method support
#   cogvideox-i2v CogVideoX I2V, dense plus public sparse method support
#   ltx        LTX Video T2V, dense plus public sparse method support
#   ltx-i2v    LTX Video I2V, dense plus public sparse method support
#   allegro    Allegro T2V, dense plus public sparse method support
#   mochi      Mochi 1 preview, dense plus public sparse method support
#   easyanimate EasyAnimate V5 T2V, dense plus public sparse method support
#   motif-video MotifVideo, unknown in current Diffusers install; dry-run only
#   ltx-video-2 LTX Video 2, unknown in current Diffusers install; dry-run only
#   sana-video SanaVideo, incompatible linear attention; dense dry-run only
#   kandinsky5 Kandinsky 5 T2V, native sparse controls; dense dry-run only
#
# Supported methods:
#   dense       Original dense attention baseline.
#   svg1        Sparse-VideoGen SVG-style method.
#   svg2        Sparse-VideoGen second method.
#   spargeattn  SpargeAttn method.
#   radial      radial-attention method.
#   sta         FastVideo Sliding Tile Attention. SparseVideo dispatches only
#               the owned FastVideo STA path: H100/TK C++ on Hopper when the
#               local extension is built, otherwise FastVideo's own Triton
#               fallback. Wan/Hunyuan parity runs require native FastVideo
#               shapes; non-native shapes run as generalized local support.
#   draft       draft-attention method. Upstream parity requires an owned copy
#               of MIT Han Lab Block-Sparse-Attention; the current generic
#               Triton block-sparse path is debug fallback only.
#   adacluster  AdaCluster method.
#   flashomni   FlashOmni kernel adapter. Upstream-equivalent video runs use
#               sparse_pattern=explicit with caller-provided sparse-info
#               tensors. sparse_pattern=global_random matches the upstream
#               synthetic kernel benchmark mask generator only; it is not a
#               video quality-parity sparsity policy.
#               sparse_pattern=paper_mmdit includes owned ports of the
#               anonymous Hunyuan attention sparse-symbol policy and Hunyuan
#               transformer forward/Taylor-cache method path.
#               For Hunyuan quality runs, the CLI defaults paper_mmdit to
#               max_order=0,use_sparse_gemm=false. The reported D/max_order=1
#               path is allowed with use_sparse_gemm=false; sparse GEMM
#               projection is retained in source for audit history, but blocked
#               for Hunyuan inference because current tests showed visual
#               degradation and slower generation.
#               sparse_pattern=local_qk_topk is a SparseVideo diagnostic path,
#               not method parity.
#   svoo        SVOO method.
#
# Current sparse support:
#   Wan-family/SkyReels (T2V and I2V) and Hunyuan (T2V and I2V) support the sparse methods above.
#   I2V models require --image <path>. Sparse attention applies to video self-attention;
#   image cross-attention remains dense as in upstream.
#   CogVideoX T2V/I2V, LTX Video, Allegro, Mochi, and EasyAnimate support the
#   public sparse methods above. SanaVideo is incompatible with sparse-softmax
#   methods; Kandinsky5 is native-N/A for processor swap. MotifVideo and
#   LTX Video 2 are unknown in the current Diffusers install.
#
# Common options:
#
#   --dry-run                 Validate settings without loading model.
#   --print-json              Print full metrics/config/runtime payload.
#                             Also reports optional kernel availability.
#   --profile upstream        Use the upstream method benchmark profile for
#                             shape/frame/config; fails if none is defined.
#   --profile-for-method draft
#                             With --method dense, use another method's
#                             upstream profile for a fair dense baseline.
#   --num-frames 81           Override exact frame count.
#   --num-inference-steps 10  Run fewer steps for a smoke test.
#   --prompt "..."            Override prompt from the command line.
#   --prompt-file prompt.txt  Read prompt from a file.
#   --cpu-offload / --no-cpu-offload
#                             Use or disable pipeline CPU offload. Some
#                             upstream profiles set this automatically.
#   --allow-debug-fallbacks   Allow explicitly labeled non-parity fallback
#                             paths for smoke/debug runs only.
#   --strict-kernels          Keep strict fallback checks enabled. This is the
#                             default and overrides --allow-debug-fallbacks.
#   --skip-decode             Use output_type=latent and skip mp4 export.
#                             This is only for backend/kernel dispatch smoke,
#                             not quality review.
#
# SVOO runtime env defaults:
#   SVOO_CACHE_ROOT           Base compiler cache, default .triton_cache.
#   TRITON_CACHE_DIR          Triton JIT cache.
#   TORCHINDUCTOR_CACHE_DIR   TorchInductor cache.
#   FLASHINFER_WORKSPACE_BASE FlashInfer workspace cache.
#   SVOO_TRITON_WARMUP=0|1    Precompile SVOO kernels before generation; default 1.
#   SVOO_TRITON_WARMUP_MODE=compile|full
#                             compile uses a capped sequence length; full uses
#                             the requested video token length.
#   SVOO_TRITON_TUNE=auto     Optional SVOO Triton autotuning.
#   SVOO_ENABLE_MEM_SAVE=0|1  Release SVOO intermediates earlier; default 1.
#   SPARSEVIDEO_FUSED_KERNEL_BACKEND=auto|native|triton|pytorch
#                            auto uses SparseVideo-owned _kernels when found.
#
# Default method hyperparameters live in
# src/sparsevideo/methods/<method>/config.yaml. Method config names follow the
# upstream repositories where possible:
#   Common:      dense_warmup_step_ratio, dense_warmup_layer_ratio.
#                These are ratio-only clearer names for keeping the first
#                fraction of denoising steps/layers dense. Defaults are 0.1
#                and 0.03; set either value to 0 to disable that common gate.
#   svg1:        num_sampled_rows,
#                sample_mse_max_row, sparsity, context_length, prompt_length.
#   svg2:        num_q_centroids,
#                num_k_centroids, top_p_kmeans, min_kc_ratio,
#                kmeans_iter_init, kmeans_iter_step, zero_step_kmeans_init,
#                context_length, prompt_length, allow_triton_fallback. The local
#                Triton block-sparse path is debug-only and enabled by
#                --allow-debug-fallbacks.
#   spargeattn:  mode=full|cdfthreshd|topk|block_sparse, value,
#                topk, cdfthreshd, simthreshd1, pvthreshd,
#                attention_sink, smooth_k, dropout_p, scale,
#                tensor_layout, output_dtype, mask_id, tune,
#                parallel_tune, sim_rule, l1, pv_l1, cos_sim, rmse,
#                rearrange_kwargs, tune_pv, verbose, model_out_path.
#                SparseVideo defaults to upstream's recommended plug-and-play
#                topk=0.5 path. Wan2.1/Hunyuan --profile upstream follows
#                the upstream example scripts and sets mode=full for their
#                dense SpargeAttn baseline; set mode/value explicitly for
#                sparse SpargeAttn comparisons. SpargeAttn upstream profiles
#                also enable sequential CPU offload, VAE tiling/slicing, and
#                decoder_chunk_size=1 as in the example scripts.
#                Upstream API names are preserved. value is kept because the
#                upstream Wan Diffusers example uses mode/value; topk and
#                cdfthreshd are the direct kernel argument names. block_sparse
#                calls block_sparse_sage2_attn_cuda and requires mask_id.
#                tune=true follows upstream SparseAttentionMeansim and saves
#                tuned state to model_out_path; tune=false with model_out_path
#                loads that tuned state for inference.
#   radial:      dense_layers, dense_timesteps, decay_factor, block_size,
#                use_sage_attention, allow_flex_fallback. Wan/Hunyuan defaults
#                follow the upstream README scripts: Wan uses dense_layers=1,
#                dense_timesteps=12, decay_factor=0.2; Hunyuan uses 0,12,0.95.
#                dense_timesteps is the upstream scheduler timestep threshold
#                (`timestep < dense_timesteps`), not a count of first denoising
#                steps.
#                Upstream Radial FlashInfer benchmarks use 1280x768 shapes
#                whose video token counts divide block_size. SparseVideo keeps
#                that fixed-BSR path for those shapes and uses the owned
#                FlashInfer variable-block wrapper for partial final blocks.
#                use_sage_attention uses the owned Sparge/Sage kernel partial
#                final-block support for non-divisible shapes.
#   sta:         STA_mode, mask_strategy_file_path, tile_size, window_size,
#                seq_shape, has_text.
#                FastVideo STA uses tile_size=6,8,8 and supports
#                seq_shape=18x48x80, 30x48x80, or 36x48x48. Other 720p shapes
#                are local generalized STA runs, not FastVideo parity profiles.
#                STA_mode supports STA_inference and STA_searching. Known
#                backbones use owned mask_strategies/mask_strategy_<model_key>.json
#                files when available; generate them with
#                python -m sparsevideo.methods.sta.search tune.
#                The SparseVideo-owned sta_h100 C++ path is selected only on
#                Hopper devices when its local extension is built; A100 uses
#                the SparseVideo-owned copy of FastVideo's Triton STA fallback.
#                Use --profile upstream to set the matching seq_shape and
#                upstream 1280x768 frame count automatically.
#   draft:       pool_h, pool_w, latent_h, latent_w, visual_len, text_len,
#                sparsity_ratio, batch_size, block_sparse_attention,
#                allow_triton_fallback.
#                Wan upstream sparse path uses sparsity_ratio=0.75 and supports
#                768x512 or 1280x768 latent layouts; Hunyuan uses 0.9 at
#                1280x768. New backbones use model-aware latent_h/latent_w/
#                visual_len defaults and the owned MIT block-sparse backend. Use
#                --profile upstream for the upstream shell/demo layouts.
#                Strict benchmark preflight requires SparseVideo-owned MIT Han
#                Lab Block-Sparse-Attention under
#                src/sparsevideo/kernels/native/draft_block_sparse. The local
#                Triton block-sparse path is debug fallback only, enabled only
#                by --allow-debug-fallbacks when the MIT backend is unavailable.
#   adacluster:  topk_num, q_kernel_num, kv_kernel_num,
#                kmeans_iter_init, kmeans_iter_step,
#                use_thresholded_kmeans_loop, initial_q_kernel_num,
#                initial_kv_kernel_num, q_distance_threshold,
#                kv_distance_threshold, thresholded_kmeans_iter_time,
#                thresholded_kmeans_max_iterations. Wan runwan/generate.py
#                imports runwan/wan/modules/model.py, whose default sparse
#                path uses fixed 100/500 Q/KV clusters. The thresholded loop
#                remains configurable because model_kvclus.py exists as an
#                alternate upstream reference, but it is not the runwan default.
#                Hunyuan keeps its upstream fixed 250/1243 cluster profile and
#                dense gates.
#   flashomni:   implementation=upstream|flex, backend, workspace_bytes,
#                sparse_block_size_for_q, sparse_block_size_for_kv,
#                causal, pos_encoding_mode, use_fp16_qk_reduction,
#                logits_soft_cap, sm_scale, rope_scale, rope_theta,
#                sparse_pattern=explicit|global_random|paper_mmdit|local_qk_topk,
#                sparse_info, sparse_kv_info, sparse_info_indptr,
#                sparse_kv_info_indptr, is_full, sparse_size, spq_Q, spq_KV,
#                text_token, threshold_q, threshold_kv, fresh_threshold,
#                max_order, first_enhance, saving_threshold_q_for_taylor,
#                max_sequence_length, num_inference_steps, simthreshd1,
#                use_sparse_gemm,
#                sparse_kv_budget.
#                tau_q/tau_kv/N/D/S_q are accepted as legacy aliases for the
#                upstream Hunyuan names above; cdfthreshd and tau_c are older
#                aliases for tau_kv and tau_q.
#                Default sparse_pattern=explicit requires caller-provided
#                upstream sparse-info tensors.
#                For CLI use, the four sparse-info values may be paths to
#                torch-saved tensors or a torch-saved dict containing the same
#                upstream keys. The explicit path accepts either upstream
#                unpacked 0/1 sparse-info tensors with logical indptr or
#                already packed tensors with the indptr returned by
#                flashomni.segment_packbits.
#                sparse_pattern=global_random follows FlashOmni's upstream
#                synthetic kernel benchmark. It is useful only for explicit
#                native-kernel smoke/speed checks with --allow-debug-fallbacks;
#                it is not a video quality-parity policy and is rejected by
#                strict preflight.
#                sparse_pattern=paper_mmdit uses SparseVideo-owned FlashOmni
#                attention sparse-info code. Hunyuan runs use the public
#                anonymous FlashOmni Hunyuan sparse-symbol policy and default
#                names: threshold_q, threshold_kv, fresh_threshold,
#                max_order, first_enhance, saving_threshold_q_for_taylor.
#                The Hunyuan forward/Taylor-cache patch is installed through
#                apply_sparse_attention and restored with the public handle.
#                For Hunyuan quality runs, this CLI defaults to the safer
#                max_order=0,use_sparse_gemm=false path. The reported
#                D/max_order=1 path is allowed with use_sparse_gemm=false;
#                use_sparse_gemm=true is retained in code for reference/audit
#                but rejected for Hunyuan CLI inference because measured runs
#                showed both quality degradation and performance regression.
#                use_sparse_gemm controls
#                the owned FlashOmni GEMM-Q/GEMM-O dispatch hooks; those hooks
#                follow the upstream sparse_size=128 GEMM layout. It is
#                runnable development code, not upstream code parity.
#   svoo:        num_q_centroids,
#                num_k_centroids, top_p_kmeans, min_kc_ratio,
#                kmeans_iter_init, kmeans_iter_step, zero_step_kmeans_init,
#                start_reuse_step, reuse_interval, use_dynamic_min_kc_ratio,
#                sparsity_csv_path, dynamic_min_kc_ratio_min,
#                dynamic_min_kc_ratio_max, context_length, prompt_length,
#                measure_attention_sparsity, sparsity_output_file,
#                sparsity_batch_size, sparsity_query_samples,
#                sparsity_threshold, sparsity_start_step,
#                use_global_constraints, lambda_schedule, diverse_top_p_k,
#                enable_mem_save, implementation=native,
#                sparse_backend=flashinfer|triton. Hunyuan SVOO upstream uses
#                FlashInfer for both varlen dense gates and sparse attention;
#                sparse_backend=triton is only a Wan fallback path.
# Use upstream-facing method config names; legacy aliases are intentionally minimal.
#
# Example SVOO overrides:
#   python scripts/infer.py --model wan1.3b --method svoo \
#     --method-config top_p_kmeans=0.9 --method-config min_kc_ratio=0.1
#
# Defaults:
#   Wan uses 81 frames at 16 fps.
#   The normal default shape is 720x1280 for convenience comparisons, but
#   Wan2.1 T2V 1.3B is a 480P model in the Wan README. Its 720P output is
#   explicitly less stable, so 720P checks should be treated as target-shape
#   stress evidence rather than a standalone model-quality baseline.
#   Hunyuan uses 129 frames at 24 fps.
#   Method configs use the reference repositories' public names where possible.
#   For SVG1, SVG2, Draft, and SVOO, this script uses the inference-shell
#   defaults for Wan/Hunyuan instead of weak parser defaults when upstream has
#   model-specific settings for quality/speed comparison.
#   Wan loads the VAE in fp32 and uses flow_shift=5.0 at 720p by default,
#   matching the Diffusers/Wan reference example. Upstream profiles may
#   override these when the referenced method script uses different scheduler
#   or VAE dtype settings. Override manually with --flow-shift or --vae-dtype
#   if needed.
#   VAE tiling/slicing default to disabled; upstream profiles or explicit
#   --vae-tiling/--vae-slicing flags enable them when the reference script does.
#   SVOO dynamic CSV sparsity is enabled by those defaults; this script resolves
#   the default profile to src/sparsevideo/methods/svoo/sparsity_profiles/.
#   Outputs go to result/inference/<model>/<method>/.
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sparsevideo._support import LIMITED_METHODS_BY_MODEL_TYPE, unvalidated_method_reason

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

STA_NATIVE_SEQ_SHAPES = {"18x48x80", "30x48x80", "36x48x48"}
STA_STRATEGY_SHAPES = {
    "wan21-t2v-1.3b": (50, 30, 12),
    "wan21-vace-1.3b": (50, 30, 12),
    "wan21-t2v-14b": (50, 40, 40),
    "wan21-i2v-14b": (50, 40, 40),
    "wan21-vace-14b": (50, 40, 40),
    "skyreels-v2-t2v-14b": (50, 40, 40),
    "skyreels-v2-i2v-14b": (50, 40, 40),
    "wan22-t2v-a14b": (40, 40, 40),
    "wan22-i2v-a14b": (40, 40, 40),
    "wan22-animate-14b": (20, 40, 40),
    "hunyuan-t2v": (50, 60, 24),
    "hunyuan-i2v": (50, 60, 24),
    "cogvideox-t2v": (50, 42, 48),
    "cogvideox-i2v": (50, 42, 48),
    "ltx-video": (50, 28, 32),
    "ltx-video-i2v": (50, 28, 32),
    "allegro": (100, 32, 24),
    "mochi-1": (64, 48, 24),
    "easyanimate-v5-t2v-12b": (50, 48, 48),
}
STA_UNSUPPORTED_STRATEGY_MODELS = {}
FLASHOMNI_SPARSE_INFO_KEYS = (
    "sparse_info",
    "sparse_kv_info",
    "sparse_info_indptr",
    "sparse_kv_info_indptr",
)
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_SEED = 0


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
    sparse_methods: Optional[Tuple[str, ...]] = None
    compatibility_label: str = "likely-compatible"
    unsupported_reason: Optional[str] = None


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
    "wan21-i2v-14b": ModelSpec(
        key="wan21-i2v-14b",
        family="wan",
        pipeline_class="WanImageToVideoPipeline",
        hf_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        local_dir="Wan2.1-I2V-14B-720P-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan22-i2v-a14b": ModelSpec(
        key="wan22-i2v-a14b",
        family="wan",
        pipeline_class="WanImageToVideoPipeline",
        hf_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        local_dir="Wan2.2-I2V-A14B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=40,
        guidance_scale=5.0,
        output_type="np",
    ),
    "hunyuan-i2v": ModelSpec(
        key="hunyuan-i2v",
        family="hunyuan_video",
        pipeline_class="HunyuanVideoImageToVideoPipeline",
        hf_id="hunyuanvideo-community/HunyuanVideo-I2V",
        local_dir="HunyuanVideo-I2V",
        fps=24,
        default_frames=129,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
    ),
    "skyreels-v2-t2v-14b": ModelSpec(
        key="skyreels-v2-t2v-14b",
        family="wan",
        pipeline_class="SkyReelsV2Pipeline",
        hf_id="Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
        local_dir="skyreels-v2-t2v-14b",
        fps=24,
        default_frames=97,
        default_steps=50,
        guidance_scale=6.0,
        output_type="np",
    ),
    "skyreels-v2-i2v-14b": ModelSpec(
        key="skyreels-v2-i2v-14b",
        family="wan",
        pipeline_class="SkyReelsV2ImageToVideoPipeline",
        hf_id="Skywork/SkyReels-V2-I2V-14B-720P-Diffusers",
        local_dir="skyreels-v2-i2v-14b",
        fps=24,
        default_frames=97,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan22-animate-14b": ModelSpec(
        key="wan22-animate-14b",
        family="wan",
        pipeline_class="WanAnimatePipeline",
        hf_id="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        local_dir="Wan2.2-Animate-14B-Diffusers",
        fps=16,
        default_frames=77,
        default_steps=20,
        guidance_scale=1.0,
        output_type="np",
    ),
    "wan21-vace-1.3b": ModelSpec(
        key="wan21-vace-1.3b",
        family="wan",
        pipeline_class="WanVACEPipeline",
        hf_id="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        local_dir="Wan2.1-VACE-1.3B-diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan21-vace-14b": ModelSpec(
        key="wan21-vace-14b",
        family="wan",
        pipeline_class="WanVACEPipeline",
        hf_id="Wan-AI/Wan2.1-VACE-14B-diffusers",
        local_dir="Wan2.1-VACE-14B-diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
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
        sparse_supported=True,
    ),
    "cogvideox-i2v": ModelSpec(
        key="cogvideox-i2v",
        family="cogvideox",
        pipeline_class="CogVideoXImageToVideoPipeline",
        hf_id="THUDM/CogVideoX-5b-I2V",
        local_dir="CogVideoX-5b-I2V",
        fps=8,
        default_frames=49,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
        sparse_supported=True,
    ),
    "ltx-video": ModelSpec(
        key="ltx-video",
        family="ltx_video",
        pipeline_class="LTXPipeline",
        hf_id="Lightricks/LTX-Video",
        local_dir="ltx-video",
        fps=25,
        default_frames=161,
        default_steps=50,
        guidance_scale=3.0,
        output_type="pil",
        sparse_supported=True,
    ),
    "ltx-video-i2v": ModelSpec(
        key="ltx-video-i2v",
        family="ltx_video",
        pipeline_class="LTXImageToVideoPipeline",
        hf_id="Lightricks/LTX-Video",
        local_dir="ltx-video",
        fps=25,
        default_frames=161,
        default_steps=50,
        guidance_scale=3.0,
        output_type="pil",
        sparse_supported=True,
    ),
    "allegro": ModelSpec(
        key="allegro",
        family="allegro",
        pipeline_class="AllegroPipeline",
        hf_id="rhymes-ai/Allegro",
        local_dir="allegro",
        fps=15,
        default_frames=88,
        default_steps=100,
        guidance_scale=7.5,
        output_type="pil",
        sparse_supported=True,
    ),
    "mochi-1": ModelSpec(
        key="mochi-1",
        family="mochi",
        pipeline_class="MochiPipeline",
        hf_id="genmo/mochi-1-preview",
        local_dir="mochi-1",
        fps=8,
        default_frames=19,
        default_steps=64,
        guidance_scale=4.5,
        output_type="pil",
        sparse_supported=True,
    ),
    "easyanimate-v5-t2v-12b": ModelSpec(
        key="easyanimate-v5-t2v-12b",
        family="easyanimate",
        pipeline_class="EasyAnimatePipeline",
        hf_id="alibaba-pai/EasyAnimateV5.1-12b-zh-diffusers",
        local_dir="easyanimate-v5-t2v-12b",
        fps=8,
        default_frames=49,
        default_steps=50,
        guidance_scale=5.0,
        output_type="pil",
        sparse_supported=True,
    ),
    "sana-video": ModelSpec(
        key="sana-video",
        family="sana_video",
        pipeline_class="SanaVideoPipeline",
        hf_id="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        local_dir="sana-video",
        fps=24,
        default_frames=17,
        default_steps=20,
        guidance_scale=5.0,
        output_type="pil",
        sparse_supported=False,
        sparse_methods=(),
        compatibility_label="incompatible",
        unsupported_reason=(
            "SanaVideo uses Diffusers' SanaLinearAttnProcessor3_0 linear attention, "
            "not softmax QK^T V attention; current SparseVideo sparse-softmax methods "
            "are incompatible."
        ),
    ),
    "motif-video": ModelSpec(
        key="motif-video",
        family="motif_video",
        pipeline_class="UnavailablePipeline",
        hf_id="",
        local_dir=None,
        fps=24,
        default_frames=1,
        default_steps=1,
        guidance_scale=1.0,
        output_type="pil",
        sparse_supported=False,
        sparse_methods=(),
        compatibility_label="unknown",
        unsupported_reason=(
            "MotifVideo is not available in the current Diffusers installation "
            "and no confirmed local/Hugging Face checkpoint is configured, so "
            "SparseVideo cannot verify a processor-swap path."
        ),
    ),
    "ltx-video-2": ModelSpec(
        key="ltx-video-2",
        family="ltx_video_2",
        pipeline_class="UnavailablePipeline",
        hf_id="",
        local_dir=None,
        fps=24,
        default_frames=1,
        default_steps=1,
        guidance_scale=1.0,
        output_type="pil",
        sparse_supported=False,
        sparse_methods=(),
        compatibility_label="unknown",
        unsupported_reason=(
            "LTX Video 2 is not available in the current Diffusers installation; "
            "SparseVideo cannot verify the requested video attn1 plus audio_attn1 "
            "structure or safely reuse the plain LTX Video processor."
        ),
    ),
    "kandinsky5-t2v": ModelSpec(
        key="kandinsky5-t2v",
        family="kandinsky5",
        pipeline_class="Kandinsky5T2VPipeline",
        hf_id="ai-forever/Kandinsky-5.0-T2V",
        local_dir="kandinsky5-t2v",
        fps=12,
        default_frames=49,
        default_steps=50,
        guidance_scale=5.0,
        output_type="pil",
        sparse_supported=False,
        sparse_methods=(),
        compatibility_label="native-N/A",
        unsupported_reason=(
            "Kandinsky5 exposes native sparse attention controls through transformer "
            "sparse_params/window parameters, so it is not a SparseVideo processor-swap target."
        ),
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
    "hunyuan-i2v": "hunyuan-i2v",
    "skyreels": "skyreels-v2-t2v-14b",
    "skyreels-v2": "skyreels-v2-t2v-14b",
    "skyreels-v2-t2v": "skyreels-v2-t2v-14b",
    "skyreels-v2-t2v-14b": "skyreels-v2-t2v-14b",
    "skyreels-i2v": "skyreels-v2-i2v-14b",
    "skyreels-v2-i2v": "skyreels-v2-i2v-14b",
    "skyreels-v2-i2v-14b": "skyreels-v2-i2v-14b",
    "wananimate": "wan22-animate-14b",
    "wan-animate": "wan22-animate-14b",
    "wan22-animate": "wan22-animate-14b",
    "wan22-animate-14b": "wan22-animate-14b",
    "vace": "wan21-vace-1.3b",
    "wan-vace": "wan21-vace-1.3b",
    "wan21-vace": "wan21-vace-1.3b",
    "wan21-vace-1.3b": "wan21-vace-1.3b",
    "wan21-vace-14b": "wan21-vace-14b",
    "wan-i2v": "wan21-i2v-14b",
    "wan14b-i2v": "wan21-i2v-14b",
    "wan21-i2v": "wan21-i2v-14b",
    "wan21-i2v-14b": "wan21-i2v-14b",
    "wan22-i2v": "wan22-i2v-a14b",
    "wan22-i2v-a14b": "wan22-i2v-a14b",
    "cog": "cogvideox-t2v",
    "cogvideox": "cogvideox-t2v",
    "cogvideox-t2v": "cogvideox-t2v",
    "cog-i2v": "cogvideox-i2v",
    "cogvideox-i2v": "cogvideox-i2v",
    "cogvideox-5b-i2v": "cogvideox-i2v",
    "ltx": "ltx-video",
    "ltx-video": "ltx-video",
    "ltx-i2v": "ltx-video-i2v",
    "ltx-video-i2v": "ltx-video-i2v",
    "allegro": "allegro",
    "mochi": "mochi-1",
    "mochi-1": "mochi-1",
    "easyanimate": "easyanimate-v5-t2v-12b",
    "easyanimate-v5": "easyanimate-v5-t2v-12b",
    "easyanimate-v5-t2v-12b": "easyanimate-v5-t2v-12b",
    "motif": "motif-video",
    "motif-video": "motif-video",
    "ltx2": "ltx-video-2",
    "ltx-video-2": "ltx-video-2",
    "sana-video": "sana-video",
    "sanavideo": "sana-video",
    "kandinsky5": "kandinsky5-t2v",
    "kandinsky5-t2v": "kandinsky5-t2v",
}


WAN_SAMPLE_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


HUNYUAN_VIDEO_NEGATIVE_PROMPT = (
    "Aerial view, aerial view, overexposed, low quality, deformation, "
    "a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
)


UPSTREAM_INFERENCE_PROFILES: Dict[tuple[str, str], Dict[str, Any]] = {
    # Sparse-VideoGen/scripts/wan/wan_t2v_720p_svg.sh
    ("svg1", "wan21-t2v-14b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 50,
        "fps": 16,
        "flow_shift": 5.0,
        "source": "training_free/Sparse-VideoGen/scripts/wan/wan_t2v_720p_svg.sh",
    },
    # Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_svg.sh
    ("svg1", "hunyuan_video"): {
        "height": 720,
        "width": 1280,
        "num_frames": 129,
        "num_inference_steps": 50,
        "fps": 24,
        "flow_shift": 7.0,
        "vae_tiling": True,
        "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_svg.sh",
    },
    # Sparse-VideoGen/scripts/wan/wan_t2v_720p_sap.sh; public SparseVideo name is svg2.
    ("svg2", "wan21-t2v-14b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 50,
        "fps": 16,
        "flow_shift": 5.0,
        "source": "training_free/Sparse-VideoGen/scripts/wan/wan_t2v_720p_sap.sh",
    },
    # Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_sap.sh; public SparseVideo name is svg2.
    ("svg2", "hunyuan_video"): {
        "height": 720,
        "width": 1280,
        "num_frames": 129,
        "num_inference_steps": 50,
        "fps": 24,
        "flow_shift": 7.0,
        "vae_tiling": True,
        "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_sap.sh",
    },
    # SpargeAttn/inference_examples/wan_infer.py defaults to --mode full.
    ("spargeattn", "wan21-t2v-1.3b"): {
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 15,
        "guidance_scale": 5.0,
        "seed": 42,
        "cpu_offload": True,
        "cpu_offload_mode": "sequential",
        "vae_tiling": True,
        "vae_slicing": True,
        "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/wan_infer.py",
    },
    ("spargeattn", "wan21-t2v-14b"): {
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 15,
        "guidance_scale": 5.0,
        "seed": 42,
        "cpu_offload": True,
        "cpu_offload_mode": "sequential",
        "vae_tiling": True,
        "vae_slicing": True,
        "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/wan_infer.py",
    },
    # SpargeAttn/inference_examples/README.md shows Wan2.2 with --mode topk --value 0.4.
    ("spargeattn", "wan22-t2v-a14b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 40,
        "fps": 16,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "seed": 42,
        "cpu_offload": True,
        "cpu_offload_mode": "sequential",
        "vae_tiling": True,
        "vae_slicing": True,
        "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "topk", "value": 0.4},
        "source": "training_free/SpargeAttn/inference_examples/README.md",
        "evidence_sources": ["training_free/SpargeAttn/inference_examples/wan_infer.py"],
    },
    # SpargeAttn/inference_examples/hunyuan_infer.py defaults to --mode full.
    ("spargeattn", "hunyuan_video"): {
        "height": 320,
        "width": 512,
        "num_frames": 61,
        "num_inference_steps": 30,
        "fps": 8,
        "seed": 42,
        "cpu_offload": True,
        "cpu_offload_mode": "sequential",
        "vae_tiling": True,
        "vae_slicing": True,
        "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/hunyuan_infer.py",
    },
    # radial-attention/wan_t2v_inference.py defaults to Wan2.1 T2V 14B.
    ("radial", "wan21-t2v-14b"): {
        "height": 768,
        "width": 1280,
        "num_frames": 69,
        "num_inference_steps": 50,
        "flow_shift": 5.0,
        "source": "training_free/radial-attention/scripts/wan_t2v_inference.sh",
    },
    # radial-attention/scripts/wan_22_t2v_inference.sh
    ("radial", "wan22-t2v-a14b"): {
        "height": 768,
        "width": 1280,
        "num_frames": 77,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "vae_tiling": True,
        "vae_slicing": False,
        "source": "training_free/radial-attention/scripts/wan_22_t2v_inference.sh",
    },
    # radial-attention/scripts/hunyuan_t2v_inference.sh
    ("radial", "hunyuan_video"): {
        "height": 768,
        "width": 1280,
        "num_frames": 117,
        "num_inference_steps": 50,
        "vae_tiling": True,
        "vae_slicing": False,
        "source": "training_free/radial-attention/scripts/hunyuan_t2v_inference.sh",
    },
    # FastVideo current checkout keeps STA inference as archived workflow docs.
    ("sta", "wan21-t2v-14b"): {
        "height": 768,
        "width": 1280,
        "num_frames": 69,
        "num_inference_steps": 50,
        "method_config": {"seq_shape": "18x48x80"},
        "source": "training_free/FastVideo/docs/attention/sta/index.md",
    },
    ("sta", "hunyuan_video"): {
        "height": 768,
        "width": 1280,
        "num_frames": 117,
        "num_inference_steps": 50,
        "method_config": {"seq_shape": "30x48x80"},
        "source": "training_free/FastVideo/docs/attention/sta/index.md",
    },
    # draft-attention/wan/run-single-inference.sh uses Wan2.1 T2V 14B at --size 768*512.
    ("draft", "wan21-t2v-14b"): {
        "height": 512,
        "width": 768,
        "num_frames": 81,
        "num_inference_steps": 50,
        "flow_shift": 5.0,
        "seed": 42,
        "method_config": {
            "latent_h": 32,
            "latent_w": 48,
            "visual_len": 32_256,
            "text_len": 0,
            "batch_size": 1,
        },
        "source": "training_free/draft-attention/wan/run-single-inference.sh",
        "evidence_sources": [
            "training_free/draft-attention/wan/generate.py",
            "training_free/draft-attention/wan/wan/configs/shared_config.py",
        ],
    },
    # draft-attention README/Hunyuan demo uses Hunyuan 768p at 129 frames.
    ("draft", "hunyuan_video"): {
        "height": 768,
        "width": 1280,
        "num_frames": 129,
        "num_inference_steps": 50,
        "seed": 42,
        "cpu_offload": True,
        "cpu_offload_mode": "sequential",
        "method_config": {
            "latent_h": 48,
            "latent_w": 80,
            "visual_len": 126_720,
            "text_len": 256,
        },
        "source": "training_free/draft-attention/README.md",
    },
    # Adacluster/runwan/runwan.py uses Wan2.1 T2V 1.3B at --size 832*480.
    ("adacluster", "wan21-t2v-1.3b"): {
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "num_inference_steps": 50,
        "fps": 16,
        "flow_shift": 5.0,
        "negative_prompt": WAN_SAMPLE_NEGATIVE_PROMPT,
        "source": "training_free/Adacluster/runwan/runwan.py",
        "evidence_sources": [
            "training_free/Adacluster/runwan/generate.py",
            "training_free/Adacluster/runwan/wan/configs/shared_config.py",
        ],
    },
    # Adacluster/runhunyuan/run_hunyuan.py hardcodes 720p, 81 frames, 30 steps, fps=15.
    ("adacluster", "hunyuan_video"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 30,
        "fps": 15,
        "cpu_offload": True,
        "cpu_offload_mode": "model",
        "vae_tiling": True,
        "vae_slicing": False,
        "source": "training_free/Adacluster/runhunyuan/run_hunyuan.py",
    },
    # SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh.
    ("svoo", "wan21-t2v-1.3b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 50,
        "fps": 16,
        # SVOO's Wan script keeps the model scheduler config, which is flow_shift=3.0
        # in local Wan Diffusers checkpoints, and loads the VAE through
        # WanPipeline.from_pretrained(..., torch_dtype=torch.bfloat16).
        "flow_shift": 3.0,
        "vae_dtype": "bf16",
        "vae_tiling": False,
        "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    ("svoo", "wan21-t2v-14b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 50,
        "fps": 16,
        "flow_shift": 3.0,
        "vae_dtype": "bf16",
        "vae_tiling": False,
        "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    ("svoo", "wan22-t2v-a14b"): {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 40,
        "fps": 16,
        "guidance_scale": 5.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 3.0,
        "vae_dtype": "bf16",
        "vae_tiling": False,
        "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    # SVOO/scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh.
    ("svoo", "hunyuan_video"): {
        "height": 720,
        "width": 1280,
        "num_frames": 129,
        "num_inference_steps": 50,
        "fps": 24,
        "seed": 23,
        "flow_shift": 7.0,
        "vae_tiling": True,
        "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/SVOO/scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/hunyuan10_t2v_inference.py"],
    },
}


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
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


def materialize_method_config_values(method: str, config: Dict[str, Any]) -> None:
    tensor_keys: tuple[str, ...] = ()
    if method == "flashomni":
        tensor_keys = FLASHOMNI_SPARSE_INFO_KEYS
    elif method == "spargeattn":
        tensor_keys = ("mask_id",)
    for key in tensor_keys:
        value = config.get(key)
        if isinstance(value, str):
            config[key] = load_torch_tensor_config_value(method, key, value)


def load_torch_tensor_config_value(method: str, key: str, value: str):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{method} {key} tensor path does not exist: {path}")

    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if torch.is_tensor(loaded):
        return loaded
    if isinstance(loaded, dict) and key in loaded and torch.is_tensor(loaded[key]):
        return loaded[key]
    raise TypeError(
        f"{method} {key} must load to a torch.Tensor, or to a dict containing "
        f"a torch.Tensor under key {key!r}; got {type(loaded).__name__} from {path}"
    )


def sync_flashomni_config_aliases(config: Dict[str, Any]) -> None:
    pairs = (
        ("threshold_q", "tau_q", 0.50),
        ("threshold_kv", "tau_kv", 0.05),
        ("fresh_threshold", "N", 6),
        ("max_order", "D", 1),
        ("saving_threshold_q_for_taylor", "S_q", 0.3),
    )
    for primary, legacy, default in pairs:
        primary_set = primary in config
        legacy_set = legacy in config
        if primary_set and legacy_set and config[primary] != config[legacy]:
            if config[legacy] == default:
                config[legacy] = config[primary]
                continue
            if config[primary] == default:
                config[primary] = config[legacy]
                continue
            raise ValueError(
                f"flashomni config keys {primary!r} and {legacy!r} refer to the same upstream "
                "setting but have different non-default values"
            )
        if primary_set:
            config[legacy] = config[primary]
        elif legacy_set:
            config[primary] = config[legacy]


def apply_flashomni_hunyuan_quality_defaults(
    spec: ModelSpec,
    config: Dict[str, Any],
    user_config: Dict[str, Any],
) -> None:
    if spec.family != "hunyuan_video":
        return
    if config.get("sparse_pattern") != "paper_mmdit":
        return

    if not any(key in user_config for key in ("max_order", "D")):
        config["max_order"] = 0
        config["D"] = 0
    if "use_sparse_gemm" not in user_config:
        config["use_sparse_gemm"] = False


def validate_flashomni_hunyuan_quality_lock(config: Dict[str, Any], model_family: Optional[str]) -> None:
    if model_family != "hunyuan_video":
        return
    if config.get("sparse_pattern") != "paper_mmdit":
        return
    if not bool(config.get("use_sparse_gemm", False)):
        return

    # Sparse GEMM projection is intentionally kept for source audit and future
    # repair, but it is not an allowed Hunyuan inference path: matched 50-step
    # tests showed visual shining artifacts and slower runtime.
    raise NotImplementedError(
        "flashomni Hunyuan paper_mmdit only supports the quality-safe inference path "
        "with use_sparse_gemm=false. Sparse GEMM projection (use_sparse_gemm=true) is "
        "currently disabled because measured Hunyuan runs showed quality degradation "
        "and performance regression. The code is retained for audit/future repair, "
        "but this path is not supported for inference."
    )


def default_svoo_sparsity_csv_path(spec: ModelSpec) -> str:
    profile_dir = SRC_ROOT / "sparsevideo" / "methods" / "svoo" / "sparsity_profiles"
    if spec.key == "hunyuan-i2v":
        sparsity_csv = profile_dir / "sparsity_hunyuan10_13B_i2v.csv"
    elif spec.family == "hunyuan_video":
        sparsity_csv = profile_dir / "sparsity_hunyuan10_13B_t2v.csv"
    elif spec.key == "wan22-i2v-a14b":
        sparsity_csv = profile_dir / "sparsity_wan22_A14B_i2v.csv"
    elif spec.key == "wan22-t2v-a14b":
        sparsity_csv = profile_dir / "sparsity_wan22_A14B_t2v.csv"
    elif spec.key == "wan21-i2v-14b":
        sparsity_csv = profile_dir / "sparsity_wan_14B_i2v.csv"
    elif spec.key == "wan21-t2v-14b":
        sparsity_csv = profile_dir / "sparsity_wan_14B_t2v.csv"
    elif spec.key == "wan21-t2v-1.3b":
        sparsity_csv = profile_dir / "sparsity_wan_1.3B_t2v.csv"
    else:
        raise ValueError(
            f"SVOO has no owned offline sparsity profile for {spec.key}; "
            "leave use_dynamic_min_kc_ratio=false to skip the offline sparsity "
            "profile stage and use online co-clustering with the fixed "
            "min_kc_ratio, or provide an explicit owned "
            "sparsity_csv_path."
        )

    return str(sparsity_csv)


def resolve_inference_profile(profile: str, spec: ModelSpec, method: str) -> Dict[str, Any]:
    if profile == "default":
        return {}
    if profile != "upstream":
        raise ValueError(f"Unknown inference profile: {profile}")

    exact = UPSTREAM_INFERENCE_PROFILES.get((method, spec.key))
    family = UPSTREAM_INFERENCE_PROFILES.get((method, spec.family))
    selected = exact if exact is not None else family
    if selected is None:
        raise ValueError(
            f"No upstream inference profile is defined for method={method!r}, model={spec.key!r}. "
            "Use --profile default, or choose a method/model pair with a referenced training_free benchmark profile."
        )
    return copy.deepcopy(selected)


def apply_profile_runtime_defaults(
    args: argparse.Namespace,
    profile: Dict[str, Any],
    fps: int,
    num_frames: int,
    steps: int,
) -> tuple[int, int, int, int, int]:
    height = args.height if args.height is not None else int(profile.get("height", DEFAULT_HEIGHT))
    width = args.width if args.width is not None else int(profile.get("width", DEFAULT_WIDTH))

    if args.fps is None and "fps" in profile:
        fps = int(profile["fps"])

    if args.num_frames is None and args.duration_seconds is None and "num_frames" in profile:
        num_frames = int(profile["num_frames"])
    if args.num_inference_steps is None and "num_inference_steps" in profile:
        steps = int(profile["num_inference_steps"])
    if args.guidance_scale is None and "guidance_scale" in profile:
        args.guidance_scale = float(profile["guidance_scale"])
    if args.guidance_scale_2 == 3.0 and "guidance_scale_2" in profile:
        args.guidance_scale_2 = float(profile["guidance_scale_2"])
    if args.true_cfg_scale == 1.0 and "true_cfg_scale" in profile:
        args.true_cfg_scale = float(profile["true_cfg_scale"])
    if args.flow_shift is None and "flow_shift" in profile:
        args.flow_shift = float(profile["flow_shift"])
    if args.vae_dtype is None and "vae_dtype" in profile:
        args.vae_dtype = str(profile["vae_dtype"])
    if args.negative_prompt is None and "negative_prompt" in profile:
        args.negative_prompt = str(profile["negative_prompt"])
    if args.seed is None and "seed" in profile:
        args.seed = int(profile["seed"])
    if args.cpu_offload is None and "cpu_offload" in profile:
        args.cpu_offload = bool(profile["cpu_offload"])
    if args.cpu_offload_mode is None and "cpu_offload_mode" in profile:
        args.cpu_offload_mode = str(profile["cpu_offload_mode"])
    if args.vae_tiling is None and "vae_tiling" in profile:
        args.vae_tiling = bool(profile["vae_tiling"])
    if args.vae_slicing is None and "vae_slicing" in profile:
        args.vae_slicing = bool(profile["vae_slicing"])
    if args.vae_decoder_chunk_size is None and "vae_decoder_chunk_size" in profile:
        args.vae_decoder_chunk_size = int(profile["vae_decoder_chunk_size"])
    if args.seed is None:
        args.seed = DEFAULT_SEED
    if args.cpu_offload is None:
        args.cpu_offload = False
    if args.cpu_offload_mode is None:
        args.cpu_offload_mode = "model"
    if args.vae_tiling is None:
        args.vae_tiling = False
    if args.vae_slicing is None:
        args.vae_slicing = False
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT

    args.height = height
    args.width = width
    if args.num_inference_steps is None:
        args.num_inference_steps = steps
    return height, width, fps, num_frames, steps


def finalize_runtime_defaults(args: argparse.Namespace) -> None:
    if args.seed is None:
        args.seed = DEFAULT_SEED
    if args.cpu_offload is None:
        args.cpu_offload = False
    if args.cpu_offload_mode is None:
        args.cpu_offload_mode = "model"
    if args.vae_tiling is None:
        args.vae_tiling = False
    if args.vae_slicing is None:
        args.vae_slicing = False


def normalize_spargeattn_model_out_path(config: Dict[str, Any], output_file: Path) -> None:
    if not config.get("tune") and not config.get("model_out_path"):
        return
    if config.get("tune") and not config.get("model_out_path"):
        config["model_out_path"] = str(output_file.with_suffix(".spargeattn_state.pt"))
        return
    path = Path(str(config["model_out_path"])).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    config["model_out_path"] = str(path)


def validate_method_config(method: str, config: Dict[str, Any], model_family: Optional[str] = None) -> None:
    if method == "spargeattn":
        if config.get("mode", "full") not in ("full", "cdfthreshd", "topk", "block_sparse"):
            raise ValueError("spargeattn mode must be full, cdfthreshd, topk, or block_sparse")
        if config.get("tensor_layout", "HND") != "HND":
            raise ValueError("spargeattn SparseVideo processor uses tensor_layout=HND")
        if config.get("return_sparsity", False):
            raise NotImplementedError("spargeattn return_sparsity=true is not supported inside inference processors")
        if config.get("mode", "full") == "block_sparse" and config.get("mask_id") is None:
            raise ValueError("spargeattn mode=block_sparse requires --method-config mask_id=<torch tensor path>")
        if config.get("pv_l1", 0.08) <= config.get("l1", 0.07):
            raise ValueError("spargeattn pv_l1 must be greater than l1")
        if config.get("sim_rule", "l1") not in ("l1", "cosine", "rmse"):
            raise ValueError("spargeattn sim_rule must be l1, cosine, or rmse")
        if not isinstance(config.get("rearrange_kwargs", {}), dict):
            raise TypeError("spargeattn rearrange_kwargs must be a JSON object")
        if config.get("model_out_path") and not config.get("tune"):
            path = Path(str(config["model_out_path"])).expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"spargeattn model_out_path does not exist: {path}")
    if method == "radial" and config.get("block_size") not in (64, 128):
        raise ValueError("radial block_size must be 64 or 128")
    if method == "flashomni":
        sync_flashomni_config_aliases(config)
        if config.get("implementation") not in ("upstream", "flex"):
            raise ValueError("flashomni implementation must be upstream or flex")
        if config.get("backend") not in ("auto", "fa2", "fa3"):
            raise ValueError("flashomni backend must be auto, fa2, or fa3")
        if config.get("pos_encoding_mode") not in ("NONE", "ROPE_LLAMA", "ALIBI"):
            raise ValueError("flashomni pos_encoding_mode must be NONE, ROPE_LLAMA, or ALIBI")
        if config.get("sparse_pattern") not in ("explicit", "global_random", "paper_mmdit", "local_qk_topk"):
            raise ValueError(
                "flashomni sparse_pattern must be explicit, global_random, paper_mmdit, or local_qk_topk"
            )
        for key in ("threshold_q", "threshold_kv", "saving_threshold_q_for_taylor"):
            value = float(config.get(key, 0.0))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"flashomni {key} must be in [0, 1]")
        if int(config.get("fresh_threshold", 1)) < 1:
            raise ValueError("flashomni fresh_threshold must be >= 1")
        if int(config.get("max_order", 0)) not in (0, 1, 2):
            raise ValueError("flashomni max_order must be 0, 1, or 2")
        if int(config.get("first_enhance", 0)) < 0:
            raise ValueError("flashomni first_enhance must be >= 0")
        if int(config.get("num_inference_steps", 1)) < 1:
            raise ValueError("flashomni num_inference_steps must be >= 1")
        if not isinstance(config.get("use_sparse_gemm", True), bool):
            raise ValueError("flashomni use_sparse_gemm must be a boolean")
        if config.get("implementation") == "flex" and config.get("sparse_pattern") != "local_qk_topk":
            raise NotImplementedError(
                "flashomni implementation=flex is only available for sparse_pattern=local_qk_topk"
            )
        if config.get("sparse_pattern") == "global_random":
            config["sparse_block_size_for_q"] = int(config.get("sparse_size", 128))
            config["sparse_block_size_for_kv"] = int(config.get("sparse_size", 128))
        validate_flashomni_hunyuan_quality_lock(config, model_family)
        bad = [key for key in FLASHOMNI_SPARSE_INFO_KEYS if config.get(key) is not None and not is_torch_tensor(config.get(key))]
        if bad:
            raise TypeError(
                "FlashOmni sparse-info inputs must be torch.Tensor values or CLI paths "
                f"to torch-saved tensors. Bad keys: {bad}"
            )
    if method == "draft" and not config.get("block_sparse_attention"):
        raise NotImplementedError(
            "draft block_sparse_attention=False disables the upstream sparse path; "
            "use --method dense for the dense baseline."
        )
    if method == "sta" and config.get("STA_mode", "STA_inference") not in ("STA_inference", "STA_searching"):
        raise NotImplementedError(
            "sta supports STA_inference in pipelines and STA_searching for mask calibration; "
            "use python -m sparsevideo.methods.sta.search tune for STA_tuning."
        )
    if method == "svoo":
        if config.get("implementation") != "native":
            raise NotImplementedError(
                "svoo implementation must be native; SparseVideo no longer uses training_free runtime bridges"
            )
        if config.get("sparse_backend") not in ("flashinfer", "triton"):
            raise ValueError("svoo sparse_backend must be flashinfer or triton")
        if config.get("use_dynamic_min_kc_ratio"):
            csv_path = config.get("sparsity_csv_path")
            if not csv_path:
                raise ValueError("svoo use_dynamic_min_kc_ratio requires sparsity_csv_path")
            path = Path(str(csv_path)).expanduser()
            if not path.is_absolute():
                path = REPO_ROOT / path
            resolved_path = path.resolve(strict=False)
            if "training_free" in path.parts or "training_free" in resolved_path.parts:
                raise RuntimeError(
                    "Refusing SVOO sparsity_csv_path inside training_free; "
                    "SparseVideo runtime sparsity profiles must live under src/sparsevideo."
                )
            path = resolved_path
            if not path.exists():
                raise FileNotFoundError(
                    f"svoo use_dynamic_min_kc_ratio requires an existing sparsity_csv_path: {path}"
                )
            config["sparsity_csv_path"] = str(path)

        if config.get("use_svoo", True):
            for key in ("kmeans_iter_init", "kmeans_iter_step"):
                if key in config and int(config.get(key, 0)) <= 0:
                    raise ValueError(f"svoo use_svoo=true requires {key} > 0")


def normalize_seq_shape_for_warning(seq_shape: Any) -> Optional[str]:
    if seq_shape is None:
        return None
    if isinstance(seq_shape, str):
        return seq_shape.lower()
    if isinstance(seq_shape, (list, tuple)) and len(seq_shape) == 3:
        return "x".join(str(int(part)) for part in seq_shape)
    return str(seq_shape)


def _parse_normalized_seq_shape(seq_shape: str) -> Optional[tuple[int, int, int]]:
    parts = seq_shape.split("x")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _normalize_int_triple(value: Any) -> Optional[tuple[int, int, int]]:
    if isinstance(value, str):
        value = value.replace("x", ",").split(",")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(part) for part in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _sta_mask_strategy_shape(path: Any) -> tuple[int, int, int]:
    strategy_path = Path(str(path)).expanduser()
    if not strategy_path.is_absolute():
        strategy_path = (REPO_ROOT / strategy_path).resolve()
    if "training_free" in strategy_path.parts:
        raise RuntimeError(
            "Refusing STA mask_strategy_file_path inside training_free; "
            "SparseVideo runtime mask strategies must live under src/sparsevideo."
        )
    with strategy_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"STA mask strategy must be a non-empty JSON object: {strategy_path}")

    timesteps = []
    layers = []
    heads = []
    for key in data:
        t_idx, layer_idx, head_idx = (int(part) for part in str(key).split("_"))
        timesteps.append(t_idx)
        layers.append(layer_idx)
        heads.append(head_idx)
    return max(timesteps) + 1, max(layers) + 1, max(heads) + 1


def _flash_attn_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_func: str,
    requirement: str,
) -> Optional[str]:
    flash_attn = kernels.get("flash_attn", {})
    if flash_attn.get("load_checked") and flash_attn.get("import_error"):
        return (
            f"{requirement}. flash_attn failed to import during preflight: "
            f"{flash_attn.get('import_error_type')}: {flash_attn.get('import_error')}."
        )
    missing = []
    if not flash_attn.get("package") and not flash_attn.get("imported"):
        missing.append("flash_attn")
    if not flash_attn.get(required_func):
        missing.append(required_func)
    if not missing:
        return None
    return f"{requirement}. Missing: {missing}."


def _flashinfer_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    flashinfer = kernels.get("flashinfer", {})
    if not flashinfer.get("load_checked"):
        return (
            f"{requirement}. benchmark preflight must import FlashInfer and flashinfer.sparse; "
            "package/source presence alone is not enough to claim parity."
        )
    if flashinfer.get("import_error"):
        return (
            f"{requirement}. flashinfer failed to import during preflight: "
            f"{flashinfer.get('import_error_type')}: {flashinfer.get('import_error')}."
        )
    missing = []
    if not flashinfer.get("imported"):
        missing.append("flashinfer")
    if not flashinfer.get("sparse_imported"):
        missing.append("flashinfer.sparse")
    for attr in required_attrs:
        if not flashinfer.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing FlashInfer API(s): {missing}."


def _spas_sage_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    sparge = kernels.get("spas_sage_attn", {})
    if not sparge.get("load_checked"):
        return None
    if sparge.get("import_error"):
        return (
            f"{requirement}. spas_sage_attn failed to import during preflight: "
            f"{sparge.get('import_error_type')}: {sparge.get('import_error')}."
        )
    missing = []
    if not sparge.get("imported"):
        missing.append("spas_sage_attn")
    for attr in required_attrs:
        if not sparge.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing spas_sage_attn API(s): {missing}."


def _sageattention_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    requirement: str,
) -> Optional[str]:
    sageattention = kernels.get("sageattention", {})
    if not sageattention.get("load_checked"):
        return None
    if sageattention.get("import_error"):
        return (
            f"{requirement}. sageattention failed to import during preflight: "
            f"{sageattention.get('import_error_type')}: {sageattention.get('import_error')}."
        )
    missing = []
    if not sageattention.get("imported"):
        missing.append("sageattention")
    if not sageattention.get("sageattn"):
        missing.append("sageattn")
    if not missing:
        return None
    return f"{requirement}. Missing sageattention API(s): {missing}."


def _flashomni_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    flashomni = kernels.get("flashomni", {})
    if not flashomni.get("load_checked"):
        return None
    if flashomni.get("import_error"):
        return (
            f"{requirement}. flashomni failed to import during preflight: "
            f"{flashomni.get('import_error_type')}: {flashomni.get('import_error')}."
        )
    missing = []
    if not flashomni.get("imported"):
        missing.append("flashomni")
    for attr in required_attrs:
        if not flashomni.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing FlashOmni API(s): {missing}."


def preflight_runtime(
    method: str,
    config: Dict[str, Any],
    device: str,
    runtime_status: Dict[str, Any],
    strict_kernels: bool = True,
    model_family: Optional[str] = None,
) -> Dict[str, Any]:
    kernels = runtime_status["optional_kernels"]
    torch_status = runtime_status["torch"]
    errors = []
    warnings = []

    if device.startswith("cuda"):
        if not torch_status.get("cuda_available"):
            errors.append(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )
    elif method != "dense":
        errors.append("Sparse methods require --device cuda for fair inference benchmarking.")

    fused = kernels["svg_svoo_fused_kernels"]
    fused_unavailable_message = (
        "SparseVideo _kernels extension is not detected; RMSNorm/RoPE will use the Triton/PyTorch "
        "path. Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton to benchmark that path explicitly."
    )
    if fused.get("native_load_checked") and fused.get("native_import_error") and not fused.get("native_extension"):
        native_error = str(fused.get("native_import_error")).rstrip(".")
        if fused.get("built_extension"):
            fused_unavailable_message = (
                "SparseVideo _kernels extension is built but failed to load; RMSNorm/RoPE will use the "
                "Triton/PyTorch path. "
                f"Import error: {fused.get('native_import_error_type')}: {native_error}. "
                "Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton only for an explicit non-native benchmark."
            )
        else:
            fused_unavailable_message = (
                "SparseVideo _kernels extension is unavailable; RMSNorm/RoPE will use the Triton/PyTorch path. "
                f"Native root error: {fused.get('native_import_error_type')}: {native_error}. "
                "Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton only for an explicit non-native benchmark."
            )
    if method in ("svg1", "svg2", "svoo") and fused.get("backend_env") == "native":
        if not fused.get("native_extension"):
            errors.append(
                "SPARSEVIDEO_FUSED_KERNEL_BACKEND=native requires a loadable SparseVideo _kernels extension. "
                f"{fused_unavailable_message} "
                f"Searched: {fused.get('candidate_dirs')}."
            )
    elif (
        method in ("svg1", "svg2", "svoo")
        and fused.get("backend_env") == "auto"
        and not fused.get("native_extension")
    ):
        if strict_kernels:
            errors.append(fused_unavailable_message)
        else:
            warnings.append(fused_unavailable_message)

    spargeattn_needs_runtime = (
        method == "spargeattn"
        and (config.get("mode") != "full" or config.get("tune") or config.get("model_out_path"))
    )
    if spargeattn_needs_runtime:
        if model_family == "hunyuan_video":
            errors.append(
                "spargeattn sparse/tuned paths are not upstream-equivalent for HunyuanVideo here: "
                "Hunyuan passes attention_mask, but upstream spas_sage_attn sparse kernels do not "
                "support attention_mask. The upstream Hunyuan profile uses mode=full; use mode=full "
                "as the dense baseline until a faithful Hunyuan sparse path is ported."
            )
        sparge = kernels["spas_sage_attn"]
        sparge_env_error = (sparge.get("env_root") or {}).get("error")
        sparsevideo_runtime = sparge.get("sparsevideo_runtime", {})
        sparsevideo_ready = (
            sparsevideo_runtime.get("package")
            and sparsevideo_runtime.get("qattn_extension")
            and sparsevideo_runtime.get("fused_extension")
        )
        if config.get("tune") or config.get("model_out_path"):
            sparsevideo_ready = (
                sparsevideo_ready
                and sparsevideo_runtime.get("autotune")
                and sparsevideo_runtime.get("gpu_process_pool")
            )
        if sparge_env_error:
            errors.append(sparge_env_error)
        elif sparsevideo_ready:
            required_attrs = (
                ("block_sparse_sage2_attn_cuda",)
                if config.get("mode") == "block_sparse"
                else ("spas_sage2_attn_meansim_cuda", "spas_sage2_attn_meansim_topk_cuda")
            )
            if config.get("tune") or config.get("model_out_path"):
                required_attrs = ("autotune",)
            if not sparge.get("load_checked"):
                errors.append(
                    "spargeattn sparse/tuned benchmark preflight must import the SparseVideo-owned "
                    "spas_sage_attn runtime; extension/source presence alone is not enough to claim parity."
                )
            else:
                load_error = _spas_sage_load_preflight_error(
                    kernels,
                    required_attrs=required_attrs,
                    requirement="spargeattn sparse/tuned paths require loadable SparseVideo-owned spas_sage_attn",
                )
                if load_error is not None:
                    errors.append(load_error)
        elif sparge.get("training_free_runtime") or sparge.get("training_free_package_detected"):
            errors.append(
                "spargeattn resolves spas_sage_attn from training_free/, which is reference-only. "
                "SparseVideo-owned source exists under src/sparsevideo/kernels/native/spargeattn, "
                "but its _qattn/_fused extensions are not built."
            )
        elif sparge.get("environment_runtime_detected") or (
            sparge.get("package") and sparge.get("qattn_extension") and sparge.get("fused_extension")
        ):
            errors.append(
                "spargeattn sparse modes require the SparseVideo-owned spas_sage_attn runtime under "
                "src/sparsevideo/kernels/native/spargeattn. Environment spas_sage_attn packages are "
                "not accepted for SparseVideo runtime parity."
            )
        else:
            errors.append(
                "spargeattn sparse modes require spas_sage_attn with _qattn and _fused extensions built. "
                "Build the SparseVideo-owned source at src/sparsevideo/kernels/native/spargeattn."
            )
    elif method == "spargeattn":
        warnings.append("spargeattn mode=full runs dense attention; set mode/value to benchmark sparse SpargeAttn.")

    if method == "radial" and config.get("use_sage_attention"):
        sparge = kernels["spas_sage_attn"]
        sparge_env_error = (sparge.get("env_root") or {}).get("error")
        sparsevideo_runtime = sparge.get("sparsevideo_runtime", {})
        sparsevideo_ready = (
            sparsevideo_runtime.get("package")
            and sparsevideo_runtime.get("qattn_extension")
            and sparsevideo_runtime.get("fused_extension")
            and sparsevideo_runtime.get("block_sparse_sage2_attn_cuda")
        )
        if sparge_env_error:
            errors.append(sparge_env_error)
        elif not sparsevideo_ready:
            errors.append(
                "radial use_sage_attention requires SparseVideo-owned spas_sage_attn "
                "with _qattn/_fused extensions and block_sparse_sage2_attn_cuda under "
                "src/sparsevideo/kernels/native/spargeattn."
            )
        elif sparge.get("training_free_runtime") and sparge.get("selected_runtime") != "sparsevideo":
            errors.append(
                "radial use_sage_attention resolves spas_sage_attn from training_free/, which is reference-only. "
                "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/spargeattn."
            )
        elif not sparge.get("load_checked"):
            errors.append(
                "radial use_sage_attention benchmark preflight must import the SparseVideo-owned "
                "spas_sage_attn runtime; extension/source presence alone is not enough to claim parity."
            )
        else:
            load_error = _spas_sage_load_preflight_error(
                kernels,
                required_attrs=("block_sparse_sage2_attn_cuda",),
                requirement="radial use_sage_attention requires loadable SparseVideo-owned spas_sage_attn",
            )
            if load_error is not None:
                errors.append(load_error)
        if int(config.get("dense_timesteps", 0) or 0) > 0 or int(config.get("dense_layers", 0) or 0) > 0:
            sageattention = kernels.get("sageattention", {})
            sage_env_error = (sageattention.get("env_root") or {}).get("error")
            sage_runtime = sageattention.get("sparsevideo_runtime", {})
            sage_ready = (
                sage_runtime.get("package")
                and sage_runtime.get("qattn_extension")
                and sage_runtime.get("fused_extension")
            )
            if sage_env_error:
                errors.append(sage_env_error)
            elif not sage_ready:
                errors.append(
                    "radial use_sage_attention with dense warmup requires the SparseVideo-owned "
                    "SageAttention dense backend with _qattn_sm* and _fused extensions under "
                    "src/sparsevideo/kernels/native/sageattention."
                )
            elif sageattention.get("training_free_runtime") and sageattention.get("selected_runtime") != "sparsevideo":
                errors.append(
                    "radial use_sage_attention resolves sageattention from training_free/, which is reference-only. "
                    "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/sageattention."
                )
            elif not sageattention.get("load_checked"):
                errors.append(
                    "radial use_sage_attention dense warmup benchmark preflight must import the "
                    "SparseVideo-owned SageAttention runtime; extension/source presence alone is not enough "
                    "to claim parity."
                )
            else:
                load_error = _sageattention_load_preflight_error(
                    kernels,
                    requirement="radial use_sage_attention dense warmup requires loadable SparseVideo-owned SageAttention",
                )
                if load_error is not None:
                    errors.append(load_error)

    if method == "radial":
        radial = kernels.get("radial_kernels", {})
        if not radial.get("method_source", {}).get("source_files"):
            errors.append(
                "radial requires SparseVideo-owned radial method source at "
                "src/sparsevideo/methods/radial/method.py."
            )
        if not radial.get("flashinfer_bsr_wrapper", {}).get("source_files"):
            errors.append(
                "radial requires SparseVideo-owned FlashInfer BSR wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        radial_sources_ready = all(
            radial.get(key, {}).get("source_files")
            for key in ("method_source", "flashinfer_bsr_wrapper")
        )
        if radial_sources_ready:
            radial_runtime = radial.get("owned_runtime", {})
            if not (radial.get("load_checked") or radial_runtime.get("load_checked")):
                errors.append(
                    "radial benchmark preflight must import the SparseVideo-owned radial method and "
                    "FlashInfer BSR wrapper modules; source-file presence alone is not enough to claim parity."
                )
            elif radial_runtime.get("import_error"):
                errors.append(
                    "radial owned method/BSR wrapper modules failed to import during preflight: "
                    f"{radial_runtime.get('import_error_type')}: {radial_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "radial_bsr_mask",
                    "shrink_mask_strict",
                    "radial_flashinfer_attention",
                    "build_bsr_from_mask",
                    "variable_block_sparse_attn",
                    "bsr_sparse_attn",
                    "ensure_cuda_home_for_flashinfer_jit",
                    "expand_attention_mask",
                    "radial_is_dense_layer_or_timestep",
                    "radial_window_width",
                ):
                    if not radial_runtime.get(attr):
                        missing.append(attr)
                if config.get("use_sage_attention"):
                    for attr in (
                        "radial_sage_attention",
                        "radial_sage_dense_attention",
                        "sparge_mask_convert",
                        "sparge_sage_qk_block_sizes",
                        "radial_append_tail_blocks",
                    ):
                        if not radial_runtime.get(attr):
                            missing.append(attr)
                if missing:
                    errors.append(
                        "radial owned method/BSR wrapper runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "flashomni" and config.get("implementation") == "upstream":
        flashomni = kernels["flashomni"]
        flashomni_env_error = (flashomni.get("env_root") or {}).get("error")
        sparsevideo_runtime = flashomni.get("sparsevideo_runtime", {})
        sparsevideo_ready = bool(sparsevideo_runtime.get("ready"))
        if flashomni_env_error:
            errors.append(flashomni_env_error)
        elif sparsevideo_ready:
            if not flashomni.get("load_checked"):
                errors.append(
                    "flashomni implementation=upstream benchmark preflight must import the "
                    "SparseVideo-owned FlashOmni runtime; extension/source presence alone is not "
                    "enough to claim parity."
                )
            else:
                load_error = _flashomni_load_preflight_error(
                    kernels,
                    required_attrs=(
                        "native_extension_imported",
                        "owned_runtime",
                        "batch_flashomni_fa_with_ragged_kv_wrapper",
                        "segment_packbits",
                        "torch_ops_flashomni_kernels",
                        "torch_ops_batch_sparseFA_with_kv_plan",
                        "torch_ops_batch_sparseFA_with_ragged_kv_run",
                    ),
                    requirement=(
                        "flashomni implementation=upstream requires loadable "
                        "SparseVideo-owned FlashOmni CUDA/C++ ops"
                    ),
                )
                if load_error is not None:
                    errors.append(load_error)
        elif not flashomni.get("package"):
            errors.append(
                "flashomni implementation=upstream requires the SparseVideo-owned "
                "flashomni runtime under src/sparsevideo/kernels/native/flashomni."
            )
        elif not flashomni.get("aot_config"):
            errors.append(
                "flashomni implementation=upstream requires SparseVideo-owned FlashOmni "
                "AOT kernels under src/sparsevideo/kernels/native/flashomni."
            )
        elif not flashomni.get("native_extension"):
            errors.append(
                "flashomni implementation=upstream requires SparseVideo-owned "
                "flashomni_kernels*.so from the local AOT build."
            )
        elif flashomni.get("training_free_runtime") or flashomni.get("training_free_package_detected"):
            errors.append(
                "flashomni resolves from training_free/, which is reference-only. "
                "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/flashomni."
            )
        elif flashomni.get("environment_runtime_detected") or flashomni.get("selected_runtime") != "sparsevideo":
            message = (
                "flashomni implementation=upstream requires the SparseVideo-owned runtime under "
                "src/sparsevideo/kernels/native/flashomni. Environment flashomni packages are "
                "not accepted for SparseVideo runtime parity."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
        owned_source = flashomni.get("sparsevideo_owned_source", {})
        if not owned_source.get("source_files"):
            message = (
                "flashomni has no SparseVideo-owned FlashOmni native source under "
                "src/sparsevideo/kernels/native/flashomni; this is not package-ready kernel parity."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
        if config.get("sparse_pattern", "explicit") == "explicit" and not config.get("is_full"):
            missing = [key for key in FLASHOMNI_SPARSE_INFO_KEYS if config.get(key) is None]
            if missing:
                errors.append(
                    "flashomni sparse_pattern=explicit follows upstream FlashOmni and requires "
                    "precomputed sparse_info, sparse_kv_info, sparse_info_indptr, and "
                    f"sparse_kv_info_indptr tensors. Missing: {missing}. "
                    "The inference CLI cannot synthesize an upstream video sparsity policy."
                )
            else:
                warnings.append(
                    "flashomni sparse_pattern=explicit has caller-provided sparse-info tensors, "
                    "so SparseVideo can verify the FlashOmni kernel adapter dispatch. The current "
                    "training_free/FlashOmni reference publishes the engine/API and synthetic "
                    "benchmark mask helper, but not a reusable Wan/Hunyuan video sparse-info "
                    "policy; this run is not benchmark-ready video-method parity unless those "
                    "tensors are proven to come from an upstream-compatible video policy."
                )
    if method == "flashomni" and config.get("sparse_pattern") == "local_qk_topk":
        message = (
            "flashomni sparse_pattern=local_qk_topk uses SparseVideo's block-mean top-k "
            "diagnostic policy, not upstream FlashOmni video-method parity."
        )
        if strict_kernels:
            errors.append(message)
        else:
            warnings.append(message)
    if method == "flashomni" and config.get("sparse_pattern") == "global_random":
        message = (
            "flashomni sparse_pattern=global_random matches FlashOmni's upstream synthetic kernel benchmark mask, "
            "not a video diffusion quality-parity sparsity policy. Use it for native-kernel smoke/speed checks only; "
            "use sparse_pattern=explicit with upstream-compatible sparse-info tensors for real method comparisons."
        )
        if strict_kernels:
            errors.append(message)
        else:
            warnings.append(message)
    if method == "flashomni" and config.get("sparse_pattern") == "paper_mmdit":
        warnings.append(
            "flashomni sparse_pattern=paper_mmdit uses SparseVideo-owned FlashOmni attention "
            "sparse-info code. Hunyuan runs use the public anonymous FlashOmni Hunyuan "
            "sparse-symbol policy and upstream default names threshold_q, threshold_kv, "
            "fresh_threshold, max_order, first_enhance, and saving_threshold_q_for_taylor. "
            "It can exercise the owned FlashOmni attention and GEMM-Q/GEMM-O kernels with "
            "generated sparse-info tensors, cached output/bias reuse, and the owned Hunyuan "
            "forward/Taylor-cache patch."
        )
        if model_family == "hunyuan_video":
            if int(config.get("max_order", 0) or 0) == 0 and not bool(config.get("use_sparse_gemm", False)):
                warnings.append(
                    "flashomni Hunyuan paper_mmdit is using the CLI quality-safe defaults "
                    "max_order=0,use_sparse_gemm=false."
                )
            elif bool(config.get("use_sparse_gemm", False)):
                errors.append(
                    "flashomni Hunyuan paper_mmdit sparse-GEMM path is not supported for inference. "
                    "Use use_sparse_gemm=false. The retained GEMM code caused measured quality "
                    "degradation and performance regression."
                )
    if method == "flashomni" and config.get("is_full"):
        warnings.append(
            "flashomni is_full=true disables sparsity in the upstream FlashOmni kernel path; "
            "use method=dense for the dense baseline and is_full=false for sparse benchmarking."
        )

    if method == "svoo" and config.get("sparse_backend") == "flashinfer":
        if not kernels["flashinfer"].get("package"):
            errors.append("svoo sparse_backend=flashinfer requires the flashinfer package.")
        elif not kernels["flashinfer"].get("sparse_module"):
            errors.append("svoo sparse_backend=flashinfer requires flashinfer.sparse APIs.")
        elif not kernels["flashinfer"].get("cuda_toolkit", {}).get("available"):
            errors.append(
                "svoo sparse_backend=flashinfer requires a CUDA toolkit with nvcc for FlashInfer sparse JIT. "
                "Set CUDA_HOME/CUDA_PATH or put nvcc on PATH."
            )
        else:
            error = _flashinfer_load_preflight_error(
                kernels,
                required_attrs=(
                    "sparse_variable_block_sparse_attention_wrapper",
                    "sparse_canonicalize_torch_dtype",
                    "sparse_mask_mode",
                    "sparse_pos_encoding_mode",
                    "sparse_determine_attention_backend",
                    "sparse_get_batch_prefill_module",
                ),
                requirement="svoo sparse_backend=flashinfer requires loadable FlashInfer sparse APIs",
            )
            if error is not None:
                errors.append(error)
    if method == "svoo" and model_family == "hunyuan_video" and config.get("sparse_backend") != "flashinfer":
        errors.append(
            "svoo Hunyuan upstream path requires sparse_backend=flashinfer; "
            "the Triton sparse backend is only a Wan fallback path and is not Hunyuan SVOO parity."
        )

    if method == "svoo":
        svoo = kernels.get("svoo_kernels", {})
        if not svoo.get("triton_package"):
            errors.append("svoo requires the triton package for its upstream co-clustering kernels.")
        if not config.get("use_svoo", True) and not svoo.get("triton_kmeans", {}).get("source_files"):
            errors.append(
                "svoo use_svoo=False requires SparseVideo-owned Triton k-means source at "
                "src/sparsevideo/kernels/kmeans.py."
            )
        if not svoo.get("triton_l2norm", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned Triton L2 normalization source at "
                "src/sparsevideo/kernels/l2norm.py."
            )
        if not svoo.get("triton_layernorm", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Triton layernorm source at "
                "src/sparsevideo/kernels/layernorm.py."
            )
        if not svoo.get("triton_modulate", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Triton modulation source at "
                "src/sparsevideo/kernels/modulate.py."
            )
        if not svoo.get("wan_fast_block_patch", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Wan fast-block patch source at "
                "src/sparsevideo/processors/wan_fast_block.py."
            )
        if not svoo.get("hunyuan_sparse_forward_patch", {}).get("source_files"):
            errors.append(
                "svoo Hunyuan upstream inference requires SparseVideo-owned sparse-forward patch source at "
                "src/sparsevideo/processors/hunyuan_sparse_forward.py."
            )
        if not svoo.get("co_cluster", {}).get("source_files"):
            errors.append("svoo requires SparseVideo-owned co-clustering source at src/sparsevideo/kernels/co_cluster.py.")
        if not svoo.get("dynamic_map", {}).get("source_files"):
            errors.append("svoo requires SparseVideo-owned dynamic-map source at src/sparsevideo/kernels/dynamic_map.py.")
        if not svoo.get("triton_permute", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned Triton permutation source at "
                "src/sparsevideo/kernels/permute.py."
            )
        if not svoo.get("triton_block_sparse_attn", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned Triton block sparse attention source at "
                "src/sparsevideo/kernels/block_sparse_attn.py."
            )
        if not svoo.get("flashinfer_block_sparse", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned FlashInfer block sparse wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        if config.get("measure_attention_sparsity"):
            if not svoo.get("sparsity_profiler", {}).get("source_files"):
                errors.append(
                    "svoo measure_attention_sparsity requires SparseVideo-owned profiler source at "
                    "src/sparsevideo/methods/svoo/sparsity.py."
                )
            if not svoo.get("sparsity_counts", {}).get("source_files"):
                errors.append(
                    "svoo measure_attention_sparsity requires SparseVideo-owned optional Triton count source at "
                    "src/sparsevideo/kernels/sparsity.py."
                )
        required_source_keys = [
            "triton_l2norm",
            "triton_layernorm",
            "triton_modulate",
            "co_cluster",
            "dynamic_map",
            "triton_permute",
            "triton_block_sparse_attn",
            "flashinfer_block_sparse",
        ]
        if not config.get("use_svoo", True):
            required_source_keys.append("triton_kmeans")
        if config.get("measure_attention_sparsity"):
            required_source_keys.extend(["sparsity_counts", "sparsity_profiler"])
        sources_ready = all(
            svoo.get(key, {}).get("source_files")
            for key in required_source_keys
        )
        if sources_ready:
            svoo_runtime = svoo.get("owned_triton_runtime", {})
            if not (svoo.get("load_checked") or svoo_runtime.get("load_checked")):
                errors.append(
                    "svoo benchmark preflight must import the SparseVideo-owned Triton/FlashInfer helper modules; "
                    "source-file presence alone is not enough to claim parity."
                )
            elif svoo_runtime.get("import_error"):
                errors.append(
                    "svoo owned Triton/FlashInfer helper modules failed to import during preflight: "
                    f"{svoo_runtime.get('import_error_type')}: {svoo_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "triton_l2norm_forward",
                    "triton_layernorm_forward",
                    "triton_modulate_shift_forward",
                    "triton_modulate_gate_residual_forward",
                    "co_cluster_tokens",
                    "co_cluster_assign",
                    "identify_dynamic_map",
                    "permute_tensor_by_labels_triton",
                    "apply_inverse_permutation_triton",
                    "block_sparse_attention",
                    "variable_block_sparse_attn",
                    "hunyuan_flashinfer_varlen_attn",
                ):
                    if not svoo_runtime.get(attr):
                        missing.append(attr)
                if not config.get("use_svoo", True):
                    for attr in ("triton_kmeans", "kmeans_assign_kernel", "kmeans_update_kernel"):
                        if not svoo_runtime.get(attr):
                            missing.append(attr)
                if config.get("measure_attention_sparsity"):
                    for attr in ("counts_from_sorted_probabilities_triton", "compute_exact_attention_sparsity"):
                        if not svoo_runtime.get(attr):
                            missing.append(attr)
                if missing:
                    errors.append(
                        "svoo owned Triton/FlashInfer runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "svg2":
        svg2 = kernels.get("svg2_kernels", {})
        if not svg2.get("triton_package"):
            errors.append("svg2 requires the triton package for its upstream-style k-means kernels.")
        if not svg2.get("triton_kmeans", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned Sparse-VideoGen Triton k-means source at "
                "src/sparsevideo/methods/svg2/kmeans.py."
            )
        if not svg2.get("dynamic_map", {}).get("source_files"):
            errors.append("svg2 requires SparseVideo-owned dynamic-map source at src/sparsevideo/kernels/dynamic_map.py.")
        if not svg2.get("triton_block_sparse_attn", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned Triton block sparse attention source at "
                "src/sparsevideo/kernels/block_sparse_attn.py."
            )
        if not svg2.get("triton_permute", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned Triton permutation source at "
                "src/sparsevideo/kernels/permute.py."
            )
        if not svg2.get("flashinfer_block_sparse", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned FlashInfer block sparse wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        svg2_sources_ready = all(
            svg2.get(key, {}).get("source_files")
            for key in (
                "triton_kmeans",
                "dynamic_map",
                "triton_block_sparse_attn",
                "triton_permute",
                "flashinfer_block_sparse",
            )
        )
        if svg2_sources_ready:
            svg2_runtime = svg2.get("owned_triton_runtime", {})
            if not (svg2.get("load_checked") or svg2_runtime.get("load_checked")):
                errors.append(
                    "svg2 benchmark preflight must import the SparseVideo-owned Triton/FlashInfer helper modules; "
                    "source-file presence alone is not enough to claim parity."
                )
            elif svg2_runtime.get("import_error"):
                errors.append(
                    "svg2 owned Triton/FlashInfer helper modules failed to import during preflight: "
                    f"{svg2_runtime.get('import_error_type')}: {svg2_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "triton_kmeans",
                    "euclid_assign_triton",
                    "centroid_update_triton",
                    "identify_dynamic_map",
                    "identify_dynamic_map_global",
                    "block_sparse_attention",
                    "permute_tensor_by_labels_triton",
                    "apply_inverse_permutation_triton",
                    "variable_block_sparse_attn",
                    "hunyuan_flashinfer_varlen_attn",
                ):
                    if not svg2_runtime.get(attr):
                        missing.append(attr)
                if missing:
                    errors.append(
                        "svg2 owned Triton/FlashInfer runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "svg1":
        svg1 = kernels.get("svg1_kernels", {})
        if not svg1.get("triton_package"):
            errors.append("svg1 requires the triton package for its upstream Sparse-VideoGen placement kernels.")
        if not svg1.get("method_source", {}).get("source_files"):
            errors.append(
                "svg1 requires SparseVideo-owned Sparse-VideoGen method source at "
                "src/sparsevideo/methods/svg1/method.py."
            )
        if not svg1.get("triton_placement", {}).get("source_files"):
            errors.append(
                "svg1 requires SparseVideo-owned Sparse-VideoGen Triton placement source at "
                "src/sparsevideo/methods/svg1/placement.py."
            )
        svg1_sources_ready = all(
            svg1.get(key, {}).get("source_files")
            for key in ("method_source", "triton_placement")
        )
        if svg1_sources_ready:
            svg1_runtime = svg1.get("owned_triton_runtime", {})
            if not (svg1.get("load_checked") or svg1_runtime.get("load_checked")):
                errors.append(
                    "svg1 benchmark preflight must import the SparseVideo-owned SVG method and "
                    "Triton placement modules; source-file presence alone is not enough to claim parity."
                )
            elif svg1_runtime.get("import_error"):
                errors.append(
                    "svg1 owned method/Triton placement modules failed to import during preflight: "
                    f"{svg1_runtime.get('import_error_type')}: {svg1_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "svg_attention",
                    "svg_flex_attention",
                    "svg1_dense_attention",
                    "profile_masks",
                    "svg_profile_mask_rows",
                    "build_svg_block_mask",
                    "svg_kv_blocks",
                    "svg_kv_block_partitions",
                    "svg_common_mask",
                    "place_svg_heads",
                    "restore_svg_heads",
                    "round_svg_window_width",
                    "svg_window_width",
                    "sparsity_to_width",
                    "resolve_prompt_length",
                    "sparse_head_placement",
                    "hidden_states_placement",
                    "sparse_head_placement_kernel",
                    "hidden_states_placement_kernel",
                ):
                    if not svg1_runtime.get(attr):
                        missing.append(attr)
                if model_family == "hunyuan_video" and not svg1_runtime.get("svg1_hunyuan_flash_attn_varlen"):
                    missing.append("svg1_hunyuan_flash_attn_varlen")
                if missing:
                    errors.append(
                        "svg1 owned method/Triton placement runtime is missing loadable API(s): "
                        f"{missing}."
                    )
        flex = kernels.get("flex_attention", {})
        missing = []
        if not flex.get("module"):
            missing.append("torch.nn.attention.flex_attention")
        if not flex.get("flex_attention"):
            missing.append("flex_attention")
        if not flex.get("block_mask"):
            missing.append("BlockMask")
        if not flex.get("torch_compile"):
            missing.append("torch.compile")
        if missing:
            detail = ""
            if flex.get("error"):
                detail = f" Import error: {flex.get('error_type')}: {flex.get('error')}."
            errors.append(
                "svg1 requires PyTorch FlexAttention APIs for the upstream Sparse-VideoGen sparse path. "
                f"Missing: {missing}.{detail}"
            )
        if model_family == "hunyuan_video":
            error = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_varlen_func",
                requirement=(
                    "svg1 Hunyuan dense gates require FlashAttention varlen, matching "
                    "Sparse-VideoGen's Hunyuan SVG path"
                ),
            )
            if error is not None:
                errors.append(error)

    if method == "adacluster":
        adacluster = kernels["adacluster_kernels"]
        if not adacluster.get("triton_package"):
            errors.append("adacluster requires the triton package for its upstream fast_kmeans_single/sparse kernels.")
        if not adacluster["fast_kmeans_single"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned upstream fast_kmeans_single source at "
                "src/sparsevideo/kernels/native/adacluster/fast_kmeans_single.py."
            )
        if not adacluster["triton_cluster_sparse_attn"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned upstream triton_cluster_sparse_attn source at "
                "src/sparsevideo/kernels/native/adacluster/triton_cluster_sparse_attn.py."
            )
        if not adacluster["triton_cluster_sparse_attn_topk"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned optimized triton_cluster_sparse_attn_topk source at "
                "src/sparsevideo/kernels/native/adacluster/triton_cluster_sparse_attn_topk.py."
            )
        adacluster_runtime = adacluster.get("owned_triton_runtime", {})
        if not (adacluster.get("load_checked") or adacluster_runtime.get("load_checked")):
            errors.append(
                "adacluster benchmark preflight must import the SparseVideo-owned upstream Triton kernels; "
                "source-file presence alone is not enough to claim parity."
            )
        elif adacluster_runtime.get("import_error"):
            errors.append(
                "adacluster owned Triton kernels failed to import during preflight: "
                f"{adacluster_runtime.get('import_error_type')}: {adacluster_runtime.get('import_error')}."
            )
        else:
            missing = []
            for attr in (
                "imported",
                "owned_runtime",
                "flash_kmeans_single",
                "triton_cluster_sparse_attn",
                "triton_cluster_sparse_attn_topk",
                "kmeans_jit_kernels",
                "cluster_sparse_attn_jit_kernel",
                "cluster_sparse_attn_topk_jit_kernel",
            ):
                if not adacluster_runtime.get(attr):
                    missing.append(attr)
            if missing:
                errors.append(
                    "adacluster owned Triton runtime is missing loadable API(s): "
                    f"{missing}."
                )
        if model_family == "hunyuan_video":
            error = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_func",
                requirement=(
                    "adacluster Hunyuan dense gates require FlashAttention, matching "
                    "the upstream Hunyuan AdaCluster path"
                ),
            )
            if error is not None:
                errors.append(error)

    if method == "draft":
        if model_family in (None, "wan", "hunyuan_video"):
            message = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_varlen_func",
                requirement="draft dense gates require FlashAttention varlen for upstream parity",
            )
            if message is not None:
                if strict_kernels:
                    errors.append(message)
                else:
                    warnings.append(message)
        draft = kernels["draft_kernels"]
        mit_backend = draft.get("mit_block_sparse_attn", {})
        mit_ready = mit_backend.get("source_files") and mit_backend.get("cuda_extension")
        fallback_ready = draft.get("triton_package") and draft["triton_block_sparse_attn"].get("source_files")
        if not mit_ready:
            message = (
                "draft upstream parity requires SparseVideo-owned MIT Han Lab "
                "Block-Sparse-Attention source and block_sparse_attn_cuda extension under "
                "src/sparsevideo/kernels/native/draft_block_sparse. The generic "
                "src/sparsevideo/kernels/block_sparse_attn.py Triton kernel is not the "
                "upstream Draft backend and is debug fallback only."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
                if not draft.get("triton_package"):
                    errors.append("draft debug fallback requires the triton package.")
                if not draft["triton_block_sparse_attn"].get("source_files"):
                    errors.append(
                        "draft debug fallback requires SparseVideo-owned generic Triton "
                        "block sparse source at src/sparsevideo/kernels/block_sparse_attn.py."
                    )
        elif draft.get("mit_load_checked") or mit_backend.get("load_checked"):
            mit_load_error = None
            if mit_backend.get("import_error"):
                mit_load_error = (
                    "draft MIT Block-Sparse-Attention backend failed to import during preflight: "
                    f"{mit_backend.get('import_error_type')}: {mit_backend.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "cuda_extension_imported",
                    "owned_runtime",
                    "block_sparse_attn_func",
                    "cuda_fwd_block",
                    "cuda_bwd_block",
                ):
                    if not mit_backend.get(attr):
                        missing.append(attr)
                if missing:
                    mit_load_error = (
                        "draft MIT Block-Sparse-Attention backend is missing loadable API(s): "
                        f"{missing}."
                    )
            if mit_load_error is not None:
                if strict_kernels:
                    errors.append(mit_load_error)
                else:
                    warnings.append(mit_load_error)
        elif not fallback_ready:
            warnings.append(
                "draft MIT backend is present; generic Triton fallback source is missing, "
                "so --allow-debug-fallbacks cannot exercise the local fallback path."
            )

    if method in ("radial", "svg2"):
        flashinfer = kernels["flashinfer"]
        fallback_name = "allow_flex_fallback" if method == "radial" else "allow_triton_fallback"
        fallback_path = "FlexAttention" if method == "radial" else "Triton block-sparse"
        if not flashinfer.get("package"):
            message = (
                f"{method} FlashInfer is not importable; the {fallback_path} path is debug-only "
                f"and requires {fallback_name}."
            )
        elif not flashinfer.get("sparse_module"):
            message = (
                f"{method} requires flashinfer.sparse for the upstream sparse kernel path; "
                f"the {fallback_path} path is debug-only and requires {fallback_name}."
            )
        elif "cuda_toolkit" in flashinfer and not flashinfer.get("cuda_toolkit", {}).get("available"):
            message = (
                f"{method} FlashInfer sparse kernels require a CUDA toolkit with nvcc for JIT. "
                "Set CUDA_HOME/CUDA_PATH or put nvcc on PATH before benchmarking."
            )
        else:
            message = None
        if message is not None:
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
        else:
            required_attrs = (
                (
                    "top_level_block_sparse_attention_wrapper",
                    "top_level_single_prefill_with_kv_cache",
                    "top_level_merge_state",
                )
                if method == "radial"
                else (
                    "sparse_variable_block_sparse_attention_wrapper",
                    "sparse_canonicalize_torch_dtype",
                    "sparse_mask_mode",
                    "sparse_pos_encoding_mode",
                    "sparse_determine_attention_backend",
                    "sparse_get_batch_prefill_module",
                )
            )
            if method == "radial":
                required_attrs = required_attrs + (
                    "sparse_variable_block_sparse_attention_wrapper",
                    "sparse_canonicalize_torch_dtype",
                    "sparse_mask_mode",
                    "sparse_pos_encoding_mode",
                    "sparse_determine_attention_backend",
                    "sparse_get_batch_prefill_module",
                )
            load_error = _flashinfer_load_preflight_error(
                kernels,
                required_attrs=required_attrs,
                requirement=f"{method} requires loadable FlashInfer sparse APIs for the upstream sparse path",
            )
            if load_error is not None:
                if strict_kernels:
                    errors.append(load_error)
                else:
                    warnings.append(load_error)

    if method == "sta":
        sta = kernels["sta_kernels"]
        fastvideo_triton = sta.get("sparsevideo_fastvideo_triton", {})
        sta_mode = config.get("STA_mode", "STA_inference")
        if sta_mode not in ("STA_inference", "STA_searching"):
            errors.append(
                "sta supports STA_inference in pipelines and STA_searching for mask calibration; "
                "use python -m sparsevideo.methods.sta.search tune for STA_tuning."
            )
        if sta_mode == "STA_searching":
            warnings.append(
                "sta STA_searching returns dense attention outputs while recording sparse-window losses; "
                "it is a calibration run, not a speed benchmark."
            )
        tile_size = _normalize_int_triple(config.get("tile_size", [6, 8, 8]))
        if tile_size != (6, 8, 8):
            errors.append(
                "sta tile_size differs from FastVideo's fixed upstream tile_size=(6,8,8); "
                "SparseVideo rejects the non-upstream generalized STA fallback for parity runs."
            )
        if not fastvideo_triton.get("source_files"):
            errors.append(
                "sta requires the SparseVideo-owned copy of FastVideo's Triton STA source at "
                "src/sparsevideo/kernels/native/sta_h100/python/fastvideo_kernel/triton_kernels/st_attn_triton.py."
            )
        elif sta.get("triton_load_checked"):
            if sta.get("triton_import_error"):
                errors.append(
                    "sta SparseVideo-owned FastVideo Triton fallback failed to import during preflight: "
                    f"{sta.get('triton_import_error_type')}: {sta.get('triton_import_error')}."
                )
            elif not sta.get("triton_sliding_tile_attention_triton"):
                errors.append(
                    "sta SparseVideo-owned FastVideo Triton fallback is missing "
                    "sliding_tile_attention_triton."
                )
        if not sta["sparsevideo_h100"].get("source", {}).get("source_files"):
            message = (
                "sta FastVideo H100/TK C++ source is missing under "
                "src/sparsevideo/kernels/native/sta_h100; this is not package-ready kernel parity."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
        seq_shape = normalize_seq_shape_for_warning(config.get("seq_shape"))
        if seq_shape is None:
            message = (
                "sta seq_shape is not set. SparseVideo will infer the video layout from token length; "
                "FastVideo STA native shapes are "
                f"{sorted(STA_NATIVE_SEQ_SHAPES)}."
            )
            warnings.append(message)
        elif seq_shape not in STA_NATIVE_SEQ_SHAPES:
            parsed_seq_shape = _parse_normalized_seq_shape(seq_shape)
            if parsed_seq_shape is None:
                errors.append(f"sta seq_shape={seq_shape} is invalid; expected TxHxW.")
            else:
                warnings.append(
                    f"sta seq_shape={seq_shape} uses SparseVideo's generalized FastVideo Triton STA path "
                    "for this backbone's inferred tile-padded video layout; this is not a FastVideo parity profile."
                )
        elif seq_shape in STA_NATIVE_SEQ_SHAPES:
            has_hopper = _has_hopper_device(torch_status)
            h100_extension = bool(sta["sparsevideo_h100"].get("native_extension"))
            h100_load_error = None
            h100_usable = h100_extension
            if h100_extension and sta.get("h100_native_load_checked"):
                h100_usable = bool(
                    sta.get("h100_native_extension_imported")
                    and sta.get("h100_sta_fwd")
                )
                if sta.get("h100_import_error"):
                    h100_load_error = (
                        "sta H100/TK C++ extension failed to load during preflight: "
                        f"{sta.get('h100_import_error_type')}: {sta.get('h100_import_error')}."
                    )
                elif not sta.get("h100_sta_fwd"):
                    h100_load_error = "sta H100/TK C++ extension is missing sta_fwd."
            if h100_load_error is not None:
                if strict_kernels:
                    errors.append(h100_load_error)
                else:
                    warnings.append(h100_load_error)
            if has_hopper and not h100_usable:
                message = (
                    "sta H100/TK C++ parity kernel is not available as a SparseVideo-owned sta_h100 extension; "
                    "runtime will use the SparseVideo-owned copy of FastVideo's Triton STA fallback instead."
                )
                if strict_kernels:
                    errors.append(message)
                else:
                    warnings.append(message)
            elif h100_usable and not has_hopper:
                message = (
                    "sta_h100 extension is built but no Hopper GPU is visible; "
                    "this run will use the SparseVideo-owned copy of FastVideo's Triton STA fallback, "
                    "matching FastVideo's non-Hopper fallback path."
                )
                warnings.append(message)

    return {"errors": errors, "warnings": warnings}


def _has_hopper_device(torch_status: Dict[str, Any]) -> bool:
    for device in torch_status.get("cuda_devices") or []:
        capability = device.get("capability") or []
        if capability and int(capability[0]) >= 9:
            return True
    return False


def draft_upstream_layout_error(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
    config: Dict[str, Any],
) -> Optional[str]:
    if not config.get("block_sparse_attention", True):
        return None

    latent_t, latent_h, latent_w = _draft_estimated_latent_shape(spec, height, width, num_frames)
    pool_h = int(config.get("pool_h", 8))
    pool_w = int(config.get("pool_w", 16))
    video_len = latent_t * latent_h * latent_w
    if pool_h * pool_w != 128:
        return (
            "draft MIT Block-Sparse-Attention backend requires pool_h * pool_w == 128 "
            f"to form upstream 128-token blocks; got pool_h={pool_h}, pool_w={pool_w}."
        )

    for key, actual in (
        ("latent_h", latent_h),
        ("latent_w", latent_w),
        ("visual_len", video_len),
    ):
        configured = config.get(key)
        if configured is not None and int(configured) != int(actual):
            return (
                f"draft upstream {key} config expects {int(configured)}, "
                f"but the requested layout has {key}={int(actual)}."
            )

    configured_text_len = config.get("text_len")
    expected_text_len = None
    if spec.family == "hunyuan_video":
        expected_text_len = 256
    elif spec.family in ("wan", "ltx_video", "allegro"):
        expected_text_len = 0
    if (
        configured_text_len is not None
        and expected_text_len is not None
        and int(configured_text_len) != expected_text_len
    ):
        return (
            f"draft upstream text_len config expects {int(configured_text_len)}, "
            f"but {spec.family} upstream path expects text_len={expected_text_len}."
        )

    if spec.family not in ("wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate"):
        return f"draft is not implemented for {spec.family}."

    return None


def radial_flashinfer_layout_warning(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
    config: Dict[str, Any],
) -> Optional[str]:
    radial_families = ("wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate")
    if spec.family not in radial_families:
        return f"radial is not implemented for {spec.family}."
    if height % 16 != 0 or width % 16 != 0:
        return (
            "radial upstream FlashInfer path expects height and width divisible by 16 "
            f"for video patch tokens; got {height}x{width}."
        )

    return None


def _radial_estimated_latent_shape(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[int, int, int]:
    if spec.family == "cogvideox":
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
        if spec.key == "cogvideox-i2v":
            latent_t = ((num_frames - 1) // 4 + 1) * 2
        return latent_t, latent_h // 2, latent_w // 2
    if spec.family == "ltx_video":
        return (num_frames - 1) // 8 + 1, height // 32, width // 32
    if spec.family == "allegro":
        latent_t = ((num_frames + 3) // 4) if num_frames % 2 == 0 else ((num_frames - 1 + 3) // 4 + 1)
        return latent_t, height // 16, width // 16
    if spec.family == "mochi":
        return (num_frames - 1) // 6 + 1, height // 16, width // 16
    if spec.family == "easyanimate":
        return (num_frames - 1) // 4 + 1, height // 16, width // 16
    return (num_frames - 1) // 4 + 1, height // 16, width // 16


def _draft_estimated_latent_shape(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[int, int, int]:
    if spec.family == "cogvideox":
        return (num_frames - 1) // 4 + 1, height // 16, width // 16
    return _radial_estimated_latent_shape(spec, height, width, num_frames)


def apply_draft_runtime_layout_defaults(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
    config: Dict[str, Any],
    user_config: Dict[str, Any],
) -> None:
    latent_t, latent_h, latent_w = _draft_estimated_latent_shape(spec, height, width, num_frames)
    defaults = {
        "latent_h": latent_h,
        "latent_w": latent_w,
        "visual_len": latent_t * latent_h * latent_w,
    }
    for key, value in defaults.items():
        if key not in user_config and config.get(key) is None:
            config[key] = value


def sta_layout_preflight_messages(
    spec: ModelSpec,
    height: int,
    width: int,
    num_frames: int,
    config: Dict[str, Any],
    *,
    strict_kernels: bool = True,
) -> Dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    sta_families = ("wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate")
    if spec.family not in sta_families:
        return {"errors": [f"sta is not implemented for {spec.family}."], "warnings": warnings}
    if height % 16 != 0 or width % 16 != 0:
        return {
            "errors": [
                "sta upstream FastVideo path expects height and width divisible by 16 "
                f"for video patch tokens; got {height}x{width}."
            ],
            "warnings": warnings,
        }

    sta_mode = config.get("STA_mode", "STA_inference")
    mask_strategy_path = config.get("mask_strategy_file_path")
    if mask_strategy_path is None and sta_mode != "STA_searching":
        message = (
            "sta STA_inference has no tuned mask_strategy_file_path for this model. "
            "SparseVideo will use the configured window_size for every layer/head as local STA support; "
            "this is not a tuned FastVideo sparse strategy."
        )
        warnings.append(message)
    elif mask_strategy_path is not None:
        try:
            strategy_shape = _sta_mask_strategy_shape(mask_strategy_path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"sta could not read mask_strategy_file_path={mask_strategy_path!r}: {exc}")
        else:
            expected_shape = STA_STRATEGY_SHAPES.get(spec.key)
            if expected_shape is None and spec.key in STA_UNSUPPORTED_STRATEGY_MODELS:
                message = (
                    f"sta has no upstream sparse inference mask strategy for {spec.key}. "
                    f"{STA_UNSUPPORTED_STRATEGY_MODELS[spec.key]} "
                    f"Provided strategy has shape steps/layers/heads={strategy_shape}."
                )
                if strict_kernels:
                    errors.append(message)
                else:
                    warnings.append(message)
            elif expected_shape is not None and strategy_shape != expected_shape:
                message = (
                    f"sta mask_strategy_file_path shape steps/layers/heads={strategy_shape} does not match "
                    f"the expected {spec.key} strategy shape {expected_shape}."
                )
                if strict_kernels:
                    errors.append(message)
                else:
                    warnings.append(message)

    latent_t, latent_h, latent_w = _radial_estimated_latent_shape(spec, height, width, num_frames)
    latent_shape = (latent_t, latent_h, latent_w)
    latent_seq_shape = f"{latent_t}x{latent_h}x{latent_w}"

    seq_shape = normalize_seq_shape_for_warning(config.get("seq_shape"))
    if seq_shape is not None:
        parsed_seq_shape = _parse_normalized_seq_shape(seq_shape)
        if parsed_seq_shape != latent_shape:
            errors.append(
                f"sta seq_shape={seq_shape} does not match the current latent layout "
                f"{latent_seq_shape} from {num_frames} frames at {width}x{height}; "
                "runtime would fail before reaching the FastVideo STA path."
            )

    tile_size = _normalize_int_triple(config.get("tile_size", [6, 8, 8])) or (6, 8, 8)
    padded_shape = tuple(
        ((dim + tile - 1) // tile) * tile
        for dim, tile in zip(latent_shape, tile_size)
    )
    padded_seq_shape = "x".join(str(part) for part in padded_shape)
    if padded_seq_shape not in STA_NATIVE_SEQ_SHAPES:
        if spec.family not in ("wan", "hunyuan_video"):
            warnings.append(
                f"sta will use generalized FastVideo Triton STA for tile-padded canvas {padded_seq_shape} "
                f"from latent layout {latent_seq_shape}."
            )
            return {"errors": errors, "warnings": warnings}
        message = (
            "sta upstream FastVideo native path only covers latent layouts "
            f"{sorted(STA_NATIVE_SEQ_SHAPES)}. Current latent layout is {latent_seq_shape} "
            f"and tile-padded canvas is {padded_seq_shape}, so SparseVideo will use the "
            "generalized FastVideo Triton STA path with padded-border dense repair for this target shape. "
            "Do not treat this as native FastVideo profile parity unless the requested target has matching "
            "quality and speed evidence."
        )
        warnings.append(message)

    return {"errors": errors, "warnings": warnings}


def model_quality_warnings(spec: ModelSpec, height: int, width: int) -> list[str]:
    warnings: list[str] = []
    if spec.key == "wan21-t2v-1.3b" and int(height) >= 720:
        warnings.append(
            "Wan2.1 T2V 1.3B is a 480P model in the Wan README; 720P is explicitly less "
            "stable and should be treated as target-shape stress evidence rather than a standalone "
            "model-quality baseline."
        )
    return warnings


def model_shape_preflight_errors(spec: ModelSpec, height: int, width: int) -> list[str]:
    if spec.family == "allegro" and (int(height) % 16 != 0 or int(width) % 16 != 0):
        return [
            "allegro requires height and width divisible by 16 for the VAE/transformer "
            f"patch grid; got {int(height)}x{int(width)}. Use a nearby shape such as "
            "--height 352 --width 640 or --height 320 --width 576."
        ]
    return []


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


def _has_component_weight(component_dir: Path) -> bool:
    if not component_dir.exists():
        return False
    for pattern in (
        "*.safetensors",
        "*.bin",
        "*.ckpt",
        "*.pt",
        "*.pth",
        "*.index.json",
        "*.index.bf16.json",
    ):
        if any(component_dir.glob(pattern)):
            return True
    return False


def _ltx_single_file_checkpoint(model_id: str) -> Optional[Path]:
    path = Path(model_id).expanduser()
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if not path.is_dir():
        return None
    if (path / "transformer" / "config.json").exists():
        return None

    preferred = (
        "ltx-video-2b-v0.9.5.safetensors",
        "ltxv-2b-0.9.6-distilled-04-25.safetensors",
        "ltxv-2b-0.9.6-dev-04-25.safetensors",
        "ltx-video-2b-v0.9.1.safetensors",
        "ltx-video-2b-v0.9.safetensors",
    )
    for name in preferred:
        checkpoint = path / name
        if checkpoint.exists():
            return checkpoint
    return None


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compatible_t5_component_root(candidate: Path, reference_config: Optional[Dict[str, Any]]) -> bool:
    text_encoder_dir = candidate / "text_encoder"
    tokenizer_dir = candidate / "tokenizer"
    if not _has_component_weight(text_encoder_dir) or not (tokenizer_dir / "spiece.model").exists():
        return False
    if reference_config is None:
        return True
    candidate_config = _read_json_file(text_encoder_dir / "config.json")
    if candidate_config is None:
        return False
    for key in ("model_type", "d_model", "num_layers", "vocab_size"):
        if candidate_config.get(key) != reference_config.get(key):
            return False
    return True


def _resolve_ltx_text_component_root(model_id: str) -> Optional[Path]:
    path = Path(model_id).expanduser()
    model_dir = path if path.is_dir() else path.parent
    reference_config = _read_json_file(model_dir / "text_encoder" / "config.json")

    candidates = [model_dir]
    if model_dir.parent.exists():
        candidates.extend(
            model_dir.parent / name
            for name in (
                "CogVideoX-5b",
                "CogVideoX-5b-I2V",
                "allegro",
                "mochi-1",
            )
        )
    for candidate in candidates:
        if _compatible_t5_component_root(candidate, reference_config):
            return candidate
    return None


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


def infer_hunyuan_prompt_length(pipe, prompt: str, max_sequence_length: int = 256) -> int:
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE

    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        raise RuntimeError("Cannot infer Hunyuan prompt_length because the pipeline has no tokenizer")

    prompts = [prompt] if isinstance(prompt, str) else prompt
    template = DEFAULT_PROMPT_TEMPLATE
    templated_prompts = [template["template"].format(item) for item in prompts]
    crop_start = template.get("crop_start", None)
    if crop_start is None:
        template_input = pipe.tokenizer(
            template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = template_input["input_ids"].shape[-1] - 2

    text_inputs = pipe.tokenizer(
        templated_prompts,
        max_length=max_sequence_length + crop_start,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    prompt_attention_mask = text_inputs.attention_mask
    if crop_start > 0:
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]
    return int(prompt_attention_mask.sum().item())


def parse_dtype(torch_module, dtype: str):
    if dtype == "bf16":
        return torch_module.bfloat16
    if dtype == "fp16":
        return torch_module.float16
    return torch_module.float32


def seed_everything(torch_module, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def resolve_wan_flow_shift(height: int, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return 5.0 if int(height) >= 720 else 3.0


def resolve_scheduler_flow_shift(spec: ModelSpec, height: int, override: Optional[float]) -> Optional[float]:
    if spec.family == "wan":
        return resolve_wan_flow_shift(height, override)
    if spec.family == "hunyuan_video" and override is not None:
        return float(override)
    return None


def configure_method_runtime_env(method: str) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python_prefix = Path(sys.prefix)
    nvcc = python_prefix / "bin" / "nvcc"
    if "CUDA_HOME" not in os.environ and nvcc.exists():
        os.environ["CUDA_HOME"] = str(python_prefix)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_path = Path(cuda_home)
        os.environ.setdefault("CUDA_PATH", str(cuda_path))
        nvcc_path = cuda_path / "bin" / "nvcc"
        if nvcc_path.exists():
            os.environ.setdefault("CUDACXX", str(nvcc_path))
        bin_path = str(cuda_path / "bin")
        if bin_path not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        lib_paths = [
            str(cuda_path / "lib"),
            str(cuda_path / "targets" / "x86_64-linux" / "lib"),
        ]
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        for lib_path in reversed(lib_paths):
            if lib_path not in ld_library_path.split(os.pathsep):
                ld_library_path = lib_path + os.pathsep + ld_library_path
        os.environ["LD_LIBRARY_PATH"] = ld_library_path.rstrip(os.pathsep)
        os.environ.setdefault(
            "FLASHINFER_EXTRA_LDFLAGS",
            "-L{0}/lib -L{0}/targets/x86_64-linux/lib "
            "-L{0}/lib/stubs -L{0}/targets/x86_64-linux/lib/stubs".format(cuda_home),
        )

    if method != "svoo":
        return

    cache_root = Path(os.environ.get("SVOO_CACHE_ROOT", REPO_ROOT / ".triton_cache"))
    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", cache_root))
    torchinductor_cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", cache_root / "torchinductor"))
    flashinfer_workspace = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", cache_root / "flashinfer"))

    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(flashinfer_workspace))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SVOO_ENABLE_MEM_SAVE", "1")

    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    flashinfer_workspace.mkdir(parents=True, exist_ok=True)


def load_pipeline(
    spec: ModelSpec,
    model_id: str,
    torch_dtype,
    vae_dtype,
    local_files_only: bool,
    height: int,
    flow_shift: Optional[float],
):
    def pipeline_load_kwargs(**kwargs):
        import tempfile

        temp_root = REPO_ROOT / ".tmp_offload"
        temp_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(temp_root))
        tempfile.tempdir = str(temp_root)
        offload_folder = temp_root / "diffusers_state_dict"
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("low_cpu_mem_usage", True)
        if os.environ.get("SPARSEVIDEO_OFFLOAD_STATE_DICT") == "1":
            kwargs.setdefault("offload_state_dict", True)
            kwargs.setdefault("offload_folder", str(offload_folder))
        device_map = os.environ.get("SPARSEVIDEO_DEVICE_MAP")
        if device_map:
            kwargs.setdefault("device_map", device_map)
        if local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    if spec.pipeline_class == "UnavailablePipeline":
        raise RuntimeError(spec.unsupported_reason or f"{spec.key} has no configured pipeline class")
    if spec.pipeline_class == "WanPipeline":
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        kwargs = {"local_files_only": True} if local_files_only else {}
        if vae_dtype is None:
            vae_dtype = torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            **kwargs,
        )
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            **pipeline_load_kwargs(torch_dtype=torch_dtype),
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe
    elif spec.pipeline_class == "WanImageToVideoPipeline":
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        kwargs = {"local_files_only": True} if local_files_only else {}
        if vae_dtype is None:
            vae_dtype = torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            **kwargs,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            **pipeline_load_kwargs(torch_dtype=torch_dtype),
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe
    elif spec.pipeline_class in ("WanAnimatePipeline", "WanVACEPipeline"):
        import torch
        from diffusers import AutoencoderKLWan, WanAnimatePipeline, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        kwargs = {"local_files_only": True} if local_files_only else {}
        if vae_dtype is None:
            vae_dtype = torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            **kwargs,
        )
        cls = WanAnimatePipeline if spec.pipeline_class == "WanAnimatePipeline" else WanVACEPipeline
        pipe = cls.from_pretrained(
            model_id,
            vae=vae,
            **pipeline_load_kwargs(torch_dtype=torch_dtype),
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe
    elif spec.pipeline_class in ("SkyReelsV2Pipeline", "SkyReelsV2ImageToVideoPipeline"):
        import torch
        from diffusers import (
            AutoencoderKLWan,
            SkyReelsV2ImageToVideoPipeline,
            SkyReelsV2Pipeline,
        )
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        kwargs = {"local_files_only": True} if local_files_only else {}
        if vae_dtype is None:
            vae_dtype = torch.float32
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            **kwargs,
        )
        cls = (
            SkyReelsV2ImageToVideoPipeline
            if spec.pipeline_class == "SkyReelsV2ImageToVideoPipeline"
            else SkyReelsV2Pipeline
        )
        pipe = cls.from_pretrained(
            model_id,
            vae=vae,
            **pipeline_load_kwargs(torch_dtype=torch_dtype),
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe
    elif spec.pipeline_class == "HunyuanVideoPipeline":
        from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline

        cls = HunyuanVideoPipeline
    elif spec.pipeline_class == "HunyuanVideoImageToVideoPipeline":
        from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoImageToVideoPipeline

        cls = HunyuanVideoImageToVideoPipeline
    elif spec.pipeline_class == "CogVideoXPipeline":
        from diffusers import CogVideoXPipeline

        cls = CogVideoXPipeline
    elif spec.pipeline_class == "CogVideoXImageToVideoPipeline":
        from diffusers import CogVideoXImageToVideoPipeline

        cls = CogVideoXImageToVideoPipeline
    elif spec.pipeline_class == "LTXPipeline":
        from diffusers import LTXPipeline

        cls = LTXPipeline
    elif spec.pipeline_class == "LTXImageToVideoPipeline":
        from diffusers import LTXImageToVideoPipeline

        cls = LTXImageToVideoPipeline
    elif spec.pipeline_class == "AllegroPipeline":
        from diffusers import AllegroPipeline

        cls = AllegroPipeline
    elif spec.pipeline_class == "MochiPipeline":
        from diffusers import MochiPipeline

        cls = MochiPipeline
    elif spec.pipeline_class == "EasyAnimatePipeline":
        from diffusers import EasyAnimatePipeline

        cls = EasyAnimatePipeline
    elif spec.pipeline_class == "SanaVideoPipeline":
        from diffusers import SanaVideoPipeline

        cls = SanaVideoPipeline
    elif spec.pipeline_class == "Kandinsky5T2VPipeline":
        from diffusers import Kandinsky5T2VPipeline

        cls = Kandinsky5T2VPipeline
    else:
        raise ValueError(f"Unknown pipeline class: {spec.pipeline_class}")

    kwargs = {"torch_dtype": torch_dtype}
    if local_files_only:
        kwargs["local_files_only"] = True
    if spec.pipeline_class == "HunyuanVideoPipeline" and model_id == "tencent/HunyuanVideo":
        kwargs["revision"] = "refs/pr/18"
    if spec.pipeline_class in ("HunyuanVideoPipeline", "HunyuanVideoImageToVideoPipeline") and flow_shift is not None:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=float(flow_shift))
    if spec.pipeline_class in ("LTXPipeline", "LTXImageToVideoPipeline"):
        checkpoint = _ltx_single_file_checkpoint(model_id)
        if checkpoint is not None:
            component_root = _resolve_ltx_text_component_root(model_id)
            if component_root is None:
                raise RuntimeError(
                    "LTX local single-file checkpoint requires a compatible local T5 text_encoder "
                    "and tokenizer because the checkpoint does not contain those weights. "
                    "Download the Diffusers component layout for LTX or place a compatible "
                    "T5EncoderModel/tokenizer next to another local video model."
                )
            from transformers import T5EncoderModel, T5Tokenizer

            text_encoder = T5EncoderModel.from_pretrained(
                component_root / "text_encoder",
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
            tokenizer = T5Tokenizer.from_pretrained(
                component_root / "tokenizer",
                local_files_only=local_files_only,
            )
            return cls.from_single_file(
                str(checkpoint),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                **kwargs,
            )
    return cls.from_pretrained(model_id, **pipeline_load_kwargs(**kwargs))


def prepare_pipeline(
    pipe,
    device: str,
    cpu_offload: bool,
    vae_tiling: bool,
    vae_slicing: bool,
    cpu_offload_mode: str = "model",
    vae_decoder_chunk_size: Optional[int] = None,
):
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if vae_decoder_chunk_size is not None:
            pipe.vae.decoder_chunk_size = int(vae_decoder_chunk_size)

    if cpu_offload:
        if cpu_offload_mode == "sequential":
            if not hasattr(pipe, "enable_sequential_cpu_offload"):
                raise RuntimeError("This pipeline does not expose enable_sequential_cpu_offload()")
            pipe.enable_sequential_cpu_offload()
        elif cpu_offload_mode == "model":
            if not hasattr(pipe, "enable_model_cpu_offload"):
                raise RuntimeError("This pipeline does not expose enable_model_cpu_offload()")
            try:
                pipe.enable_model_cpu_offload(device=device)
            except TypeError:
                pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError(f"Unsupported cpu_offload_mode={cpu_offload_mode!r}")
    elif getattr(pipe, "hf_device_map", None) is not None:
        return
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
    output_type = "latent" if getattr(args, "skip_decode", False) else spec.output_type
    if spec.pipeline_class == "WanAnimatePipeline":
        if (
            getattr(args, "image", None) is None
            or getattr(args, "pose_video", None) is None
            or getattr(args, "face_video", None) is None
        ):
            raise RuntimeError("WanAnimate real inference requires image, pose_video, and face_video inputs")
    if spec.pipeline_class == "WanVACEPipeline":
        if getattr(args, "reference_video", None) is None or getattr(args, "mask_video", None) is None:
            raise RuntimeError("WanVACE real inference requires video and mask inputs")
    if spec.pipeline_class == "SanaVideoPipeline":
        frame_key = "frames"
    elif spec.pipeline_class == "WanAnimatePipeline":
        frame_key = "segment_frame_length"
    else:
        frame_key = "num_frames"
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": output_type,
    }
    kwargs[frame_key] = num_frames
    if spec.pipeline_class in (
        "WanImageToVideoPipeline",
        "SkyReelsV2ImageToVideoPipeline",
        "HunyuanVideoImageToVideoPipeline",
        "CogVideoXImageToVideoPipeline",
        "LTXImageToVideoPipeline",
    ):
        kwargs["image"] = _load_i2v_image(args.image)
    if spec.pipeline_class == "WanAnimatePipeline":
        kwargs["image"] = _load_i2v_image(args.image)
        kwargs["pose_video"] = _load_video_frames(args.pose_video)
        kwargs["face_video"] = _load_video_frames(args.face_video)
    if spec.pipeline_class == "WanVACEPipeline":
        if args.reference_video is not None:
            kwargs["video"] = _load_video_frames(args.reference_video)
        if args.mask_video is not None:
            kwargs["mask"] = _load_video_frames(args.mask_video)
    if spec.key in ("wan22-t2v-a14b", "wan22-i2v-a14b"):
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    if spec.family == "hunyuan_video":
        kwargs["true_cfg_scale"] = args.true_cfg_scale
    if spec.family == "cogvideox":
        kwargs["use_dynamic_cfg"] = False
    if spec.family == "ltx_video":
        kwargs["frame_rate"] = fps
    return kwargs


def apply_hunyuan_i2v_prompt_template_compat(pipe, call_kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Adapt Hunyuan I2V's prompt crop anchor for local tokenizer variants."""
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video_image2video import DEFAULT_PROMPT_TEMPLATE

    template = dict(DEFAULT_PROMPT_TEMPLATE)
    tokenizer = getattr(pipe, "tokenizer", None)
    prompt = call_kwargs.get("prompt")
    status: Dict[str, Any] = {
        "default_double_return_token_id": template.get("double_return_token_id"),
        "selected_double_return_token_id": template.get("double_return_token_id"),
        "override": False,
    }
    if tokenizer is None or prompt is None:
        call_kwargs["prompt_template"] = template
        return status

    prompt_item = prompt[0] if isinstance(prompt, list) and prompt else prompt
    if not isinstance(prompt_item, str):
        call_kwargs["prompt_template"] = template
        return status

    rendered_prompt = template["template"].format(prompt_item)
    max_length = int(template.get("crop_start", 0) or 0) + 256
    text_inputs = tokenizer(
        rendered_prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=False,
    )
    token_ids = text_inputs.input_ids[0].tolist()
    default_token_id = int(template.get("double_return_token_id", 271))
    default_count = token_ids.count(default_token_id)
    status["default_token_count"] = default_count

    if default_count == 0:
        assistant_header_end_token_id = 128007
        assistant_header_end_count = token_ids.count(assistant_header_end_token_id)
        status["assistant_header_end_token_count"] = assistant_header_end_count
        if assistant_header_end_count > 0:
            template["double_return_token_id"] = assistant_header_end_token_id
            status["selected_double_return_token_id"] = assistant_header_end_token_id
            status["override"] = True

    call_kwargs["prompt_template"] = template
    return status


def call_pipeline_with_model_compat(pipe, call_kwargs: Dict[str, Any], torch_module, spec: ModelSpec, device: str):
    if spec.pipeline_class != "HunyuanVideoImageToVideoPipeline" or not device.startswith("cuda"):
        return pipe(**call_kwargs)

    previous_default_device = None
    if hasattr(torch_module, "get_default_device"):
        previous_default_device = torch_module.get_default_device()
    torch_module.set_default_device(device)
    try:
        return pipe(**call_kwargs)
    finally:
        torch_module.set_default_device(previous_default_device or "cpu")


def _load_i2v_image(image_path: Optional[str]):
    if image_path is None:
        raise ValueError(
            "I2V models require --image <path>. Provide a path to the conditioning image."
        )
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def _load_video_frames(video_path: str):
    from diffusers.utils import load_video
    return load_video(video_path)


def make_output_file(args: argparse.Namespace, model: str, method: str, num_frames: int) -> Path:
    if args.output_file is not None:
        return args.output_file
    filename = f"seed{args.seed}_{args.height}x{args.width}_{num_frames}f.mp4"
    return args.output_dir / model / method / filename


def append_metrics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def print_metrics_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def terminal_one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def append_terminal_message(messages: List[str], value: Any) -> None:
    text = terminal_one_line(value)
    if text and text not in messages:
        messages.append(text)


def collect_runtime_messages(payload: Dict[str, Any], key: str) -> List[str]:
    runtime = payload.get("runtime") or {}
    messages: List[str] = []
    for section_name in ("preflight", "generation_checks"):
        section = runtime.get(section_name) or {}
        for message in section.get(key) or []:
            append_terminal_message(messages, message)
    if key == "errors" and not messages:
        append_terminal_message(messages, payload.get("error"))
    return messages


def print_run_summary(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    timings = payload.get("timings") or {}
    lines = [
        f"status={payload.get('status')}",
        f"model={payload.get('model')}",
        f"method={payload.get('method')}",
    ]
    if payload.get("failed_stage"):
        lines.append(f"failed_stage={payload['failed_stage']}")
    if payload.get("error_type"):
        lines.append(f"error_type={payload['error_type']}")
    output_file = payload.get("output_file")
    lines.append(f"output_file={output_file if output_file else '<skip-decode>'}")
    if getattr(args, "dry_run", False):
        lines.append("metrics_file=<not written in dry-run>")
    else:
        lines.append(f"metrics_file={args.metrics_file}")
    if "generate_sec" in timings:
        lines.append(f"generate_sec={timings['generate_sec']:.3f}")
    if "total_sec" in timings:
        lines.append(f"total_sec={timings['total_sec']:.3f}")
    if "seconds_per_frame" in payload:
        lines.append(f"seconds_per_frame={payload['seconds_per_frame']:.3f}")
    if "cuda_peak_allocated_gb" in payload:
        lines.append(f"cuda_peak_allocated_gb={payload['cuda_peak_allocated_gb']:.3f}")
    if "cuda_peak_reserved_gb" in payload:
        lines.append(f"cuda_peak_reserved_gb={payload['cuda_peak_reserved_gb']:.3f}")
    errors = collect_runtime_messages(payload, "errors")
    warnings = collect_runtime_messages(payload, "warnings")
    for index, error in enumerate(errors[:4], start=1):
        lines.append(f"error[{index}]={error}")
    if len(errors) > 4:
        lines.append(f"error_more={len(errors) - 4}")
    for index, warning in enumerate(warnings[:3], start=1):
        lines.append(f"warning[{index}]={warning}")
    if len(warnings) > 3:
        lines.append(f"warning_more={len(warnings) - 3}")
    lines.append("details=use --print-json for the full metrics/config/runtime payload")
    print("\n".join(lines))


def print_final_run_metrics(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    if getattr(args, "print_json", False):
        print_metrics_json(payload)
        return
    print_run_summary(args, payload)


def sparsevideo_source_fingerprints(method: str) -> Dict[str, Any]:
    if method != "flashomni":
        return {}
    paths = {
        "flashomni_policy_sha256": REPO_ROOT / "src" / "sparsevideo" / "methods" / "flashomni" / "policy.py",
        "flashomni_method_sha256": REPO_ROOT / "src" / "sparsevideo" / "methods" / "flashomni" / "method.py",
    }
    fingerprints: Dict[str, Any] = {}
    for key, path in paths.items():
        try:
            fingerprints[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            fingerprints[key] = f"unavailable:{type(exc).__name__}"
    return fingerprints


def pipeline_output_summary(torch, value: Any) -> Dict[str, Any]:
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        summary: Dict[str, Any] = {"type": type(value).__name__, "length": len(value)}
        if value:
            summary["first"] = pipeline_output_summary(torch, value[0])
        return summary
    return {"type": type(value).__name__}


def is_torch_tensor(value: Any) -> bool:
    try:
        import torch

        return torch.is_tensor(value)
    except Exception:
        return False


def json_ready(value: Any) -> Any:
    if is_torch_tensor(value):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def quiet_runtime_status_call(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def configure_torch_compile_logging(*, verbose_compile_logs: bool) -> None:
    if verbose_compile_logs:
        return

    import torch._inductor.config as inductor_config
    import torch._inductor.select_algorithm as select_algorithm

    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)
    logging.getLogger("torch._inductor.runtime.triton_heuristics").setLevel(logging.CRITICAL)
    inductor_config.autotune_num_choices_displayed = 0
    select_algorithm.PRINT_AUTOTUNE = False


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


def sparse_attention_handle_summary(handle) -> Dict[str, Any]:
    summary = getattr(handle, "summary", None)
    if callable(summary):
        return summary()
    return {
        "type": type(handle).__name__,
        "summary_available": False,
    }


def validate_svoo_warmup_status(status: Dict[str, Any], *, strict_kernels: bool) -> Optional[str]:
    message = None
    if not status.get("enabled"):
        message = "SVOO kernel warmup is disabled; strict benchmark runs must precompile the owned kernel path."
    elif status.get("error"):
        message = f"SVOO kernel warmup failed: {status['error']}"
    elif not status.get("ran"):
        reason = status.get("reason") or "unknown"
        message = f"SVOO kernel warmup did not run: {reason}"

    if message is None:
        return None
    if strict_kernels:
        raise RuntimeError(message)
    return message


def method_requires_sparse_runtime_dispatch(method: str, method_config: Dict[str, Any]) -> bool:
    if method == "dense":
        return False
    if method == "spargeattn" and method_config.get("mode") == "full":
        return False
    if method == "flashomni" and method_config.get("is_full"):
        return False
    return True


def _runtime_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def expected_sparse_runtime_backends(method: str) -> set[str]:
    return {
        "adacluster": {
            "adacluster_flashinfer",
            "triton_cluster_sparse_attn",
            "triton_cluster_sparse_attn_topk",
        },
        "draft": {"mit_block_sparse", "triton_debug_fallback"},
        "flashomni": {
            "flashomni_explicit_upstream",
            "flashomni_global_random_upstream",
            "flashomni_local_qk_topk_upstream",
            "flex_debug_fallback",
        },
        "radial": {"flashinfer", "sage_block_sparse", "flex_attention_debug_fallback"},
        "spargeattn": {
            "spas_sage",
            "spas_sage_block_sparse",
            "spas_sage_cdfthreshd",
            "spas_sage_topk",
            "spas_sage_tuned",
        },
        "sta": {"fastvideo_sta_h100", "fastvideo_sta_a100_triton", "fastvideo_sta_triton"},
        "svg1": {"flex_attention"},
        "svg2": {"flashinfer", "triton_debug_fallback"},
        "svoo": {"svoo_flashinfer", "svoo_triton"},
    }.get(method, set())


def _debug_fallback_backends(backend_counts: Dict[str, Any]) -> list[str]:
    return sorted(
        str(backend)
        for backend, count in (backend_counts or {}).items()
        if _runtime_count(count) > 0 and "debug_fallback" in str(backend)
    )


def validate_sparse_runtime_dispatch(
    method: str,
    method_config: Dict[str, Any],
    handle_summary: Dict[str, Any],
    *,
    strict_kernels: bool,
) -> Optional[str]:
    if not method_requires_sparse_runtime_dispatch(method, method_config):
        return None

    method_runtime = (handle_summary or {}).get("method_runtime")
    if not isinstance(method_runtime, dict):
        message = (
            f"{method} did not expose runtime dispatch stats; strict quality/speed runs "
            "must prove the sparse path actually executed."
        )
    else:
        dispatch_counts = method_runtime.get("dispatch_counts") or {}
        if method == "sta" and method_config.get("STA_mode") == "STA_searching":
            search_calls = _runtime_count(dispatch_counts.get("search"))
            if search_calls > 0:
                return None
            total_calls = _runtime_count(method_runtime.get("total_calls"))
            message = (
                "sta STA_searching did not record search dispatch during calibration; "
                f"dispatch_counts={dispatch_counts}, total_calls={total_calls}."
            )
            if strict_kernels:
                raise RuntimeError(message)
            return message
        sparse_calls = _runtime_count(dispatch_counts.get("sparse"))
        if sparse_calls > 0:
            backend_counts = method_runtime.get("backend_counts") or {}
            expected_backends = expected_sparse_runtime_backends(method)
            observed_backends = {
                str(backend)
                for backend, count in backend_counts.items()
                if _runtime_count(count) > 0
            }
            matched_backends = sorted(observed_backends & expected_backends)
            if expected_backends and not matched_backends:
                message = (
                    f"{method} recorded sparse dispatch without an expected sparse backend; "
                    f"backend_counts={backend_counts}, expected={sorted(expected_backends)}. "
                    "Strict quality/speed runs require method-specific backend evidence."
                )
                if strict_kernels:
                    raise RuntimeError(message)
                return message
            debug_backends = _debug_fallback_backends(backend_counts)
            if not debug_backends:
                return None
            message = (
                f"{method} dispatched debug fallback backend(s) during generation: {debug_backends}. "
                "Strict quality/speed runs require the upstream-equivalent native sparse backend."
            )
            if strict_kernels:
                raise RuntimeError(message)
            return message
        total_calls = _runtime_count(method_runtime.get("total_calls"))
        message = (
            f"{method} did not dispatch sparse attention during generation; "
            f"dispatch_counts={dispatch_counts}, total_calls={total_calls}. "
            "Strict quality/speed runs require observed sparse/native dispatch."
        )

    if strict_kernels:
        raise RuntimeError(message)
    return message


def maybe_save_spargeattn_tuned_state(handle, method_config: Dict[str, Any]) -> Optional[str]:
    if not method_config.get("tune"):
        return None
    model_out_path = method_config.get("model_out_path")
    if not model_out_path:
        return None
    method_instance = getattr(handle, "_method_instance", None)
    export_state_dict = getattr(method_instance, "export_state_dict", None)
    if not callable(export_state_dict):
        raise RuntimeError("spargeattn tune=true did not install an exportable tuned-state method")

    import torch

    state = export_state_dict()
    if not state:
        raise RuntimeError(
            "spargeattn tune=true produced no tuned state. Check that the sparse path ran "
            "with CUDA, sequence length >=128, supported head_dim, and no attention_mask."
        )
    path = Path(str(model_out_path)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path)


def sparse_method_supported(spec: ModelSpec, method: str) -> bool:
    if method == "dense":
        return True
    if not spec.sparse_supported:
        return False
    if spec.sparse_methods is None:
        return True
    return method in spec.sparse_methods


def unsupported_sparse_method_message(spec: ModelSpec, method: str) -> str:
    method_reason = ""
    if spec.sparse_supported and spec.sparse_methods is not None and method not in spec.sparse_methods:
        method_reason = unvalidated_method_reason(method, smoke_methods=spec.sparse_methods)
    reasons = [reason for reason in (spec.unsupported_reason, method_reason) if reason]
    reason = f" {' '.join(reasons)}" if reasons else ""
    return (
        f"{method} is not implemented for {spec.family}; "
        f"compatibility_label={spec.compatibility_label}; "
        f"supported sparse methods: {list(spec.sparse_methods or ()) or 'none'}.{reason}"
    )


def should_preload_fused_native_kernels(spec: ModelSpec, method: str) -> bool:
    if method not in ("svg1", "svg2", "svoo"):
        return False
    if spec.family in ("wan", "hunyuan_video"):
        return True
    return os.environ.get("SPARSEVIDEO_FUSED_KERNEL_BACKEND") == "native"


def should_defer_fused_native_kernel_load(spec: ModelSpec, method: str, *, dry_run: bool) -> bool:
    return (not dry_run) and spec.family == "hunyuan_video" and method in ("svg2", "svoo")


def run(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[MODEL_ALIASES[args.model]]
    fps = args.fps if args.fps is not None else spec.fps
    strict_kernels = args.strict_kernels or not args.allow_debug_fallbacks
    profile_method = args.profile_for_method or args.method
    if args.num_frames is not None:
        num_frames = args.num_frames
    elif args.duration_seconds is not None:
        num_frames = default_num_frames(args.duration_seconds, fps)
    else:
        num_frames = spec.default_frames
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    try:
        profile = resolve_inference_profile(args.profile, spec, profile_method)
    except ValueError as exc:
        finalize_runtime_defaults(args)
        height = args.height if args.height is not None else DEFAULT_HEIGHT
        width = args.width if args.width is not None else DEFAULT_WIDTH
        model_id = resolve_model_id(spec, args.model_root, args.model_path)
        output_file = make_output_file(args, spec.key, args.method, num_frames)
        failed_metrics = {
            "model": spec.key,
            "model_arg": args.model,
            "model_id": model_id,
            "method": args.method,
            "method_config": {},
            "profile": args.profile,
            "profile_method": profile_method,
            "profile_overrides": {},
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "requested_duration_seconds": args.duration_seconds,
            "num_inference_steps": steps,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_offload": args.cpu_offload,
            "cpu_offload_mode": args.cpu_offload_mode,
            "vae_dtype": args.vae_dtype,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
            "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
            "strict_kernels": strict_kernels,
            "allow_debug_fallbacks": args.allow_debug_fallbacks,
            "seed": args.seed,
            "output_file": str(output_file),
            "scheduler_flow_shift": None,
            "wan_flow_shift": None,
            "runtime": {"preflight": {"errors": [str(exc)], "warnings": []}},
            "status": "failed",
            "failed_stage": "profile",
            "timings": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.dry_run:
            append_metrics(args.metrics_file, failed_metrics)
        print_final_run_metrics(args, failed_metrics)
        return 1
    height, width, fps, num_frames, steps = apply_profile_runtime_defaults(
        args, profile, fps, num_frames, steps,
    )
    if args.vae_dtype is None and spec.family == "wan":
        args.vae_dtype = "fp32"

    configure_method_runtime_env(args.method)
    import sparsevideo

    user_method_config = parse_method_config(args)
    method_config = sparsevideo.default_method_config(
        args.method, num_inference_steps=steps, model_family=spec.family, model_key=spec.key,
    )
    if profile_method == args.method:
        method_config.update(copy.deepcopy(profile.get("method_config", {})))
    method_config.update(
        sparsevideo.normalize_method_config(args.method, user_method_config)
    )
    if args.method == "flashomni":
        apply_flashomni_hunyuan_quality_defaults(spec, method_config, user_method_config)
    if args.method == "draft":
        apply_draft_runtime_layout_defaults(
            spec, height, width, num_frames, method_config, user_method_config,
        )
    if spec.pipeline_class == "HunyuanVideoImageToVideoPipeline" and args.method in ("svg1", "svg2", "svoo"):
        if "context_length" not in user_method_config:
            method_config["context_length"] = None
        if "prompt_length" not in user_method_config:
            method_config["prompt_length"] = None
    if args.method == "radial" and not strict_kernels:
        method_config["allow_flex_fallback"] = True
    if args.method == "svg2" and not strict_kernels:
        method_config["allow_triton_fallback"] = True
    if args.method == "draft" and not strict_kernels:
        method_config["allow_triton_fallback"] = True
    model_id = resolve_model_id(spec, args.model_root, args.model_path)
    output_file = make_output_file(args, spec.key, args.method, num_frames)
    scheduler_flow_shift = resolve_scheduler_flow_shift(spec, args.height, args.flow_shift)
    wan_flow_shift = scheduler_flow_shift if spec.family == "wan" else None
    unsupported = not sparse_method_supported(spec, args.method)
    try:
        if not unsupported:
            materialize_method_config_values(args.method, method_config)
            if args.method == "spargeattn":
                normalize_spargeattn_model_out_path(method_config, output_file)
            if (
                args.method == "svoo"
                and method_config.get("use_dynamic_min_kc_ratio")
                and (
                    not method_config.get("sparsity_csv_path")
                    or method_config.get("sparsity_csv_path") == "sparsity_profiles/sparsity_results.csv"
                )
            ):
                method_config["sparsity_csv_path"] = default_svoo_sparsity_csv_path(spec)
            validate_method_config(args.method, method_config, model_family=spec.family)
    except (FileNotFoundError, NotImplementedError, TypeError, ValueError) as exc:
        failed_metrics = {
            "model": spec.key,
            "model_arg": args.model,
            "model_id": model_id,
            "method": args.method,
            "method_config": method_config,
            "profile": args.profile,
            "profile_method": profile_method,
            "profile_overrides": profile,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "requested_duration_seconds": args.duration_seconds,
            "num_inference_steps": steps,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_offload": args.cpu_offload,
            "cpu_offload_mode": args.cpu_offload_mode,
            "vae_dtype": args.vae_dtype,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
            "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
            "strict_kernels": strict_kernels,
            "allow_debug_fallbacks": args.allow_debug_fallbacks,
            "seed": args.seed,
            "output_file": str(output_file),
            "scheduler_flow_shift": scheduler_flow_shift,
            "wan_flow_shift": wan_flow_shift,
            "runtime": {"preflight": {"errors": [str(exc)], "warnings": []}},
            "status": "failed",
            "failed_stage": "validate_method_config",
            "timings": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.dry_run:
            append_metrics(args.metrics_file, failed_metrics)
        print_final_run_metrics(args, failed_metrics)
        return 1
    from sparsevideo._runtime import (
        adacluster_load_status,
        draft_block_sparse_load_status,
        flash_attn_load_status,
        flashomni_load_status,
        flashinfer_load_status,
        native_kernel_load_status,
        optional_kernel_status,
        radial_runtime_load_status,
        sageattention_load_status,
        spas_sage_attn_load_status,
        sta_load_status,
        svg1_runtime_load_status,
        svg2_runtime_load_status,
        svoo_runtime_load_status,
        torch_runtime_status,
    )

    runtime_status = {
        "optional_kernels": optional_kernel_status(),
        "torch": torch_runtime_status(),
    }
    defer_fused_native_kernel_load = should_defer_fused_native_kernel_load(
        spec, args.method, dry_run=args.dry_run,
    )
    if should_preload_fused_native_kernels(spec, args.method) and not defer_fused_native_kernel_load:
        runtime_status["optional_kernels"]["svg_svoo_fused_kernels"].update(
            native_kernel_load_status()
        )
    needs_flash_attn_load = (
        not unsupported
        and (
            args.method == "draft"
            or (args.method in ("svg1", "adacluster") and spec.family == "hunyuan_video")
        )
    )
    if needs_flash_attn_load:
        runtime_status["optional_kernels"].setdefault("flash_attn", {}).update(
            quiet_runtime_status_call(flash_attn_load_status)
        )
    if not unsupported and args.method == "adacluster":
        runtime_status["optional_kernels"].setdefault("adacluster_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(adacluster_load_status))
        runtime_status["optional_kernels"]["adacluster_kernels"]["load_checked"] = True
    if not unsupported and args.method == "draft":
        runtime_status["optional_kernels"].setdefault("draft_kernels", {}).setdefault(
            "mit_block_sparse_attn", {}
        ).update(quiet_runtime_status_call(draft_block_sparse_load_status))
        runtime_status["optional_kernels"]["draft_kernels"]["mit_load_checked"] = True
    needs_flashinfer_load = (
        not unsupported
        and (
            args.method in ("adacluster", "radial", "svg2")
            or (args.method == "svoo" and method_config.get("sparse_backend") == "flashinfer")
        )
    )
    if needs_flashinfer_load:
        runtime_status["optional_kernels"].setdefault("flashinfer", {}).update(
            quiet_runtime_status_call(flashinfer_load_status)
        )
    if not unsupported and args.method == "radial":
        runtime_status["optional_kernels"].setdefault("radial_kernels", {}).setdefault(
            "owned_runtime", {}
        ).update(quiet_runtime_status_call(radial_runtime_load_status))
        runtime_status["optional_kernels"]["radial_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svg1":
        runtime_status["optional_kernels"].setdefault("svg1_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svg1_runtime_load_status))
        runtime_status["optional_kernels"]["svg1_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svg2":
        runtime_status["optional_kernels"].setdefault("svg2_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svg2_runtime_load_status))
        runtime_status["optional_kernels"]["svg2_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svoo":
        runtime_status["optional_kernels"].setdefault("svoo_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svoo_runtime_load_status))
        runtime_status["optional_kernels"]["svoo_kernels"]["load_checked"] = True
    needs_flashomni_load = (
        not unsupported
        and args.method == "flashomni"
        and method_config.get("implementation") == "upstream"
    )
    if needs_flashomni_load:
        runtime_status["optional_kernels"].setdefault("flashomni", {}).update(
            quiet_runtime_status_call(flashomni_load_status)
        )
    if not unsupported and args.method == "sta":
        runtime_status["optional_kernels"].setdefault("sta_kernels", {}).update(
            quiet_runtime_status_call(sta_load_status)
        )
    spargeattn_needs_runtime = (
        not unsupported
        and args.method == "spargeattn"
        and (
            method_config.get("mode") != "full"
            or method_config.get("tune")
            or method_config.get("model_out_path")
        )
    )
    radial_needs_sparge = (
        not unsupported and args.method == "radial" and method_config.get("use_sage_attention")
    )
    if spargeattn_needs_runtime or radial_needs_sparge:
        runtime_status["optional_kernels"].setdefault("spas_sage_attn", {}).update(
            quiet_runtime_status_call(
                spas_sage_attn_load_status,
                require_autotune=bool(
                    args.method == "spargeattn"
                    and (method_config.get("tune") or method_config.get("model_out_path"))
                ),
            )
        )
    radial_needs_sageattention = (
        radial_needs_sparge
        and (
            int(method_config.get("dense_timesteps", 0) or 0) > 0
            or int(method_config.get("dense_layers", 0) or 0) > 0
        )
    )
    if radial_needs_sageattention:
        runtime_status["optional_kernels"].setdefault("sageattention", {}).update(
            quiet_runtime_status_call(sageattention_load_status)
        )
    if unsupported:
        runtime_status["preflight"] = {"errors": [], "warnings": []}
    else:
        runtime_status["preflight"] = preflight_runtime(
            args.method,
            method_config,
            args.device,
            runtime_status,
            strict_kernels=strict_kernels,
            model_family=spec.family,
        )
    if not unsupported and args.method == "draft":
        draft_error = draft_upstream_layout_error(
            spec, height, width, num_frames, method_config,
        )
        if draft_error is not None:
            runtime_status["preflight"]["errors"].append(draft_error)
    if not unsupported and args.method == "radial":
        radial_warning = radial_flashinfer_layout_warning(
            spec, height, width, num_frames, method_config,
        )
        if radial_warning is not None:
            if method_config.get("use_sage_attention") or strict_kernels:
                runtime_status["preflight"]["errors"].append(radial_warning)
            else:
                runtime_status["preflight"]["warnings"].append(radial_warning)
    if not unsupported and args.method == "sta":
        sta_messages = sta_layout_preflight_messages(
            spec, height, width, num_frames, method_config,
            strict_kernels=strict_kernels,
        )
        runtime_status["preflight"]["errors"].extend(sta_messages["errors"])
        runtime_status["preflight"]["warnings"].extend(sta_messages["warnings"])
    runtime_status["preflight"]["errors"].extend(
        model_shape_preflight_errors(spec, height, width)
    )
    runtime_status["preflight"]["warnings"].extend(
        model_quality_warnings(spec, height, width)
    )
    base_metrics: Dict[str, Any] = {
        "model": spec.key,
        "model_arg": args.model,
        "model_id": model_id,
        "method": args.method,
        "method_config": method_config,
        "profile": args.profile,
        "profile_method": profile_method,
        "profile_overrides": profile,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
        "requested_duration_seconds": args.duration_seconds,
        "num_inference_steps": steps,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "cpu_offload_mode": args.cpu_offload_mode,
        "vae_dtype": args.vae_dtype,
        "vae_tiling": args.vae_tiling,
        "vae_slicing": args.vae_slicing,
        "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
        "skip_decode": args.skip_decode,
        "output_type": "latent" if args.skip_decode else spec.output_type,
        "compatibility_label": spec.compatibility_label,
        "unsupported_reason": spec.unsupported_reason,
        "strict_kernels": strict_kernels,
        "allow_debug_fallbacks": args.allow_debug_fallbacks,
        "seed": args.seed,
        "negative_prompt": args.negative_prompt,
        "output_file": None if args.skip_decode else str(output_file),
        "scheduler_flow_shift": scheduler_flow_shift,
        "wan_flow_shift": wan_flow_shift,
        "runtime": runtime_status,
        "source_fingerprints": sparsevideo_source_fingerprints(args.method),
    }

    if args.dry_run:
        if unsupported:
            base_metrics.update(status="unsupported_dry_run")
            base_metrics["error"] = unsupported_sparse_method_message(spec, args.method)
            print_final_run_metrics(args, base_metrics)
            return 0
        if runtime_status["preflight"]["errors"]:
            base_metrics.update(
                status="failed",
                failed_stage="preflight",
                timings={},
                error_type="RuntimeError",
                error="; ".join(runtime_status["preflight"]["errors"]),
            )
            print_final_run_metrics(args, base_metrics)
            return 1
        base_metrics.update(status="dry_run")
        print_final_run_metrics(args, base_metrics)
        return 0

    if unsupported:
        base_metrics.update(
            status="unsupported",
            error=unsupported_sparse_method_message(spec, args.method),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 2

    if runtime_status["preflight"]["errors"]:
        base_metrics.update(
            status="failed",
            failed_stage="preflight",
            timings={},
            error_type="RuntimeError",
            error="; ".join(runtime_status["preflight"]["errors"]),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 1

    stage = "start"
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    handle = None

    try:
        stage = "import"
        with redirect_stdout(sys.stderr):
            import torch
            from diffusers.utils import export_to_video
        configure_torch_compile_logging(verbose_compile_logs=args.verbose_compile_logs)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        seed_everything(torch, args.seed)
        try:
            torch.backends.cuda.preferred_linalg_library(backend="magma")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt = read_prompt(args)
        torch_dtype = parse_dtype(torch, args.dtype)
        vae_torch_dtype = parse_dtype(torch, args.vae_dtype) if args.vae_dtype is not None else None

        stage = "load_pipeline"
        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            pipe = load_pipeline(
                spec,
                model_id,
                torch_dtype,
                vae_torch_dtype,
                args.local_files_only,
                height=args.height,
                flow_shift=scheduler_flow_shift,
            )
            prepare_pipeline(
                pipe,
                args.device,
                args.cpu_offload,
                args.vae_tiling,
                args.vae_slicing,
                cpu_offload_mode=args.cpu_offload_mode,
                vae_decoder_chunk_size=args.vae_decoder_chunk_size,
            )
            sync_if_cuda(torch, args.device)
        timings["load_pipeline_sec"] = time.perf_counter() - t0

        if defer_fused_native_kernel_load:
            stage = "deferred_runtime_preflight"
            with redirect_stdout(sys.stderr):
                runtime_status["optional_kernels"]["svg_svoo_fused_kernels"].update(
                    native_kernel_load_status()
                )
            deferred_preflight = preflight_runtime(
                args.method,
                method_config,
                args.device,
                runtime_status,
                strict_kernels=strict_kernels,
                model_family=spec.family,
            )
            runtime_status["preflight"]["warnings"].extend(deferred_preflight["warnings"])
            if deferred_preflight["errors"]:
                runtime_status["preflight"]["errors"].extend(deferred_preflight["errors"])
                raise RuntimeError("; ".join(deferred_preflight["errors"]))

        stage = "apply_sparse_attention"
        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            if args.method in ("svg1", "svg2", "svoo") and spec.family == "hunyuan_video":
                hunyuan_i2v = spec.pipeline_class == "HunyuanVideoImageToVideoPipeline"
                if method_config.get("context_length") is None and not hunyuan_i2v:
                    method_config["context_length"] = 256
                if method_config.get("prompt_length") is None and not hunyuan_i2v:
                    method_config["prompt_length"] = infer_hunyuan_prompt_length(
                        pipe, prompt, int(method_config["context_length"]),
                    )
            handle = sparsevideo.apply_sparse_attention(pipe, method=args.method, config=method_config)
            base_metrics["sparse_attention_handle"] = sparse_attention_handle_summary(handle)
            sync_if_cuda(torch, args.device)
        timings["apply_sparse_attention_sec"] = time.perf_counter() - t0

        if args.method == "svoo":
            stage = "svoo_kernel_warmup"
            t0 = time.perf_counter()
            with redirect_stdout(sys.stderr):
                from sparsevideo.methods.svoo.warmup import warmup_svoo_kernels_from_pipeline

                warmup_status = warmup_svoo_kernels_from_pipeline(
                    pipe,
                    model_type=spec.family,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    config=method_config,
                    dtype=torch_dtype,
                    device=args.device,
                )
                sync_if_cuda(torch, args.device)
            timings["svoo_kernel_warmup_sec"] = time.perf_counter() - t0
            base_metrics["svoo_kernel_warmup"] = warmup_status
            warmup_warning = validate_svoo_warmup_status(
                warmup_status, strict_kernels=strict_kernels,
            )
            if warmup_warning is not None:
                runtime_status["preflight"]["warnings"].append(warmup_warning)

        stage = "generate"
        if not args.skip_decode:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        if not args.skip_decode and args.skip_existing and output_file.exists():
            base_metrics.update(status="skipped_existing", timings=timings)
            with redirect_stdout(sys.stderr):
                handle.restore()
            base_metrics["sparse_attention_handle_after_restore"] = sparse_attention_handle_summary(handle)
            append_metrics(args.metrics_file, base_metrics)
            print_final_run_metrics(args, base_metrics)
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
        if spec.pipeline_class == "HunyuanVideoImageToVideoPipeline":
            base_metrics["hunyuan_i2v_prompt_template_compat"] = apply_hunyuan_i2v_prompt_template_compat(
                pipe, call_kwargs,
            )

        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            result = call_pipeline_with_model_compat(pipe, call_kwargs, torch, spec, args.device)
            sync_if_cuda(torch, args.device)
        base_metrics["sparse_attention_handle"] = sparse_attention_handle_summary(handle)
        timings["generate_sec"] = time.perf_counter() - t0

        stage = "validate_sparse_dispatch"
        dispatch_warning = validate_sparse_runtime_dispatch(
            args.method,
            method_config,
            base_metrics["sparse_attention_handle"],
            strict_kernels=strict_kernels,
        )
        if dispatch_warning is not None:
            runtime_status.setdefault("generation_checks", {"errors": [], "warnings": []})
            runtime_status["generation_checks"]["warnings"].append(dispatch_warning)

        stage = "spargeattn_save_state"
        t0 = time.perf_counter()
        tuned_state_path = None
        if args.method == "spargeattn":
            with redirect_stdout(sys.stderr):
                tuned_state_path = maybe_save_spargeattn_tuned_state(handle, method_config)
        timings["spargeattn_save_state_sec"] = time.perf_counter() - t0
        if tuned_state_path is not None:
            base_metrics["spargeattn_tuned_state_path"] = tuned_state_path

        if args.skip_decode:
            stage = "summarize_latent_output"
            base_metrics["latent_output"] = pipeline_output_summary(torch, getattr(result, "frames", None))
            timings["export_video_sec"] = 0.0
        else:
            stage = "export_video"
            t0 = time.perf_counter()
            with redirect_stdout(sys.stderr):
                export_to_video(result.frames[0], str(output_file), fps=fps)
            timings["export_video_sec"] = time.perf_counter() - t0
        with redirect_stdout(sys.stderr):
            handle.restore()
        base_metrics["sparse_attention_handle_after_restore"] = sparse_attention_handle_summary(handle)
        handle = None

        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="ok",
            timings=timings,
            seconds_per_frame=timings["generate_sec"] / max(num_frames, 1),
            **cuda_memory_gb(torch),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 0
    except Exception as exc:
        restore_error = None
        if handle is not None:
            base_metrics["sparse_attention_handle"] = sparse_attention_handle_summary(handle)
            try:
                with redirect_stdout(sys.stderr):
                    handle.restore()
                base_metrics["sparse_attention_handle_after_restore"] = sparse_attention_handle_summary(handle)
            except Exception as restore_exc:
                restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="failed",
            failed_stage=stage,
            timings=timings,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if restore_error is not None:
            base_metrics["restore_error"] = restore_error
        append_metrics(args.metrics_file, base_metrics)
        if args.print_json:
            traceback.print_exc()
        print_final_run_metrics(args, base_metrics)
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
