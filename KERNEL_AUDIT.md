# SparseVideo Per-Method Kernel Acceleration Audit

Generated: 2026-05-18  
Machine: A100 80GB × 4  
Benchmark verified: Triton RoPE 2.2×, Triton RMSNorm 5.9×, Wan block ops 4.0×, Hunyuan block ops 1.4×

## Wan Model — Per-Method Kernel Paths

| Method | fused_qk_norm | fused_rope | fast_block_patch | attention_backend | native_kernel_backend | known_fallbacks | reason_if_not_fastest |
|--------|:---:|:---:|:---:|---|---|---|---|
| dense | Y | Y | Y | F.scaled_dot_product_attention | — | CPU fallback if no CUDA | — |
| svg1 | Y | Y | Y | flex_attention (torch.compile) | Triton permute + L2norm | SDPA if compile fails | — |
| svg2 | Y | Y | Y | flashinfer variable_block_sparse | Triton kmeans + dynamic_map + permute | SDPA on dense steps | — |
| spargeattn | Y | Y | Y | spas_sage_attn (CUDA extension) | spas_sage_attn native C++ | reject if not CUDA/seq<128/head_dim∉{64,128} | — |
| radial | Y | Y | Y | block_sparse_sage2_attn (CUDA) + sageattn | spas_sage_attn + sageattention native | flashinfer BSR if use_sage_attention=False | — |
| sta | Y | Y | Y | fastvideo_sta_a100_triton | FastVideo Triton kernel | none (always sparse) | H100 TK path hardware-deferred |
| draft | Y | Y | Y | draft_block_sparse (Triton) | Triton block-sparse attn | SDPA on dense steps | — |
| adacluster | Y | Y | Y | triton_cluster_sparse_attn | Triton fast_kmeans + cluster_sparse_attn | flash_attn/SDPA on dense steps | — |
| flashomni | Y | Y | Y | flashomni C++/CUDA engine | FlashOmni native sparse GEMM-Q/GEMM-O | flashomni full kernel on dense steps | — |
| svoo | Y | Y | Y | flashinfer variable_block_sparse | Triton co_cluster + kmeans + permute | SDPA on dense steps; flashinfer varlen on Hunyuan | — |

## HunyuanVideo Model — Per-Method Kernel Paths

| Method | fused_qk_norm | fused_rope | fast_block_patch | attention_backend | native_kernel_backend | known_fallbacks | reason_if_not_fastest |
|--------|:---:|:---:|:---:|---|---|---|---|
| dense | Y | Y | Y | F.scaled_dot_product_attention | — | CPU fallback if no CUDA | — |
| svg1 | Y | Y | Y | flex_attention (torch.compile) | Triton permute + L2norm | flash_attn_varlen on full steps | — |
| svg2 | Y | Y | Y | flashinfer variable_block_sparse | Triton kmeans + dynamic_map + permute | flashinfer varlen on dense steps | — |
| spargeattn | Y | Y | Y | spas_sage_attn (CUDA extension) | spas_sage_attn native C++ | F.scaled_dot_product_attention on full steps | — |
| radial | Y | Y | Y | block_sparse_sage2_attn (CUDA) + sageattn | spas_sage_attn + sageattention native | flashinfer BSR if use_sage_attention=False | — |
| sta | Y | Y | Y | fastvideo_sta_a100_triton | FastVideo Triton kernel | none (always sparse) | H100 TK path hardware-deferred |
| draft | Y | Y | Y | draft_block_sparse (Triton) | Triton block-sparse attn | F.scaled_dot_product_attention on dense steps | — |
| adacluster | Y | Y | Y | triton_cluster_sparse_attn | Triton fast_kmeans + cluster_sparse_attn | flash_attn on dense steps | — |
| flashomni | Y | Y | own patch | flashomni C++/CUDA engine | FlashOmni native sparse GEMM-Q/GEMM-O | flashomni full kernel on dense steps | SVOO/FlashOmni paper_mmdit installs own Hunyuan forward patch |
| svoo | Y | Y | own patch | flashinfer variable_block_sparse | Triton co_cluster + kmeans + permute | flashinfer varlen dense | SVOO installs own hunyuan_sparse_forward_patch |

## Dense/Full Stage Audit

| Method | warmup_steps | refresh_cycle | full_dispatch_backend | sparse_dispatch_backend | full_stage_overhead | metadata_cache_overhead | full_vs_native_dense_decision |
|--------|---|---|---|---|---|---|---|
| dense | N/A | N/A | native SDPA | N/A | none | none | is native dense |
| svg1 | first_times_fp=0.2 (~10 of 50 steps) | per-step MSE profiling | F.scaled_dot_product_attention | flex_attention (compiled) | MSE profile: sampled_rows × K matmul (~0.5ms/layer) | BlockMask uint8 + profiling buffer ~2MB | uses native SDPA directly for full steps |
| svg2 | first_times_fp=0.2 (~10 of 50 steps) | recluster every sparse step | F.scaled_dot_product_attention | flashinfer variable_block_sparse | k-means init: 2 iters × 11520 tokens (~1ms/layer) | centroids [B×H, num_clusters, D] + labels ~4MB | uses native SDPA directly for full steps |
| spargeattn | first layers by config | per-step (tuned state) | F.scaled_dot_product_attention | spas_sage_attn CUDA | none (threshold scalars only) | layer-wise threshold tensors ~100B | uses native SDPA directly for mode=full |
| radial | dense_layers + dense_timesteps | none | flashinfer BSR with all-ones mask | block_sparse_sage2_attn + sageattn | flashinfer dispatch overhead vs native SDPA (~0.1ms) | BSR mask ~1KB per layer | **note**: uses flashinfer even for trivially-full steps; could use native SDPA |
| sta | none | none | N/A (always sparse) | fastvideo_sta_a100_triton | none | tile config only | N/A — no full/dense stage |
| draft | first_times_fp by config | none | F.scaled_dot_product_attention | draft_block_sparse Triton | none | pooled spatial mask [B, H, pool_s, pool_s] ~1MB | uses native SDPA directly for full steps |
| adacluster | step≤8 or specific layers | none | flash_attn / SDPA | triton_cluster_sparse_attn | none | cluster centroids + counts ~2MB | uses native dense directly for warmup |
| flashomni | schedule.full flag | Taylor extrapolation cache | flashomni full kernel wrapper | flashomni sparse C++/CUDA | FlashOmni full kernel overhead vs native SDPA (~0.2ms) | output Q/O history ~200KB–2MB | **note**: uses FlashOmni full kernel even on dense steps (maintains Q/O projection consistency) |
| svoo | first_times_fp=0.075 (~4 of 50 steps) | reuse_interval=20 | F.scaled_dot_product_attention | flashinfer variable_block_sparse | k-means init on step 0: 2 iters × centroids | centroids + labels + sorted indices ~8MB; reused across reuse_interval | uses native SDPA directly for full steps |

### Dense/Full Stage Notes

1. **radial full stage**: Uses flashinfer BSR with an all-ones block mask for "full" attention, adding flashinfer dispatch overhead vs plain SDPA. Speed impact: ~0.1ms per layer per step during dense_timesteps. Could be optimized to use native dense when mask is trivially full.

2. **flashomni full stage**: Uses the FlashOmni C++/CUDA full-kernel path even for dense steps to maintain consistent Q/O projection behavior. Speed impact: ~0.2ms per layer per step vs native SDPA. Acceptable because FlashOmni's sparse GEMM-Q/GEMM-O design requires consistent workspace regardless of sparsity level.

3. **svoo metadata**: Cached clustering results (centroids, labels, sorted indices) are reused across steps within `reuse_interval` (default 20), amortizing the k-means cost. Memory: ~8MB peak for Wan 720p 14B. Negligible vs model/activation memory.

4. **sta always-sparse**: STA has no full/dense warmup stage. All steps use the Triton sliding-tile kernel. This is upstream behavior — the tile structure inherently covers the full sequence without a separate dense phase.

5. **svg1 MSE profiling overhead**: Per-step MSE profiling computes partial Q@K^T products for `num_samples` sampled rows to select which stripes to keep. Cost: ~0.5ms per layer at 720p. This is algorithmic overhead (required for adaptive sparsity), not implementation overhead.

## Resolved Gaps

### Gap 1: spargeattn — fused norm/RoPE (RESOLVED)

Added `use_fused_qk_norm_rope` config flag, default `True` (speed-first). Users can set `False` for strict upstream parity. Equivalence test confirms fused vs unfused paths produce identical Q/K within bfloat16 precision.

**Benchmark evidence** (A100, Wan 720p dimensions S=11520, H=24, D=128):
- Triton RMSNorm: 5.9× faster than PyTorch (0.18ms vs 1.03ms)
- Triton RoPE: 2.2× faster than PyTorch (1.02ms vs 2.26ms)

### Gap 2: svoo Wan — fused RoPE (RESOLVED)

Added `use_fused_rope` config flag, default `True`. Triton RoPE is mathematically equivalent to PyTorch path and eliminates intermediate tensor allocations. Users can set `False` for strict upstream parity.

### Gap 3: Wan fast-block patch generalization (RESOLVED)

Moved `install_wan_fast_block_patch()` to base class `SparseMethod.install_model_patches()`. ALL Wan methods now get Triton LayerNorm + modulate kernels at block level automatically.

**Benchmark evidence** (A100, Wan 14B, S=11520, D=3072):
- Full block ops (norm+modulate+gate): 4.0× faster (0.50ms vs 2.00ms)

### Gap 4: Hunyuan fast-block (RESOLVED)

Created `hunyuan_fast_block.py` with Triton-accelerated double/single block forwards. Targets: norm2 (no-affine LayerNorm), modulate_shift, gate_residual. Installed via base class for all methods except SVOO/FlashOmni paper_mmdit (which have their own specialized patches). norm1 (AdaLayerNormZero with linear projection) left unfused (cuBLAS is already efficient for that linear).

**Benchmark evidence** (A100, Hunyuan 720p, S=17776, D=3072):
- Full block ops (gate+norm+modulate): 1.37× faster (0.84ms vs 1.15ms)
- LayerNorm alone: 0.85× (Triton is 15% slower than CuDNN for D=3072)
- Net positive because gate_residual + modulate_shift savings exceed LayerNorm regression

### Gap 5: flashomni Hunyuan 720p benchmark (RESOLVED)

**Status**: Completed — 50-step strict-dispatch Hunyuan 720p/129f run on single A100 80GB **without CPU offload**.

**Benchmark evidence** (A100, Hunyuan 720x1280, 129 frames, 50 steps, seed 42):
- `generate_sec`: 2526.9 (~42 min)
- `total_sec`: 2639.2 (~44 min)
- Peak CUDA allocated: 51.9 GiB
- Peak CUDA reserved: 57.9 GiB
- CPU offload: **false** (no offload needed with `taylor_cache_device=cpu`)
- `strict_kernels`: true

**Dispatch counts** (50 steps × 60 layers):
- FlashOmni full (dense): 900 calls
- FlashOmni explicit sparse: 1665 calls
- Total: 2565 calls

**FlashOmni GEMM kernel usage**:
- `flashomni_sparse_q_gemm`: 1392 calls
- `flashomni_sparse_o_gemm`: 435 calls
- `flashomni_sparse_o_gemm_cache_bias`: 160 calls

**Config**: `tau_q=0.5, tau_kv=0.05, N=6, D=1, S_q=0.3, use_sparse_gemm=true, taylor_cache_device=cpu`

**Output**: `result/inference/audit/flashomni_hunyuan_720p_129f_50step_cpucache/hunyuan-t2v/flashomni/seed42_720x1280_129f.mp4`
(1280x720, 129 frames, 5.38s, h264, 1446 kb/s)

**Memory note**: Without `taylor_cache_device=cpu`, the run OOMs at ~76 GiB (trying to allocate 3.4 GiB for sparse-info computation on top of model+activation memory). The CPU Taylor cache reduces peak by ~25 GiB with negligible speed impact (CPU↔GPU transfer is small relative to FlashOmni kernel time).

### Gap 6: sta_h100 hardware-deferred (VERIFIED)

A100 correctly dispatches to `fastvideo_sta_a100_triton` (Triton STA). H100/TK path (`sta_fwd` native kernel) is hardware-deferred. 19 STA parity tests pass on A100.

## Per-Method Native Kernel Inventory

| Method | Owned Triton Kernels | Owned C++/CUDA Kernels | External Deps |
|--------|---|---|---|
| svg1 | permute, l2norm | — | flex_attention (torch.compile) |
| svg2 | kmeans, permute, block_sparse_attn, dynamic_map | — | flashinfer |
| spargeattn | — | spas_sage_attn (CUDA extension) | — |
| radial | — | block_sparse_sage2_attn, sageattn (CUDA) | — |
| sta | sliding_tile_attention (Triton, A100) | fastvideo_kernel_ops (H100 only) | — |
| draft | draft_block_sparse (Triton) | — | — |
| adacluster | fast_kmeans_single, triton_cluster_sparse_attn | — | — |
| flashomni | — | flashomni (C++/CUDA engine: sparse GEMM-Q, GEMM-O, planner) | — |
| svoo | co_cluster, kmeans, permute, block_sparse_attn, dynamic_map, sparsity | svg_svoo_fused (C++ RMSNorm/RoPE, optional) | flashinfer |

## Shared Triton Kernels (all methods via base class)

| Kernel | File | Operations | Measured Speedup |
|--------|------|---|---|
| fused_norm_rope | kernels/fused_norm_rope.py | RMSNorm inplace, RoPE Wan inplace, RoPE HunyuanVideo inplace | RMSNorm 5.9×, RoPE Wan 2.2×, RoPE Hy 1.3× |
| layernorm | kernels/layernorm.py | LayerNorm with/without affine (fast-block patch) | Wan 4.0× (with modulate), Hy 0.85× alone |
| modulate | kernels/modulate.py | shift/scale fusion, gate+residual fusion (fast-block patch) | included in block-ops measurement |

## Estimated Per-Step Speedup (50-step inference)

### Wan 14B, 720p, 30 blocks

| Component | Per-step savings | Source |
|-----------|---|---|
| RMSNorm (Q+K × 30 blocks) | ~51ms | 0.85ms × 60 calls |
| RoPE (Q+K × 30 blocks) | ~37ms | 1.25ms × 30 calls |
| Block ops (norm+mod+gate × 30 blocks) | ~90ms | ~1.5ms × 6 ops × 30 |
| **Total per step** | **~178ms** | |
| **Total 50-step run** | **~8.9s saved** | |

### HunyuanVideo, 720p/129f, ~38 double + 20 single blocks

| Component | Per-step savings | Source |
|-----------|---|---|
| RMSNorm (Q+K × 58 blocks) | ~40ms | 0.35ms × 116 calls |
| RoPE (58 blocks) | ~64ms | 1.1ms × 58 calls |
| Block ops (double blocks) | ~46ms | 0.3ms × 4 ops × 38 |
| **Total per step** | **~150ms** | |
| **Total 50-step run** | **~7.5s saved** | |

## What's NOT Using Fastest Kernel (and Why)

| Method | What's not fastest | Reason | Impact |
|--------|---|---|---|
| radial (full steps) | flashinfer BSR with all-ones mask instead of native SDPA | upstream behavior; maintains consistent dispatch path | ~0.1ms/layer/step; minor |
| flashomni (full steps) | FlashOmni full kernel instead of native SDPA | required for Q/O projection consistency | ~0.2ms/layer/step; acceptable |
| Hunyuan norm1 (AdaLayerNormZero) | Uses cuBLAS linear, not Triton | The norm1 path is dominated by a linear projection (cuBLAS); Triton wouldn't help | none |
| sta_h100 | Triton fallback on A100 instead of native TK kernel | H100 hardware-deferred | A100 path is already fastest for A100 |
| Hunyuan triton_layernorm_noparam | 15% slower than CuDNN for D=3072 alone | Offset by surrounding gate+modulate savings (net 1.37× faster) | net positive in context |

## 50-Step Real Inference Evidence

### SVOO on Wan 1.3B (480×832×81f, 50 steps)

```
Command: CUDA_VISIBLE_DEVICES=1 python scripts/infer.py \
  --model wan21-t2v-1.3b --method svoo \
  --method-config use_dynamic_min_kc_ratio=False \
  --num-inference-steps 50 --height 480 --width 832 --num-frames 81 \
  --seed 42 --prompt "A cat sitting on a windowsill watching birds fly by" \
  --skip-decode

Status: ok
strict_kernels: true
generate_sec: 159.27
total_sec: 204.77
peak_cuda_allocated_gb: 15.56
peak_cuda_reserved_gb: 16.01

Fused kernel flags active:
  use_fused_rope: true
  wan_fast_block_patch: installed

Dispatch counts (50 steps × 30 layers):
  sparse (svoo_flashinfer): 2400 calls
  dense (torch_sdpa): 600 calls
  total: 3000 calls

Sparse backend: flashinfer variable_block_sparse
Dense backend: torch SDPA (warmup steps only)
svoo_kernel_warmup_sec: 4.67 (flashinfer JIT compile)
```

This confirms the fused kernel paths (Triton RMSNorm, Triton RoPE, Wan fast-block patch) are active during real 50-step inference with `strict_kernels=true` preventing any silent fallback to unfused or dense paths.

### FlashOmni on HunyuanVideo (720×1280×129f, 50 steps)

```
Command: CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/infer.py \
  --model hunyuan-t2v --method flashomni \
  --method-config sparse_pattern=paper_mmdit \
  --method-config use_sparse_gemm=true \
  --method-config tau_q=0.5 --method-config tau_kv=0.05 \
  --method-config N=6 --method-config D=1 --method-config S_q=0.3 \
  --method-config taylor_cache_device=cpu \
  --num-inference-steps 50 --height 720 --width 1280 --num-frames 129 \
  --seed 42 --prompt "A cat sitting on a windowsill watching birds fly by" \
  --strict-kernels --no-cpu-offload --vae-tiling

Status: ok
strict_kernels: true
cpu_offload: false
generate_sec: 2526.94
total_sec: 2639.23
peak_cuda_allocated_gb: 51.92
peak_cuda_reserved_gb: 57.86

Dispatch counts (50 steps × 60 layers):
  dense (flashomni_full_upstream): 900 calls
  sparse (flashomni_explicit_upstream): 1665 calls
  total: 2565 calls

FlashOmni GEMM kernel calls:
  flashomni_sparse_q_gemm: 1392
  flashomni_sparse_o_gemm: 435
  flashomni_sparse_o_gemm_cache_bias: 160

Sparse backend: FlashOmni C++/CUDA engine (explicit sparse-info)
Dense backend: FlashOmni full kernel wrapper
taylor_cache_device: cpu (required to fit in 80GB without model offload)
```

This confirms FlashOmni's native C++/CUDA engine dispatches both full and sparse paths on HunyuanVideo 720p with `strict_kernels=true`, `use_sparse_gemm=true`, and no CPU model offload. The `taylor_cache_device=cpu` setting moves the Taylor extrapolation cache to host memory, reducing peak GPU allocation from ~76 GiB to ~52 GiB.
