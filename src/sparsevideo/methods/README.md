# Method Packages

Each sparse attention method owns a package under `sparsevideo.methods`.

Required layout:

```text
methods/<name>/
  __init__.py   # lazy public exports only
  config.yaml   # editable default hyperparameters
  config.py     # YAML loading, aliases, dynamic/model-aware logic
  method.py     # SparseMethod adapter and processor wiring
```

Complex methods should split local concerns instead of growing `method.py`:

```text
methods/<name>/
  ops.py        # method-specific attention logic and state helpers
  kernels/      # kernels used only by this method
```

Keep shared model integration in `sparsevideo.processors` and shared reusable
kernel primitives in `sparsevideo.kernels`. A method package may depend on
those shared layers, but shared layers must not import method packages.

## Fidelity Rules

The upstream `training_free/` repositories are the semantic reference. A
method package must not rename an upstream public parameter unless it is kept
only as a compatibility alias. If an upstream option is recognized but not
ported, the method should reject it when enabled instead of silently changing
behavior.

## Config Defaults

Each method has `methods/<name>/config.yaml`. Edit `defaults` for generic
settings and `model_defaults` for concrete backbone keys such as
`wan21-t2v-1.3b` or `cogvideox-i2v`. Do not add runtime-type-wide groups such as
`wan` or `hunyuan_video`; config resolution only applies exact `model_key`
entries. `config.py` should only keep alias validation and dynamic values such
as local asset paths.

## Common Dense Warmup

All sparse methods use the same common dense warmup gates:
`dense_warmup_step_ratio` for early denoising steps and
`dense_warmup_layer_ratio` for earliest transformer layers. Method-local
dense step/layer/timestep gates are not part of dispatch.

Current status:

```text
dense       Baseline; original processor path.
svg1        Adapter port. Uses upstream names/defaults; mask construction is
            implemented locally with upstream-reference tests for profiling
            masks, FlexAttention common masks, head placement, warmup gates,
            CPU-RNG row sampling, and upstream warmup dense backends: Wan SDPA
            layout plus Hunyuan varlen FlashAttention. The sparse path requires
            PyTorch FlexAttention and preflights flex_attention, BlockMask, and
            torch.compile before quality/speed runs; Hunyuan also preflights
            flash_attn_varlen_func when dense warmup is enabled. Strict preflight
            also imports the SparseVideo-owned SVG1 method and Triton placement
            modules, checking profiling, block-mask, head placement/restore,
            and placement-kernel APIs before quality/speed runs.
svg2        Adapter port. Uses upstream k-means/top-p names/defaults; executes
            through method-owned Sparse-VideoGen SAP Triton k-means and
            FlashInfer block-sparse kernels.
            Strict preflight imports FlashInfer and the owned Sparse-VideoGen
            Triton/helper modules, then checks the k-means, dynamic-map,
            permutation, block-sparse, and variable-block FlashInfer APIs
            needed by the sparse path before quality/speed runs.
            Hunyuan follows upstream SAP text handling by clustering video
            tokens only, then appending prompt and unused-prompt clusters to
            the dynamic map. Hunyuan dense warmup uses the upstream FlashInfer
            two-segment varlen path, not FlashAttention varlen.
spargeattn  Kernel wrapper. Defaults to upstream's recommended plug-and-play
            spas_sage2_attn_meansim_topk_cuda path with topk=0.5. Wan2.1 and
            Hunyuan dense baselines use method=dense; SpargeAttn no longer
            exposes a method-local full/dense mode. Preserves sparse
            upstream mode/value example-script API and direct kernel names:
            topk, cdfthreshd, simthreshd1, pvthreshd,
            attention_sink, smooth_k, scale, tensor_layout, output_dtype, and
            block_sparse_sage2_attn_cuda mask_id. Non-full modes need
            spas_sage_attn with _qattn/_fused extensions. SparseVideo owns
            the upstream C++/CUDA/Python source under
            src/sparsevideo/kernels/native/spargeattn and prefers an in-place
            build from that directory. Runtime packages resolved from
            training_free/ are rejected because that tree is reference-only.
            tune/parallel_tune/l1/pv_l1/tune_pv/model_out_path route through
            the owned SparseAttentionMeansim autotune path and save/load tuned
            state with SparseVideo model paths. Defaults follow the upstream
            SparseAttentionMeansim/video wrapper values l1=0.07, pv_l1=0.08,
            tune_pv=true. Sparse modes reject unsupported runtime conditions
            instead of silently returning dense attention. Sparse/tuned modes
            preflight loadability and required spas_sage_attn APIs from the owned runtime
            before model load; extension/source presence alone is not accepted.
            Wan full mode follows the
            upstream dispatch path, while Hunyuan full mode follows upstream
            SDPA on [B,H,N,D]. SpargeAttn intentionally keeps stock Diffusers
            QK norm/RoPE instead of the SVG/SVOO fused norm/RoPE kernels,
            because the upstream SpargeAttn video wrappers only replace the
            attention call.
radial      Adapter port. Uses upstream decay_factor/block_size names and
            model-aware inference-shell defaults. Dense warmup is controlled
            only by the common dense_warmup_* ratios. FlashInfer path uses the
            upstream shrinkMaskStrict BSR
            mask construction. Missing FlashInfer support is a hard error.
            use_sage_attention routes the sparse stage through
            SparseVideo-owned spas_sage_attn
            block_sparse_sage2_attn_cuda and the upstream sparge_mask_convert
            layout conversion. It requires upstream-compatible
            video_len % block_size == 0 shapes and the local _qattn/_fused
            extensions under src/sparsevideo/kernels/native/spargeattn.
            FlashInfer mode preflights loadability and the top-level
            BlockSparseAttentionWrapper/single-prefill/merge-state APIs.
            Strict preflight also imports the SparseVideo-owned radial method
            module and FlashInfer BSR wrapper, checking shrinkMaskStrict-style
            BSR mask construction, radial FlashInfer/Sage dispatch helpers,
            Sage mask conversion helpers, and build_bsr_from_mask/
            bsr_sparse_attn. Source-file presence alone is not accepted.
            use_sage_attention preflights loadability and required
            spas_sage_attn/SageAttention APIs from the owned runtimes; native
            extension presence alone is not accepted.
            Dense warmup uses the upstream FlashInfer all-mask path when
            use_sage_attention=false; use_sage_attention dense warmup uses the
            SparseVideo-owned SageAttention runtime under
            src/sparsevideo/kernels/native/sageattention.
sta         Adapter port with SparseVideo-owned FastVideo STA wrapper. Sparse
            inference requires CUDA. A100 runs use the SparseVideo-owned SM80
            block-sparse CUDA backend under src/sparsevideo/kernels/native/draft_block_sparse
            with FastVideo's STA tile-window mask.
            Strict preflight checks the A100 block-sparse CUDA backend on Ampere.
            FastVideo native shapes include 18x48x80, 30x48x80, and 36x48x48;
            other 720p layouts are rejected instead of silently using the
            non-upstream generalized STA kernel.
draft       Adapter port with upstream reorg/restore indices, head-global
            percentile mask semantics, common dense warmup gates, and model-aware
            sparsity_ratio defaults. Upstream layout names latent_h, latent_w,
            visual_len, text_len, and batch_size are exposed and checked when
            set. block_sparse_attention=False disables the sparse path upstream,
            so SparseVideo rejects it; use dense for the baseline. Upstream
            dense warmup uses Draft's flash_attn_varlen_func path when available.
            Upstream sparse layouts are narrow: Wan supports 21x32x48 and
            21x48x80 latent layouts; Hunyuan supports 33x48x80. Other layouts
            are parity gaps, not successful upstream-equivalent runs. Strict
            preflight requires flash_attn_varlen_func for dense warmup
            instead of silently falling back to SDPA, and imports the
            owned MIT Block-Sparse-Attention runtime to check block_sparse_attn_func
            plus block_sparse_attn_cuda fwd/bwd ops before benchmark runs.
adacluster  Adapter port. Uses upstream topk_num/q_kernel_num/kv_kernel_num
            names; current clustering runs through SparseVideo-owned Triton
            k-means and block-sparse attention kernels. Hunyuan follows the
            upstream topk_from_qkv_minmax sparse mask policy; the old
            method-local first-eight-step/layer dense gates are disabled so
            dense dispatch stays controlled by the common dense warmup ratios.
            Hunyuan reinitializes Q and K centroids
            on later sparse calls, matching the upstream processor. Wan follows upstream
            thresholded_kmeans_loop defaults and keeps the fixed cluster-count
            path available with the original parameter names. Strict preflight
            imports the owned AdaCluster Triton modules and checks
            flash_kmeans_single, triton_cluster_sparse_attn, and their JIT
            kernels before benchmark runs.
flashomni   Kernel adapter. FlashOmni itself is a sparse kernel interface; an
            upstream-equivalent run uses sparse_pattern=explicit with caller-
            provided sparse_info, sparse_kv_info, sparse_info_indptr, and
            sparse_kv_info_indptr tensors.
            sparse_pattern=global_random keeps FlashOmni's upstream synthetic
            kernel benchmark names sparse_size/spq_Q/spq_KV/text_token, but is
            not a video quality-parity sparsity policy.
            sparse_pattern=paper_mmdit is SparseVideo-owned development code
            derived from the public paper and benchmark/test_attn_score.py
            score-CDF sparse-info mechanics. It uses the paper's configuration
            names tau_q/tau_kv/N/D/S_q, refreshes sparse symbols at update
            steps, reuses cached attention outputs at dispatch steps, and can
            route Q/O projections through owned FlashOmni GEMM-Q/GEMM-O hooks
            with use_sparse_gemm=true and sparse_block_size_for_q=128. This is
            still not code-level upstream parity because no upstream video
            sparse-symbol policy is available in the reference checkout.
            FlashOmni wrapper plan names causal, pos_encoding_mode,
            use_fp16_qk_reduction, logits_soft_cap, sm_scale, rope_scale, and
            rope_theta are exposed and forwarded to the native wrapper.
            The explicit path accepts either upstream unpacked 0/1 sparse-info
            tensors with logical indptr or already packed tensors with the
            indptr returned by flashomni.segment_packbits.
            sparse_pattern=local_qk_topk keeps the old block-mean top-k path
            only as a SparseVideo diagnostic policy, not method parity.
            implementation=upstream calls FlashOmni's
            BatchFlashOmniFAWithRaggedKVWrapper and requires its CUDA/C++ ops;
            attention_mask is routed through FlashOmni custom_mask for the
            explicit/full/global_random upstream paths.
            SparseVideo owns a FlashOmni runtime/source copy under
            src/sparsevideo/kernels/native/flashomni and requires it for
            upstream/runtime parity once the local flashomni_kernels extension
            is built. Strict preflight imports that owned runtime and checks
            BatchFlashOmniFAWithRaggedKVWrapper, segment_packbits, and the
            batch_sparseFA plan/run torch ops before benchmark runs; native
            extension/source presence alone is not accepted.
            Environment flashomni packages are not accepted for SparseVideo
            runtime parity.
svoo        SparseVideo-owned native/Triton path by default. The old
            implementation=upstream bridge is disabled because runtime code must
            not import from training_free. It uses local co-clustering helpers,
            local Triton kernels, copied local sparsity profiles, and optional
            installed FlashInfer for block-sparse execution.
            Hunyuan text/padding clusters follow upstream when prompt_length is
            supplied, and Hunyuan dense warmup uses the upstream FlashInfer
            two-segment varlen path; scripts/infer.py computes prompt_length
            from the Hunyuan tokenizer.
            Wan follows upstream's split fast path: Triton QK norm is enabled,
            but RoPE stays on the stock PyTorch path rather than the Hunyuan
            native fused RoPE path.
            Clustering reuse starts as soon as a matching clustering result is
            cached; start_reuse_step is no longer exposed.
            SVOO_ENABLE_MEM_SAVE/enable_mem_save follows upstream's default
            early release of large sparse-attention intermediates. Measurement
            and global constraints are ported with owned SparseVideo code/kernels.
            FlashInfer sparse
            backend preflights loadability and the sparse VariableBlock wrapper
            APIs needed by the upstream path; strict preflight also imports the
            owned Triton/helper modules and checks co-cluster, norm, modulation,
            permutation, block-sparse, FlashInfer, and sparsity profiler APIs.
```

This table is intentionally conservative. "Adapter port" means the SparseVideo
method is wired into Diffusers-style processors with upstream parameter names,
but it is not a line-for-line vendor of the upstream repository.
