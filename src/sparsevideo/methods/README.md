# Method Packages

Each sparse attention method owns a package under `sparsevideo.methods`.

Required layout:

```text
methods/<name>/
  __init__.py   # lazy public exports only
  method.py     # SparseMethod adapter and processor wiring
```

Complex methods should split local concerns instead of growing `method.py`:

```text
methods/<name>/
  config.py     # upstream names, defaults, aliases, unsupported flags
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

Current status:

```text
dense       Baseline; original processor path.
svg1        Adapter port. Uses upstream names/defaults; mask construction is
            implemented locally and should still be validated against upstream.
svg2        Adapter port. Uses upstream k-means/top-p names/defaults; executes
            through SparseVideo shared k-means and block-sparse kernels.
spargeattn  Kernel wrapper. Uses upstream mode/value API; non-full modes need
            the optional spas_sage_attn package. Upstream tuning flags are
            recognized and rejected until that path is ported.
radial      Adapter port. Uses upstream dense_layers/dense_timesteps/
            decay_factor names; use_sage_attention is recognized but unported.
sta         Adapter port with fastvideo_kernel path when installed; otherwise
            uses SparseVideo Triton CUDA fallback. CPU fallback is disabled for
            fair benchmarking. The upstream native FastVideo STA path is used
            only for supported seq_shape values such as 18x48x80, 30x48x80,
            and 36x48x48; other 720p layouts use the Triton path.
draft       Adapter port. Uses upstream pool_h/pool_w/sparsity_ratio names.
            block_sparse_attention=False disables the sparse path upstream, so
            SparseVideo rejects it; use dense for the baseline.
adacluster  Adapter port. Uses upstream topk_num/q_kernel_num/kv_kernel_num
            names, including Hunyuan late-layer overrides; current clustering
            runs through shared SparseVideo kernels.
flashomni   Kernel adapter. FlashOmni itself is a sparse kernel interface; the
            local sparse_kv_budget policy is SparseVideo adapter policy, not an
            upstream video method policy. implementation=upstream calls
            FlashOmni's BatchFlashOmniFAWithRaggedKVWrapper and requires its
            CUDA/C++ ops; implementation=flex is an explicit slow fallback.
            Sparse-info tensors are recognized but not wired through the CLI/API
            yet.
svoo        SparseVideo-owned native/Triton path by default. The old
            implementation=upstream bridge is disabled because runtime code must
            not import from training_free. It uses local co-clustering helpers,
            local Triton kernels, copied local sparsity profiles, and optional
            installed FlashInfer for block-sparse execution.
            Hunyuan text/padding clusters follow upstream when prompt_length is
            supplied; scripts/infer.py computes it from the Hunyuan tokenizer.
            Measurement and global/routing experimental branches are recognized
            but unported.
```

This table is intentionally conservative. "Adapter port" means the SparseVideo
method is wired into Diffusers-style processors with upstream parameter names,
but it is not a line-for-line vendor of the upstream repository.
