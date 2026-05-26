## Golden Rules
### Rule 1 - Think Before Coding.
No silent assumptions. State what you're assuming. Surface tradeoffs. Ask before guessing. Push back when a simpler approach exists.
### Rule 2 - Simplicity First.
Minimum code that solves the problem. No speculative features. No abstractions for single-use code. If a senior engineer would call it overcomplicated — simplify.
### Rule 3 - Surgical Changes.
Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Don't refactor what isn't broken. Match existing style.
### Rule 4 - Goal-Driven Execution.
Define success criteria. Loop until verified. Don't tell Claude what steps to follow, tell it what success looks like and let it iterate.
### Rule 5 - Auto-Review After Changes.
After making any modification, review the changed code before finalizing. Check for correctness, unintended side effects, style consistency, and whether the change still satisfies the original goal.


# SparseVideo Project Instructions

## Target: Same Upstream Implementation, Including Native Kernels

SparseVideo's target is to provide the same inference implementation as each referenced `training_free/` method, moved
into owned SparseVideo code and made plug-and-play for real video diffusion pipelines.

This means the project target is not just "the API can run". The target is:

- same algorithmic path as upstream;
- same public method/config semantics as upstream;
- same native kernel path as upstream when upstream uses one;
- same real pipeline behavior as upstream, with `dense` as the baseline and restore path.

SparseVideo is not a demo, not a method-name registry, and not a Python approximation layer. The public API should let
users switch `dense` <-> sparse methods and restore dense attention, but a method is complete only when the
SparseVideo-owned runtime reproduces the referenced `training_free/` inference path.

## Hard Kernel Requirement

Native kernels are part of the target implementation, not optional accelerators. If upstream uses Triton, C++/CUDA,
FlashInfer, SageAttention, fused ops, sparse plan/run kernels, clustering kernels, or any other custom backend in the
intended inference path, SparseVideo must own, build or load, preflight, and dispatch an equivalent implementation under
`src/sparsevideo/`. Without that backend, the method is `partial` or `not benchmark-ready`, even if import works, unit
tests pass, or an `.mp4` is produced.

Missing or unused kernels are a correctness and completion problem, not just a performance problem. A method that falls
back to dense attention, plain PyTorch, Python mask construction, or a different backend than upstream must be labeled as
debug/partial unless upstream itself uses that exact path for the reported benchmark.

Inference and benchmark scripts must prove native backend status. They should record which Triton/C++/CUDA/FlashInfer/
SageAttention/custom kernels were loaded, which path was selected, and fail or report `partial`/`not benchmark-ready`
when the intended kernel path is unavailable, uncompiled, disabled, or unused.

A Python/PyTorch approximation, dense fallback, or runtime dependency on `training_free/` is allowed only for explicitly
labeled import/debug/correctness paths. It cannot be used for quality claims, speed claims, or completion claims unless
the upstream benchmark uses the same path.

A method is in target only if all of this is true:

1. Upstream parity: owned code under `src/sparsevideo/` matches the `training_free/` execution path for token layout,
   sparse timestep/layer gates, masks, clusters, sparse maps, permutation/restore logic, dense fallback behavior under
   the shared warmup policy, and output behavior.
2. Config parity: public parameter names, meanings, defaults, and benchmark profiles match upstream. Local aliases can
   exist, but upstream names stay primary.
3. Kernel parity: every upstream-required Triton/C++/CUDA/FlashInfer/SageAttention/custom kernel path is available from
   owned SparseVideo code, passes preflight, and is observed in runtime dispatch.
4. Pipeline parity: real Diffusers/DiffSynth-style video pipelines can apply sparse attention, compare against `dense`,
   and restore the original dense path.
5. Evidence parity: matched real inference records show quality, speed, memory, resolved config, backend status, output
   paths, and restore behavior against `dense`.

`training_free/` is reference material only. Do not modify it for SparseVideo behavior and do not rely on it at runtime.
Required runtime code, helper logic, sparsity profiles, and kernels must live under `src/sparsevideo/`. Method-specific
kernels should live with that method or in a clearly owned method kernel subtree. Shared kernels are acceptable only for
genuinely reusable primitives with stable interfaces.

The only acceptable missing-kernel cases are basic import, CPU correctness tests, and debug smoke tests. Quality runs,
speed runs, and benchmark scripts must use the required native backend, or fail preflight/report `partial` or
`not benchmark-ready`. Silent fallback to dense, Python, or a different backend is a correctness failure.

Hardware-specific kernel paths that cannot be executed on the visible machine should be marked `hardware-deferred`, not
`pass`. For example, H100/TK STA dispatch is deferred on an A100-only machine. Deferred hardware paths do not block the
current-hardware audit, but they must stay separate from full upstream parity and become required again when matching
hardware is available.

SparseVideo is out of target if it is only a wrapper around `training_free/`, a toy tensor benchmark, a method-name
registry with approximate implementations, a Python-only rewrite of native-kernel methods, or a benchmark script that
claims speed or quality while falling back from the upstream sparse path.

## Non-Negotiable Completion Gates

A method cannot be called complete, benchmark-ready, or upstream-equivalent until all gates below are checked and
recorded:

- Upstream parity gate: the real execution path is audited against `training_free/` for token layout, sparse timestep/layer
  gates, masks, clusters, sparse maps, permutation/restore logic, text/video handling, dense fallback behavior under the
  shared warmup policy, and output behavior.
- Config parity gate: resolved public parameters, meanings, defaults, and benchmark profiles match the upstream
  scripts/configs used for reported quality and speed.
- Kernel parity gate: every upstream-required Triton/C++/CUDA/FlashInfer/SageAttention/custom kernel path has an owned
  SparseVideo implementation, builds or loads successfully, passes preflight, and is observed in runtime dispatch.
- No-silent-fallback gate: benchmark or quality runs must fail, or explicitly report `partial`/`not benchmark-ready`, if
  the intended native sparse backend is missing, uncompiled, disabled, unused, or replaced by dense/Python fallback.
- Real inference gate: matched dense-vs-sparse generation records include quality, speed, memory, output paths, resolved
  config, runtime backend status, and restore-path validation.

## Success Criteria

- One small public API for applying and restoring sparse attention.
- Unified method registration and config handling.
- Support real video diffusion pipelines, not only toy tensors.
- Correctness tests before performance claims.
- Native/Triton/C++/CUDA/FlashInfer/SageAttention/custom kernels are mandatory for parity/performance when upstream uses them; they are optional only for basic import/debug paths.
- New methods should be easy to add without model-specific hacks.
- The project should remain lightweight and pip-package friendly.

## Included Training-Free Methods

Use these public names:

1. `svg1`: `training_free/Sparse-VideoGen` SVG method.
2. `svg2`: `training_free/Sparse-VideoGen` second method.
3. `spargeattn`: `training_free/SpargeAttn`.
4. `radial`: `training_free/radial-attention`.
5. `sta`: `training_free/FastVideo` Sliding Tile Attention.
6. `draft`: `training_free/draft-attention`.
7. `adacluster`: `training_free/Adacluster`.
8. `flashomni`: `training_free/FlashOmni`.
9. `svoo`: `training_free/SVOO`.

Always keep `dense` as the baseline and fallback.

## Dense Warmup Policy

Dense/full-attention dispatch is intentionally normalized across SparseVideo methods. The only public controls for
method-internal dense warmup are `dense_warmup_step_ratio` and `dense_warmup_layer_ratio`, implemented through
`src/sparsevideo/methods/_schedule.py`. Use `method=dense` for the full dense baseline.

Do not add or restore method-local dense/full routing knobs or hardcoded upstream dense gates such as per-method
`full`/`is_full` modes, `first_*` warmup keys, fixed first-N-step dense branches, fixed layer lists, or
`skip_time_steps`-style runtime dense dispatch. Upstream sparse schedules, masks, clustering, kernels, and layouts still
need parity, but upstream-specific dense/full fallback gates should be expressed through the shared warmup ratios or left
out of runtime dispatch.

## Current Audit State

Before reworking a method, run `scripts/audit_parity.py --format markdown` and read `PARITY_STATUS.md`. As of
2026-05-18, `dense`, `svg1`, `svg2`, `spargeattn`, `radial`, `draft`, `adacluster`, `svoo`, and `sta` on A100 pass the
current audit and should not be repeatedly rewritten without new evidence.

The remaining software gap is `flashomni`: SparseVideo owns the FlashOmni C++/CUDA runtime, attention adapter,
GEMM-Q/GEMM-O path, paper/benchmark score-CDF `paper_mmdit` policy, tests, and a 50-step strict-dispatch run. It is
still not code-level upstream Wan/Hunyuan video-method parity because the public upstream reference exposes only the
engine/runtime and benchmark helpers, not a Wan/Hunyuan video sparse-symbol policy. The public paper also says
`tau_q`/`tau_kv` progressively converge during denoising, but no public code-level schedule formula is available; the
current `paper_mmdit` path uses the reported target values directly. The audit also checks the local FlashOmni git
history for hidden policy filenames; avoid repeating that source archaeology unless new upstream refs appear. Do not
claim `flashomni` complete unless concrete upstream video-policy source/evidence appears and the owned policy is
compared against it.

`sta_h100` is hardware-deferred on this A100-only machine. A100 STA evidence does not prove Hopper/TK dispatch.

## Engineering Rules

- Think before coding. State assumptions and ask when guessing could cause damage.
- Keep changes surgical. Touch only files needed for the task.
- Prefer simple, explicit code over speculative abstractions.
- Preserve existing style unless changing it is necessary.
- Do not install old upstream method requirements into the main environment blindly.
- Treat `training_free/` repositories as references unless a task explicitly says to run them.
- Verify the result with the narrowest useful command or test.

## Implementation Fidelity Rules

- The goal is the same implementation as the referenced method, not a similar-looking approximation.
- For each method, compare the real execution path against upstream before claiming parity: sparse timestep/layer gating, token layout, mask generation, clustering, dynamic maps, permutation/restore, dense fallback behavior under the shared warmup policy, and backend/kernel calls.
- Keep upstream public parameter names, meanings, and comparison defaults. Distinguish parser defaults from the shell/config defaults actually used for reported quality/speed.
- Do not simplify, omit, or rename method options unless the change is documented as a compatibility alias and the upstream name remains primary.
- If a method is only partially ported or uses a fallback/approximation, say so explicitly in code comments, docs, runtime warnings, or review notes. Do not let the registry name imply full parity.

## Method Config Rules

- `default_method_config()` must support model-aware defaults through `model_family` and `model_key` when upstream uses different quality/speed settings per model or resolution.
- Method configs may expose only the shared `dense_warmup_step_ratio` and `dense_warmup_layer_ratio` controls for
  method-internal dense warmup. Do not expose upstream method-local dense/full switches as runtime API.
- For quality and speed comparisons, copy defaults from the upstream inference shell/config actually used for that benchmark, not from weak parser defaults.
- Every method default alignment needs a dry-run test that checks the resolved config emitted by `scripts/infer.py`.
- If a config value is chosen for safety, memory, or local hardware rather than upstream parity, name it as a local override and do not present it as the upstream default.

## Reference Code And Kernels

- Do not modify `training_free/` for SparseVideo runtime behavior. It is reference material only.
- If runtime code, CUDA/C++ kernels, sparsity profiles, or helper logic are needed, copy or reimplement them under `src/sparsevideo/` with a clear method-owned or shared location.
- Kernel coverage is not optional for the target implementation: if upstream uses Triton, C++/CUDA, FlashInfer, SageAttention, fused ops, clustering kernels, or custom sparse kernels in its intended inference path, SparseVideo parity requires an equivalent owned backend or an explicitly documented unimplemented gap.
- A method is not complete while its native kernel path is missing, uncompiled, unused, or replaced by a Python/PyTorch approximation.
- Python/PyTorch fallback paths must not be used for speedup claims unless the upstream benchmark also uses that path.
- Basic `import sparsevideo` must work without native kernels; serious inference and benchmark runs must report native/Triton/Python/dense-fallback backend status.
- Prefer method-owned kernels when the implementation is method-specific. Use shared kernels only for genuinely reusable primitives with a stable interface.
- Preflight sparse runs should fail or clearly warn when the intended package/native kernel path is unavailable for a benchmark.

## Inference And Benchmark Rules

- Quality and speed comparisons must use the same model, prompt, seed, resolution, frame count, scheduler, VAE dtype, guidance settings, and number of inference steps for dense and sparse.
- Smoke tests with tiny `num_inference_steps` are only for checking that inference runs. They are not quality evidence and must not be used for speedup claims.
- For Wan 720p quality runs, use 81 frames, 16 fps, normal 50 steps, fp32 Wan VAE, and `flow_shift=5.0` unless explicitly testing another setting.
- For Hunyuan quality runs, use 129 frames by default and normal comparison steps unless the task says otherwise.
- Always keep `dense` as the baseline and run it through the same pipeline settings as sparse methods.
- Inspect generated artifacts when debugging quality. A green JSON status or passing unit test is not enough to claim visual quality.
- Record output paths, resolved method config, runtime kernel status, generation time, peak CUDA memory, and exact command or script used.
- On multi-GPU machines, run independent inference/quality/speed experiments concurrently whenever practical, with at most one large model run per GPU unless explicitly testing multi-process sharing.

## Script Expectations

- Keep inference entrypoints minimal. Prefer one clear script with `--model`, `--method`, and `--method-config KEY=VALUE` over many per-method scripts.
- Scripts should list supported methods and important defaults in comments or `--help`.
- Public names must be `svg1` and `svg2`; do not expose old ambiguous names like `svg` or `sap` as the primary API.

## Environment

```text
Python: /home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
Models: /home/dataset-assist-0/luojy/models
```
