# SparseVideo All-Backbone All-Method Goal

Goal: make SparseVideo a plug-and-play sparse-attention layer for Diffusers video DiT pipelines across every supported softmax video-attention backbone. Use one API, `sparsevideo.apply(pipe, method=...)`; keep `dense` as the baseline and guarantee restore-to-dense.

Current status: Wan, HunyuanVideo, SkyReels, CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate have processor wiring. The newly added backbones now have real smoke coverage for `dense` plus all public sparse methods: `svg1`, `svg2`, `spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, and `svoo`. The reduced-surface method guards for these new backbones have been removed.

Public methods: `svg1`, `svg2`, `spargeattn`, `radial`, `sta`, `draft`, `adacluster`, `flashomni`, `svoo`, plus `dense`.

## Target Backbones

Already broad-supported, preserve behavior and avoid regressions:
- `wan21-t2v-1.3b`, `wan21-t2v-14b`, `wan22-t2v-a14b`
- `wan21-i2v-14b`, `wan22-i2v-a14b`
- `wan22-animate-14b`
- `wan21-vace-1.3b`, `wan21-vace-14b`
- `skyreels-v2-t2v-14b`, `skyreels-v2-i2v-14b`
- `hunyuan-t2v`, `hunyuan-i2v`

New backbones that must be upgraded from `dense/svg2/svoo` to all public sparse methods:
- `cogvideox-t2v`
- `cogvideox-i2v`
- `ltx-video`
- `ltx-video-i2v`
- `allegro`
- `mochi-1`
- `easyanimate-v5-t2v-12b`

Non-targets unless the architecture changes:
- `sana-video`: incompatible with the current softmax sparse-attention methods because it uses linear attention.
- `kandinsky5-t2v`: native-N/A because it already exposes native sparse parameters instead of a processor-swap surface.
- `motif-video` and `ltx-video-2`: unknown until their current Diffusers attention ownership and token layout are verified.

## Required Method Expansion

For CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate, support all of:
- `svg1`
- `svg2`
- `spargeattn`
- `radial`
- `sta`
- `draft`
- `adacluster`
- `flashomni`
- `svoo`

Do not count `dense` as sparse support. Do not claim a method is supported on a backbone until `sparsevideo.apply()` installs the correct processor or model patch, sparse dispatch is observed, and `restore()` returns the original dense path.

Current new-backbone evidence files:
- `svg1`: `.tmp_smoke/metrics_svg1_new_backbones_recheck.jsonl`
- `spargeattn`: `.tmp_smoke/metrics_spargeattn_new_backbones_recheck2.jsonl`
- `adacluster`: `.tmp_smoke/metrics_adacluster_new_backbones_recheck.jsonl`
- `radial`: `.tmp_smoke/metrics_radial_new_backbones_recheck.jsonl`, `.tmp_smoke/metrics_radial_new_backbones_recheck2.jsonl`, `.tmp_smoke/metrics_radial_cog_i2v_125f.jsonl`
- `sta`: `.tmp_smoke/metrics_sta_new_backbones_ok.jsonl`
- `draft`: `.tmp_smoke/metrics_draft_new_backbones.jsonl`
- `flashomni`: `.tmp_smoke/metrics_flashomni_new_backbones.jsonl`

## Architecture Rules

- Separate self-attn plus cross-attn: Wan family, SkyReels, LTX Video, Allegro. Apply sparse methods to video self-attn only; keep cross-attn dense.
- Concatenated text/video sequence: CogVideoX and EasyAnimate. Split text and video tokens, preserve text and text-video interactions as dense where the method cannot own them safely, apply sparse dispatch only to the video-video block, then recombine without changing output contracts.
- Joint dual-stream attention: HunyuanVideo and Mochi. Preserve the text stream and sparse only the video-video path unless the upstream method explicitly defines a safe joint-stream policy.
- Model-specific token counts, frame/height/width layout, masks, and timestep gates must come from the pipeline state or validated method config. No hard-coded shape hacks for a single smoke command.

## Method-Specific Requirements

- `svg1` and `svg2`: match the Sparse-VideoGen reference behavior for timestep warmup, token policy, clustering, permutation, and restoration. Config names must stay compatible with the current public registry.
- `spargeattn`: support every target backbone or fail preflight with a precise architectural reason. It must own Q/K/V and sparse dispatch, not only wrap a dense attention call.
- `radial`: implement the actual radial token layout/mask behavior for each target token layout. Dense masked approximation is not enough.
- `sta`: provide model-aware spatial/temporal layout, mask strategy, and available native backend dispatch. A100/current-hardware support is required; H100/TK artifacts are required only when Hopper hardware is available.
- `draft`: implement the upstream reorder, draft selection, and restore path end to end for the target sequence layout.
- `adacluster`: implement adaptive clustering against the model's video-token layout and preserve output ordering.
- `flashomni`: become turnkey for supported backbones by generating or sourcing the required sparse-info tensors in the method path. A caller-only missing-tensor path is not support.
- `svoo`: keep the existing smoke-passing behavior and extend any model-specific warmup/config gaps discovered while enabling the other methods.

## Implementation Constraints

- `training_free/*` is the reference for method behavior. Compare execution paths, not only method names or formulas.
- Keep `LIMITED_METHODS_BY_MODEL_TYPE` or equivalent guards until each model-method pair has evidence. Removing guards is the last step, not the first.
- Keep one public API and the existing registry/config flow. Do not add per-model public method names.
- Optional native kernels must not be required for basic import. CPU or dense fallbacks may exist for tests, but benchmark/support claims require real sparse backend dispatch.
- Touch only the files needed for detection, processors, method behavior, preflight, inference aliases, tests, and audit gates.

## Verification Gate

The goal is complete only when all required target model-method pairs satisfy these checks:
- `sparsevideo.apply(pipe, method=...)` accepts the method for the model without unsupported-method guards.
- The method installs the expected processor or model patch and `restore()` returns the dense processor/path.
- A real smoke run records `status="ok"`, non-empty `backend_counts`, and `dispatch_counts.sparse > 0` for sparse methods.
- The audit gate requires every target backbone to have `dense` plus all public sparse methods in smoke evidence, not only `svg2/svoo`.
- Existing Wan, HunyuanVideo, SkyReels, CogVideoX, LTX Video, Allegro, Mochi, and EasyAnimate smoke evidence must not regress.
- Method-level quality/strict-dispatch gates remain intact before making performance or parity claims.

## Priority

1. Build a model-method gap matrix from the current guards and audit evidence.
2. Port one method at a time across the seven reduced-surface backbones, starting with the least model-specific path.
3. Add focused unit tests for token splitting, layout restoration, processor restore, preflight, and config acceptance.
4. Run multi-GPU smoke experiments in parallel when GPUs are idle, with one large model run per GPU when resource-safe.
5. Update `scripts/audit_parity.py` so incomplete model-method pairs fail the all-backbone gate until evidence exists.
