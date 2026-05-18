# SparseVideo Parity Status

Last audited: 2026-05-18

Objective: SparseVideo methods must use the same inference implementation as
their referenced `training_free/` methods, including owned native kernels and
real video pipeline evidence. Passing tests or producing an mp4 is not enough
unless the matching metrics show strict sparse/native backend dispatch.

Run the current audit with:

```bash
/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python scripts/audit_parity.py --format markdown --fail-on-incomplete
```

Current status: incomplete.

The only current software blocker is `flashomni` video method parity. The
`sta_h100` Hopper/TK path is hardware-deferred because this machine has no
H100/Hopper GPU.

## Passed

Do not rework these unless new evidence changes the audit result:

- `dense`
- `svg1`
- `svg2`
- `spargeattn`
- `radial`
- `draft`
- `adacluster`
- `svoo`
- `sta` on A100 (`fastvideo_sta_a100_triton`)

These methods have current 50-step quality records plus strict backend dispatch
evidence, or dense baseline evidence for `dense`.

## Still Partial

### `flashomni`

SparseVideo-owned FlashOmni C++/CUDA runtime imports and explicit sparse-info
dispatch has been observed. This proves the kernel adapter path only.

It is still not complete because the local `training_free/FlashOmni` reference
publishes the FlashOmni engine/API and synthetic benchmark helper
`get_qkvo_global_sparse`, plus the benchmark score-CDF sparse-info fill helpers
in `benchmark/test_attn_score.py`, but not a reusable Wan/Hunyuan video
sparse-info or sparse-symbol generation policy. Do not treat `global_random`
or ad-hoc explicit sparse-info tensors as video method parity.

The owned method path now includes the paper-style update/dispatch state and
can route Q/O projections through `flashomni_gemm` /
`flashomni_gemm_reduction` when `sparse_pattern=paper_mmdit` and
`use_sparse_gemm=true`. The sparse-symbol policy now follows the public paper's
feature-cache contribution/guidance rule plus `benchmark/test_attn_score.py`
score-CDF sparse-info mechanics. It uses the reported target `tau_q`/`tau_kv`
values directly; the public paper also says these thresholds progressively
converge during denoising, but it does not publish the code-level schedule
formula. This narrows the remaining gap to code-level upstream Wan/Hunyuan
policy evidence; it is still not a complete upstream video-method port because
that policy and schedule are not available in the local reference checkout.

External source status, rechecked on 2026-05-18: the OpenReview submission advertises
`https://anonymous.4open.science/r/FlashOmni-B980`, but that endpoint currently
returns `{"error":"not_connected"}` here. The OpenReview API exposes only the
main PDF and public review discussion, not source code or supplementary files.
Those public notes reinforce that the HunyuanVideo method depends on fixed
hyperparameters, a feature-caching plus block-sparse skipping schedule,
sparse-symbol state, and the ordered `GEMM-Q -> Attention -> GEMM-O` path. They
do not provide enough implementation detail to claim code-level upstream parity.
The author's public project page links the FlashOmni code to
`https://github.com/qiaolian9/FlashOmni`. The public GitHub page currently
shows only the engine/runtime directories `3rdparty`, `aot_build_utils`,
`benchmark`, `csrc`, `flashomni`, and `include/flashomni`, plus top-level
build/readme files. It does not expose Wan/Hunyuan inference scripts or a video
sparse-symbol policy beyond the benchmark helpers already mirrored in
`training_free/FlashOmni`.

The local `training_free/FlashOmni` git checkout was also inspected on
2026-05-18. It is at commit `2f27ee944202bf5d625fc28319c8198bf5ef7653`, with
branches `main`, `origin/HEAD`, and `origin/main`, no tags, and no git-history
file-name candidates matching Wan/Hunyuan/policy/infer/pipeline/video sparse
symbol or sparse-info terms. This makes the remaining blocker an unavailable
upstream artifact, not a missed local checkout file.

The public GitHub repository homepage shows one issue, but the `/issues` page
and the GitHub issues API returned 404 without accessible issue content on
2026-05-18. No public issue or PR content was available as a substitute source
for the missing video sparse-symbol policy.

The OpenReview forum was also checked through the public API on 2026-05-18.
Author rebuttal comments reference paths such as
`./FlashOmni/example/hunyuan/nvprof`, `./FlashOmni/example/hunyuan/nvprof/e2e`,
and `./FlashOmni/benchmark/nvprof_attn/hunyuan`. Those paths are not present
in the public GitHub repository or in the local `training_free/FlashOmni`
checkout. This is useful negative evidence: the public discussion points to
additional experiment artifacts, but those artifacts are not available to port
or compare against.

The OpenReview revised PDF was also text-checked on 2026-05-18. It adds a
qualitative statement that sparsity is kept low early and gradually increased
later, and it repeats the `tau_q`/`tau_kv` convergence note. It still does not
provide a Wan/Hunyuan policy source, example files, or a concrete convergence
schedule formula.

The arXiv e-print source for `2509.25401` was checked on 2026-05-18. It
contains TeX source, figures, table text, and algorithm listings, but no
code/example directory and no Wan/Hunyuan video sparse-symbol policy source.
`sections/Appendix.tex` repeats that `tau_q` and `tau_kv` progressively
converge to their target values during denoising, but it does not provide the
convergence schedule formula or implementation. This keeps the schedule as a
real missing upstream artifact rather than an implementation detail we can
faithfully copy.

Current owned progress: `src/sparsevideo/methods/flashomni/policy.py` provides
a paper/benchmark-derived `paper_mmdit` sparse-info policy using the paper
configuration names `tau_q`, `tau_kv`, `N`, `D`, and `S_q`. It builds
compressed Q/K block scores, refreshes sparse symbols on update steps, applies
the paper's feature-cache contribution/guidance selection rule, applies the
score-CDF kv-skip mechanics from `training_free/FlashOmni/benchmark/test_attn_score.py`,
and uses cached attention-output reuse on dispatch steps so feature-cache
`sparse_q` symbols are not treated as plain dropped output blocks. The
processor path can also use owned FlashOmni GEMM-Q and GEMM-O hooks to skip
query projection work on dispatch steps and split output projection into
cached-bias plus active-reduction work. This is useful development code and has
CUDA smoke tests plus a current-policy 50-step paper-policy run through the
owned FlashOmni kernels, but it is not code-level upstream Wan/Hunyuan repo
parity. The exact upstream `tau_q`/`tau_kv` convergence schedule remains part
of the missing video policy evidence.

Current Hunyuan reported-config evidence on 2026-05-18:
`result/inference/audit/flashomni_paper_mmdit_hunyuan_current_policy_20260518/metrics.jsonl`
ran Hunyuan at 720x1280, 129 frames, 50 denoise steps, `cpu_offload=true`,
and the reported paper config `tau_q=0.5`, `tau_kv=0.05`, `N=6`, `D=1`,
`S_q=0.3`. This verifies the reported target tuple, not an upstream schedule
for gradually reaching those targets. The audit accepts this evidence only
when both the policy and method source hashes match the current checkout. It
records current policy hash
`7e20bf6de4b7a8d681a4971e6950254275b2e8980a8b45087f796db21acf900e`, current
method hash `37195332f2b285bc3f82231ebc13fcbec0c28cbb5745247b15f42523d2d7d103`,
completed with `status=ok`, `flashomni_full_upstream: 540`,
`flashomni_explicit_upstream: 2460`, `dense: 540`, `sparse: 2460`,
`generate_sec=4744.209`, peak CUDA allocation about 38.65 GiB, and output
`result/inference/audit/flashomni_paper_mmdit_hunyuan_current_policy_20260518/hunyuan-t2v/flashomni/seed0_720x1280_129f.mp4`.
This replaces the earlier 2-step smoke as the reported-config runtime
evidence. It proves the owned paper/benchmark-derived policy can drive a real
50-step Hunyuan run through both FlashOmni update and explicit sparse dispatch
paths. It still does not prove code-level upstream video-method parity because
the upstream Wan/Hunyuan sparse-symbol policy source is not public.

A lightweight artifact check also exists at
`result/inference/audit/flashomni_paper_mmdit_hunyuan_current_policy_20260518/qc/contact_sheet.jpg`.
`ffprobe` reports a valid 1280x720, 129-frame, 5.375-second video. The file is
about 1.3 MiB with about 1.88 Mb/s bitrate, so the audit artifact QC has no
small-file or low-bitrate warning. Treat this run as runtime/dispatch evidence
until the generated video is visually accepted against a matching dense
baseline; do not use the JSON status alone as visual quality proof.

Completion evidence still needed:

- concrete upstream code/evidence defining the Wan/Hunyuan video
  sparse-symbol update/dispatch policy and `tau_q`/`tau_kv` convergence
  schedule if we want to claim code-level parity;
- if upstream policy appears later, compare the owned policy against it and
  rerun 50-step strict-dispatch video inference with that policy.

## Hardware Deferred

### `sta_h100`

SparseVideo owns both the FastVideo Triton STA fallback and the `sta_h100`
source/extension. On the current machine, all visible GPUs are A100
compute-capability 8.0, so the real Hopper/TK C++ path cannot be exercised.

The A100 path is complete for current hardware and is explicitly labeled
`fastvideo_sta_a100_triton`. It has matching 50-step Wan14B quality and speed
evidence against the legacy FastVideo Triton fallback. This evidence is not
H100/TK runtime dispatch evidence, and it should not be reported as H100 parity.

Deferred evidence needed when H100/Hopper hardware is available:

- run on Hopper/H100 hardware;
- produce real inference metrics showing `fastvideo_sta_h100` backend dispatch;
- keep A100 runs labeled as FastVideo Triton fallback, not H100 parity.
