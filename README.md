# SparseVideo

SparseVideo makes sparse attention methods plug-and-play for video diffusion generation.

The package goal is simple: users should be able to take an existing video diffusion pipeline, enable one sparse
attention method, compare it with another method, and restore the original dense attention path without rewriting model
code.

```python
from sparsevideo import apply_sparse_attention

handle = apply_sparse_attention(pipe, method="svoo", config={})
video = pipe(prompt).frames[0]
handle.restore()
```

SparseVideo is an extension layer, not a new video generation framework.

```text
Diffusers / DiffSynth-Studio = model and pipeline ecosystems
SparseVideo                  = sparse attention adapters, validation, and benchmarking
```

## Goals

- Make sparse attention easy to apply to video diffusion pipelines.
- Provide one interface for switching between sparse attention methods.
- Support real generation workloads first through Diffusers, then DiffSynth-Studio.
- Keep implementations modular enough for new methods to be added cleanly.
- Prefer correctness and reproducibility before speed and memory optimization.
- Avoid model-specific hacks when a general adapter design is practical.
- Keep the package lightweight enough to publish and install with pip.

## Method Scope

Training-free methods to support:

| Public name | Source | Note |
| --- | --- | --- |
| `svg1` | `training_free/Sparse-VideoGen` | Sparse-VideoGen SVG method. |
| `svg2` | `training_free/Sparse-VideoGen` | Sparse-VideoGen second method; use `SVG2` as the public name. |
| `spargeattn` | `training_free/SpargeAttn` | Optional CUDA extension path with dense fallback. |
| `radial` | `training_free/radial-attention` | Port attention logic into SparseVideo adapters. |
| `sta` | `training_free/FastVideo` | Sliding Tile Attention from FastVideo; do not depend on the full framework. |
| `draft` | `training_free/draft-attention` | Treat upstream code as a reference implementation. |
| `adacluster` | `training_free/Adacluster` | Treat upstream code as a reference implementation. |
| `flashomni` | `training_free/FlashOmni` | Optional sparse attention engine, not a core dependency. |
| `svoo` | `training_free/SVOO` | Port as a first-class SparseVideo method. |

`dense` should always remain available as the correctness baseline and fallback.

## Design Rules

- Public APIs should be small and stable.
- Sparse methods should share the same apply/restore lifecycle.
- Method configs should be explicit dictionaries or typed configs, not hidden globals.
- Optional CUDA kernels must not be required for basic package import.
- Upstream `training_free/` repositories are references; do not let their old requirements define the main environment.
- When a sparse path is unsupported or unsafe, fall back to dense attention or raise a clear error.

## Environment

Primary development environment:

```text
Python: /home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
Models: /home/dataset-assist-0/luojy/models
```

Recommended baseline:

```text
Python:    3.11 or 3.12
PyTorch:   CUDA-enabled stable build
Diffusers: latest stable when possible
CUDA:      12.x recommended
```

Do not install old upstream method requirements into the main environment blindly. Many method repos pin older Torch,
Diffusers, FlashAttention, FlashInfer, or Triton versions.

For local development:

```bash
python -m pip install -e .
python - <<'PY'
import sparsevideo
print(sparsevideo.list_methods())
PY
```

## Development Milestones

### v0.1

- Package skeleton and pip-installable project.
- Diffusers Wan text-to-video/image-to-video adapter.
- Public `apply_sparse_attention(...)` API with `handle.restore()`.
- `dense`, `topk`, and one verified sparse method.
- Dense-vs-sparse correctness tests on small tensors.

### v0.2

- Public method names aligned with the method table, including `svg1` and `svg2`.
- Verified Wan adapters for more included methods.
- Basic benchmark output with method, config, model, prompt, seed, latency, and peak memory.

### v0.3

- HunyuanVideo and CogVideoX Diffusers adapters.
- DiffSynth-Studio integration.
- Optional SpargeAttn, FlashOmni, and other kernel-backed paths.

## Current Status

This repository is still an early implementation. Before treating it as complete, verify:

- `pip install -e .` works in the target environment.
- The README example imports successfully.
- Each public method name is registered.
- Wan dense attention can be patched and restored.
- At least one sparse method matches dense output on small correctness tests.
- Unsupported models or missing optional kernels fail with clear messages.

## Citation

SparseVideo adapts ideas from the upstream training-free sparse attention projects listed above. Preserve original
licenses and citations when porting method-specific code.
