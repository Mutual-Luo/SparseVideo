# SparseVideo

Plug-and-play sparse attention for video diffusion models. Apply training-free sparse attention methods to accelerate inference in Wan, HunyuanVideo, and CogVideoX pipelines with a single API call.

## Installation

```bash
pip install sparsevideo
```

For methods that require Triton or FlashInfer backends:

```bash
# All optional kernels plus DiffSynth-Studio backend
pip install sparsevideo[all]

# Per-method extras
pip install sparsevideo[svoo]        # FlashInfer + Triton
pip install sparsevideo[spargeattn]  # Triton
pip install sparsevideo[sta]         # Triton

# DiffSynth-Studio scripts/backend only
pip install sparsevideo[diffsynth]
```

### Building Native CUDA Kernels

Basic import and dry-run checks do not require native extensions, but parity and benchmark runs do when the upstream
method uses C++/CUDA/Triton/FlashInfer kernels. After installing, build SparseVideo-owned kernels with:

```bash
sparsevideo-build-kernels
```

Or from Python:

```python
from sparsevideo.kernels._build import main
main()
```

This requires: CUDA toolkit, `ninja`, and `torch` with CUDA support. Compiled kernels are cached in `~/.cache/sparsevideo/` and persist across sessions.

## Quick Start

```python
import torch
from diffusers import WanPipeline
import sparsevideo

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Replace attention with a sparse method (one line)
pipe = sparsevideo.replace_attention(pipe, method="svoo")

# Generate video as usual
video = pipe("A cat playing piano", num_frames=81, num_inference_steps=50).frames[0]

# Restore dense attention when done
sparsevideo.restore_sparse_attention(pipe)
```

## Supported Methods

| Method | Paper | Backend | Models | Current audit |
|---|---|---|---|---|
| `dense` | — (baseline) | PyTorch SDPA | All | pass |
| `svg1` | Sparse-VideoGen | flex_attention | Wan, Hunyuan | pass |
| `svg2` | Sparse-VideoGen | FlashInfer + Triton k-means | Wan, Hunyuan | pass |
| `spargeattn` | SpargeAttn | Triton (spas_sage_attn) | Wan, Hunyuan | pass |
| `radial` | Radial Attention | FlashInfer / SageAttention | Wan, Hunyuan | pass |
| `sta` | FastVideo (STA) | Triton (A100) / C++ (H100) | Wan, Hunyuan | A100 pass; H100 hardware-deferred |
| `draft` | Draft Attention | Triton block-sparse | Wan, Hunyuan | pass |
| `adacluster` | AdaCluster | Triton k-means + block-sparse | Wan, Hunyuan | pass |
| `svoo` | SVOO | FlashInfer / Triton co-clustering | Wan, Hunyuan | pass |
| `flashomni` | FlashOmni | C++/CUDA sparse attention | Wan, Hunyuan | pass; visual QC caveat |

On this A100 machine the current audit is complete, including FlashOmni Hunyuan reported-config dispatch evidence.
FlashOmni still carries a separate visual-QC caveat (see the table note above). `sta_h100` remains hardware-deferred
until a Hopper/H100 machine is available.

## API Reference

```python
# Replace attention in a pipeline
pipe = sparsevideo.replace_attention(
    pipe,                        # Diffusers pipeline
    method="svoo",               # Method name
    config={"sparse_backend": "flashinfer"},  # Optional config overrides
)

# Low-level API returns a handle for explicit runtime summaries/restoration:
# handle = sparsevideo.apply_sparse_attention(...)
# sparsevideo.apply(...) is kept for compatibility.

# Restore original dense attention
sparsevideo.restore_sparse_attention(pipe)

# Get default config for a method (model-aware)
config = sparsevideo.default_method_config("svoo", model_family="wan")

# List all registered methods
methods = sparsevideo.list_methods()
```

## Inference Script

A unified inference script is provided for benchmarking and quality evaluation:

```bash
python scripts/infer_diffusers.py --model wan14b --method svoo --num-frames 81 --num-inference-steps 50

# Dry-run: validate resolved config and kernel status without loading the model
python scripts/infer_diffusers.py --model wan14b --method svoo --dry-run --print-json

# Override method config (repeatable KEY=VALUE, parsed as JSON when possible)
python scripts/infer_diffusers.py --model wan14b --method svoo --method-config top_p_kmeans=0.9
```

## DiffSynth-Studio

DiffSynth-Studio uses its own pipeline classes and native checkpoint layout, so keep it on the separate DiffSynth
entrypoints:

```bash
# List supported DiffSynth bundle keys, including deferred/local-only entries
bash scripts/download/download_diffsynth_models.sh --list

# Prefer ModelScope when available, then fall back to Hugging Face through an HF mirror.
bash scripts/download/download_diffsynth_models.sh --all --source modelscope-first --hf-endpoint https://hf-mirror.com --no-proxy

# Prefer Hugging Face/HF mirror first, then fall back to ModelScope when useful.
bash scripts/download/download_diffsynth_models.sh --all --source hf-first --hf-endpoint https://hf-mirror.com --no-proxy

# Resolve local files without loading a model
python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --dry-run --print-json

# Apply/restore SparseVideo on a downloaded DiffSynth bundle without generation
python scripts/infer_diffsynth.py --model wan21-t2v-1.3b --method svg2 --apply-only
```

## Runtime Kernel Status

Check which backends are available:

```python
from sparsevideo._runtime import optional_kernel_status
status = optional_kernel_status()
for name, info in status.items():
    print(f"{name}: {info}")
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1.0 with CUDA
- diffusers >= 0.30.0
- einops >= 0.6.0

Optional (method-dependent):
- triton >= 2.2.0
- flashinfer-python >= 0.1.0
- diffsynth >= 2.0.12 for DiffSynth-Studio pipelines
- CUDA toolkit (for building native extensions)

## License

Apache-2.0
