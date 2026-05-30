# SparseVideo [Under Development...]

Plug-and-play sparse attention for video diffusion models. One line to accelerate inference — no model modifications required.

## Installation

```bash
pip install sparsevideo --no-build-isolation
```

## Quick Start

```diff
  import torch
  from diffusers import WanPipeline
  import sparsevideo

  pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16)
  pipe.to("cuda")

+ pipe = sparsevideo.replace_attention(pipe, method="svoo")

  video = pipe("A cat playing piano", num_frames=81, num_inference_steps=50).frames[0]
```

## Supported Frameworks

### Diffusers

| Backbone | Pipeline class | Model keys |
|---|---|---|
| **Wan 2.1** | `WanPipeline` | `wan21-t2v-1.3b`, `wan21-t2v-14b` |
| **Wan 2.1 I2V** | `WanImageToVideoPipeline` | `wan21-i2v-14b` |
| **Wan 2.1 VACE** | `WanVACEPipeline` | `wan21-vace-1.3b`, `wan21-vace-14b` |
| **Wan 2.2** | `WanPipeline` | `wan22-t2v-a14b`, `wan22-i2v-a14b` |
| **Wan 2.2 Animate** | `WanAnimatePipeline` | `wan22-animate-14b` |
| **HunyuanVideo** | `HunyuanVideoPipeline` | `hunyuan-t2v`, `hunyuan-i2v` |
| **CogVideoX** | `CogVideoXPipeline` / `CogVideoXImageToVideoPipeline` | `cogvideox-t2v`, `cogvideox-i2v` |
| **LTX-Video** | `LTXPipeline` / `LTXImageToVideoPipeline` | `ltx-video`, `ltx-video-i2v` |
| **Mochi-1** | `MochiPipeline` | `mochi-1` |
| **EasyAnimate V5** | `EasyAnimatePipeline` | `easyanimate-v5-t2v-12b` |

### DiffSynth-Studio

All Wan-based DiffSynth bundles via `WanVideoPipeline`, including Wan2.1/2.2 T2V, I2V, Fun, VACE, FLF2V, speed-control, and community models (LongCAT, Video-As-Prompt, KREA, etc.).

## Supported Methods

| Method | Paper | Backend | Wan | Hunyuan | CogVideoX | LTX | Mochi | EasyAnimate |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dense` | baseline | PyTorch SDPA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `svg1` | Sparse-VideoGen | flex\_attention | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `svg2` | Sparse-VideoGen | FlashInfer + Triton k-means | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `spargeattn` | SpargeAttn | Triton sparse-sage | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `radial` | Radial Attention | FlashInfer / SageAttention | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `sta` | FastVideo (STA) | block-sparse CUDA (SM80+) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `draft` | Draft Attention | Triton block-sparse | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `adacluster` | AdaCluster | Triton k-means + block-sparse | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `svoo` | SVOO | FlashInfer / Triton co-clustering | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `flashomni` | FlashOmni | C++/CUDA sparse attention | ✓ | — | — | — | — | — |

## License

Apache-2.0
