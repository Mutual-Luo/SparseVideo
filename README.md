<p align="center">
  <img src="assets/logo.png" alt="SparseVideo Logo" width="400"/>
</p>

# SparseVideo: One line to accelerate inference for video diffusion models [Under Development...]

**Plug-and-play sparse attention for video diffusion models. One line to accelerate inference.**

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

## Supported Sparse Attention DiT Methods

| | Method | | Method | | Method |
|:---:|---|:---:|---|:---:|---|
| ✅ | `dense`: Baseline | ✅ | `spargeattn`: SpargeAttn [[paper]](https://arxiv.org/abs/2502.18137) | ✅ | `adacluster`: AdaCluster [[paper]](https://arxiv.org/abs/2604.18348) |
| ✅ | `svg1`: Sparse-VideoGen [[paper]](https://arxiv.org/abs/2502.01776) | ✅ | `radial`: Radial Attention [[paper]](https://arxiv.org/abs/2506.19852) | ✅ | `svoo`: SVOO [[paper]](https://arxiv.org/abs/2603.18636) |
| ✅ | `svg2`: Sparse-VideoGen 2 [[paper]](https://arxiv.org/abs/2505.18875) | ✅ | `sta`: STA [[paper]](https://arxiv.org/abs/2502.04507) | ✅ | `flashomni`: FlashOmni [[paper]](https://arxiv.org/abs/2509.25401) |
| ✅ | `svgear`: SVG-EAR [[paper]](https://arxiv.org/abs/2603.08982) | ✅ | `draft`: Draft Attention [[paper]](https://arxiv.org/abs/2505.14708) | | |

## Supported Frameworks

Works as a drop-in, one-line replacement for both **Diffusers** and **DiffSynth-Studio** pipelines. Just call `sparsevideo.replace_attention(pipe, method=...)`, no model modifications required.

### Diffusers

Supported models:

| | Model | | Model | | Model |
|:---:|---|:---:|---|:---:|---|
| ✅ | Wan 2.1 Text-to-Video 1.3B | ✅ | Wan 2.2 Text-to-Video A14B | ✅ | Video-as-Prompt Wan 2.1 14B |
| ✅ | Wan 2.1 Text-to-Video 14B | ✅ | Wan 2.2 Image-to-Video A14B | ✅ | HunyuanVideo Text-to-Video |
| ✅ | Wan 2.1 Image-to-Video 14B | ✅ | Wan 2.2 Speech-to-Video 14B | ✅ | HunyuanVideo Image-to-Video |
| ✅ | Wan 2.1 VACE 1.3B | ✅ | Wan 2.2 Animate 14B | ✅ | CogVideoX Text-to-Video |
| ✅ | Wan 2.1 VACE 14B | ✅ | Wan 2.2-Fun A14B Control | ✅ | CogVideoX Image-to-Video |
| ✅ | Wan 2.1-Fun 1.3B Control | ✅ | Wan 2.2-Fun A14B Control-Camera | ✅ | EasyAnimate V5 Text-to-Video 12B |
| ✅ | Wan 2.1-Fun 1.3B InP | ✅ | SkyReels-V2 Text-to-Video 14B | ✅ | LTX-Video Text-to-Video |
| ✅ | Wan 2.1-Fun V1.1 1.3B Control | ✅ | SkyReels-V2 Image-to-Video 14B | ✅ | LTX-Video Image-to-Video |
| ✅ | Wan 2.1-Fun V1.1 1.3B Control-Camera | ✅ | MoVA 720P | ✅ | LTX-2 |
| ✅ | Wan 2.1-Fun V1.1 14B Control | ✅ | LongCat-Video | ✅ | Mochi-1 |
| ✅ | Wan 2.1-Fun V1.1 14B Control-Camera | ✅ | Krea Realtime Video 14B | ✅ | Allegro |
| ✅ | Wan 2.1 Speed-Control 1.3B | | | | |

### DiffSynth-Studio

Supported models:

| | Model | | Model | | Model |
|:---:|---|:---:|---|:---:|---|
| ✅ | Wan 2.1 Text-to-Video 1.3B | ✅ | Wan 2.1-Fun V1.1 1.3B Control | ✅ | Wan 2.2 Animate 14B |
| ✅ | Wan 2.1 Text-to-Video 14B | ✅ | Wan 2.1-Fun V1.1 1.3B Control-Camera | ✅ | Wan 2.2 Dancer 14B |
| ✅ | Wan 2.1 Image-to-Video 14B 480P | ✅ | Wan 2.1-Fun V1.1 14B Control | ✅ | Wan 2.2-Fun A14B Control |
| ✅ | Wan 2.1 Image-to-Video 14B 720P | ✅ | Wan 2.1-Fun V1.1 14B Control-Camera | ✅ | Wan 2.2-Fun A14B Control-Camera |
| ✅ | Wan 2.1 First-Last-Frame-to-Video 14B 720P | ✅ | Wan 2.1 VACE 1.3B | ✅ | LongCat-Video |
| ✅ | Wan 2.1 Speed-Control 1.3B | ✅ | Wan 2.1 VACE 14B | ✅ | Video-as-Prompt Wan 2.1 14B |
| ✅ | Wan 2.1-Fun 1.3B Control | ✅ | Wan 2.2 Text-to-Video A14B | ✅ | Krea Realtime Video 14B |
| ✅ | Wan 2.1-Fun 1.3B InP | ✅ | Wan 2.2 Image-to-Video A14B | ✅ | MoVA 720P |
| ✅ | Wan 2.1-Fun 14B Control | ✅ | Wan 2.2 Text/Image-to-Video 5B | ✅ | LTX-2 |
| ✅ | Wan 2.1-Fun 14B InP | ✅ | Wan 2.2 Speech-to-Video 14B | ✅ | LTX-2.3 |

## License

Apache-2.0
