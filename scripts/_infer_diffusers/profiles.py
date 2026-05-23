from __future__ import annotations

import copy
import argparse
from typing import Any, Dict

from .models import (
    ModelSpec,
    HUNYUAN_VIDEO_NEGATIVE_PROMPT,
    WAN_SAMPLE_NEGATIVE_PROMPT,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SEED,
    DEFAULT_NEGATIVE_PROMPT,
)


UPSTREAM_INFERENCE_PROFILES: Dict[tuple, Dict[str, Any]] = {
    # Sparse-VideoGen/scripts/wan/wan_t2v_720p_svg.sh
    ("svg1", "wan21-t2v-14b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 50,
        "fps": 16, "flow_shift": 5.0,
        "source": "training_free/Sparse-VideoGen/scripts/wan/wan_t2v_720p_svg.sh",
    },
    # Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_svg.sh
    ("svg1", "hunyuan_video"): {
        "height": 720, "width": 1280, "num_frames": 129, "num_inference_steps": 50,
        "fps": 24, "flow_shift": 7.0, "vae_tiling": True, "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_svg.sh",
    },
    # Sparse-VideoGen/scripts/wan/wan_t2v_720p_sap.sh; public SparseVideo name is svg2.
    ("svg2", "wan21-t2v-14b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 50,
        "fps": 16, "flow_shift": 5.0,
        "source": "training_free/Sparse-VideoGen/scripts/wan/wan_t2v_720p_sap.sh",
    },
    # Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_sap.sh; public SparseVideo name is svg2.
    ("svg2", "hunyuan_video"): {
        "height": 720, "width": 1280, "num_frames": 129, "num_inference_steps": 50,
        "fps": 24, "flow_shift": 7.0, "vae_tiling": True, "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/Sparse-VideoGen/scripts/hyvideo/hyvideo_t2v_720p_sap.sh",
    },
    # SpargeAttn/inference_examples/wan_infer.py defaults to --mode full.
    ("spargeattn", "wan21-t2v-1.3b"): {
        "height": 480, "width": 832, "num_frames": 81, "fps": 15, "guidance_scale": 5.0,
        "seed": 42, "cpu_offload": True, "cpu_offload_mode": "sequential",
        "vae_tiling": True, "vae_slicing": True, "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/wan_infer.py",
    },
    ("spargeattn", "wan21-t2v-14b"): {
        "height": 480, "width": 832, "num_frames": 81, "fps": 15, "guidance_scale": 5.0,
        "seed": 42, "cpu_offload": True, "cpu_offload_mode": "sequential",
        "vae_tiling": True, "vae_slicing": True, "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/wan_infer.py",
    },
    # SpargeAttn/inference_examples/README.md shows Wan2.2 with --mode topk --value 0.4.
    ("spargeattn", "wan22-t2v-a14b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 40,
        "fps": 16, "guidance_scale": 4.0, "guidance_scale_2": 3.0, "seed": 42,
        "cpu_offload": True, "cpu_offload_mode": "sequential",
        "vae_tiling": True, "vae_slicing": True, "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "topk", "value": 0.4},
        "source": "training_free/SpargeAttn/inference_examples/README.md",
        "evidence_sources": ["training_free/SpargeAttn/inference_examples/wan_infer.py"],
    },
    # SpargeAttn/inference_examples/hunyuan_infer.py defaults to --mode full.
    ("spargeattn", "hunyuan_video"): {
        "height": 320, "width": 512, "num_frames": 61, "num_inference_steps": 30,
        "fps": 8, "seed": 42, "cpu_offload": True, "cpu_offload_mode": "sequential",
        "vae_tiling": True, "vae_slicing": True, "vae_decoder_chunk_size": 1,
        "method_config": {"mode": "full", "value": None},
        "source": "training_free/SpargeAttn/inference_examples/hunyuan_infer.py",
    },
    # radial-attention/wan_t2v_inference.py defaults to Wan2.1 T2V 14B.
    ("radial", "wan21-t2v-14b"): {
        "height": 768, "width": 1280, "num_frames": 69, "num_inference_steps": 50,
        "flow_shift": 5.0,
        "source": "training_free/radial-attention/scripts/wan_t2v_inference.sh",
    },
    # radial-attention/scripts/wan_22_t2v_inference.sh
    ("radial", "wan22-t2v-a14b"): {
        "height": 768, "width": 1280, "num_frames": 77, "num_inference_steps": 40,
        "guidance_scale": 4.0, "guidance_scale_2": 3.0, "vae_tiling": True, "vae_slicing": False,
        "source": "training_free/radial-attention/scripts/wan_22_t2v_inference.sh",
    },
    # radial-attention/scripts/hunyuan_t2v_inference.sh
    ("radial", "hunyuan_video"): {
        "height": 768, "width": 1280, "num_frames": 117, "num_inference_steps": 50,
        "vae_tiling": True, "vae_slicing": False,
        "source": "training_free/radial-attention/scripts/hunyuan_t2v_inference.sh",
    },
    # FastVideo current checkout keeps STA inference as archived workflow docs.
    ("sta", "wan21-t2v-14b"): {
        "height": 768, "width": 1280, "num_frames": 69, "num_inference_steps": 50,
        "method_config": {"seq_shape": "18x48x80"},
        "source": "training_free/FastVideo/docs/attention/sta/index.md",
    },
    ("sta", "hunyuan_video"): {
        "height": 768, "width": 1280, "num_frames": 117, "num_inference_steps": 50,
        "method_config": {"seq_shape": "30x48x80"},
        "source": "training_free/FastVideo/docs/attention/sta/index.md",
    },
    # draft-attention/wan/run-single-inference.sh uses Wan2.1 T2V 14B at --size 768*512.
    ("draft", "wan21-t2v-14b"): {
        "height": 512, "width": 768, "num_frames": 81, "num_inference_steps": 50,
        "flow_shift": 5.0, "seed": 42,
        "method_config": {"latent_h": 32, "latent_w": 48, "visual_len": 32_256, "text_len": 0, "batch_size": 1},
        "source": "training_free/draft-attention/wan/run-single-inference.sh",
        "evidence_sources": [
            "training_free/draft-attention/wan/generate.py",
            "training_free/draft-attention/wan/wan/configs/shared_config.py",
        ],
    },
    # draft-attention README/Hunyuan demo uses Hunyuan 768p at 129 frames.
    ("draft", "hunyuan_video"): {
        "height": 768, "width": 1280, "num_frames": 129, "num_inference_steps": 50,
        "seed": 42, "cpu_offload": True, "cpu_offload_mode": "sequential",
        "method_config": {"latent_h": 48, "latent_w": 80, "visual_len": 126_720, "text_len": 256},
        "source": "training_free/draft-attention/README.md",
    },
    # Adacluster/runwan/runwan.py uses Wan2.1 T2V 1.3B at --size 832*480.
    ("adacluster", "wan21-t2v-1.3b"): {
        "height": 480, "width": 832, "num_frames": 81, "num_inference_steps": 50,
        "fps": 16, "flow_shift": 5.0, "negative_prompt": WAN_SAMPLE_NEGATIVE_PROMPT,
        "source": "training_free/Adacluster/runwan/runwan.py",
        "evidence_sources": [
            "training_free/Adacluster/runwan/generate.py",
            "training_free/Adacluster/runwan/wan/configs/shared_config.py",
        ],
    },
    # Adacluster/runhunyuan/run_hunyuan.py hardcodes 720p, 81 frames, 30 steps, fps=15.
    ("adacluster", "hunyuan_video"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 30,
        "fps": 15, "cpu_offload": True, "cpu_offload_mode": "model", "vae_tiling": True, "vae_slicing": False,
        "source": "training_free/Adacluster/runhunyuan/run_hunyuan.py",
    },
    # SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh.
    ("svoo", "wan21-t2v-1.3b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 50,
        "fps": 16, "flow_shift": 3.0, "vae_dtype": "bf16", "vae_tiling": False, "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    ("svoo", "wan21-t2v-14b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 50,
        "fps": 16, "flow_shift": 3.0, "vae_dtype": "bf16", "vae_tiling": False, "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    ("svoo", "wan22-t2v-a14b"): {
        "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 40,
        "fps": 16, "guidance_scale": 5.0, "guidance_scale_2": 3.0,
        "flow_shift": 3.0, "vae_dtype": "bf16", "vae_tiling": False, "vae_slicing": False,
        "source": "training_free/SVOO/scripts/inference/wan/wan_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/wan_t2v_inference.py"],
    },
    # SVOO/scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh.
    ("svoo", "hunyuan_video"): {
        "height": 720, "width": 1280, "num_frames": 129, "num_inference_steps": 50,
        "fps": 24, "seed": 23, "flow_shift": 7.0, "vae_tiling": True, "vae_slicing": False,
        "negative_prompt": HUNYUAN_VIDEO_NEGATIVE_PROMPT,
        "source": "training_free/SVOO/scripts/inference/hunyuan10/hunyuan10_t2v_720p_svoo.sh",
        "evidence_sources": ["training_free/SVOO/hunyuan10_t2v_inference.py"],
    },
}


def resolve_inference_profile(profile: str, spec: ModelSpec, method: str) -> Dict[str, Any]:
    if profile == "default":
        return {}
    if profile != "upstream":
        raise ValueError(f"Unknown inference profile: {profile}")
    exact = UPSTREAM_INFERENCE_PROFILES.get((method, spec.key))
    family = UPSTREAM_INFERENCE_PROFILES.get((method, spec.family))
    selected = exact if exact is not None else family
    if selected is None:
        raise ValueError(
            f"No upstream inference profile is defined for method={method!r}, model={spec.key!r}. "
            "Use --profile default, or choose a method/model pair with a referenced training_free benchmark profile."
        )
    return copy.deepcopy(selected)


def apply_profile_runtime_defaults(
    args: argparse.Namespace,
    profile: Dict[str, Any],
    fps: int,
    num_frames: int,
    steps: int,
) -> tuple:
    height = args.height if args.height is not None else int(profile.get("height", DEFAULT_HEIGHT))
    width = args.width if args.width is not None else int(profile.get("width", DEFAULT_WIDTH))

    if args.fps is None and "fps" in profile:
        fps = int(profile["fps"])
    if args.num_frames is None and args.duration_seconds is None and "num_frames" in profile:
        num_frames = int(profile["num_frames"])
    if args.num_inference_steps is None and "num_inference_steps" in profile:
        steps = int(profile["num_inference_steps"])
    if args.guidance_scale is None and "guidance_scale" in profile:
        args.guidance_scale = float(profile["guidance_scale"])
    if args.guidance_scale_2 == 3.0 and "guidance_scale_2" in profile:
        args.guidance_scale_2 = float(profile["guidance_scale_2"])
    if args.true_cfg_scale == 1.0 and "true_cfg_scale" in profile:
        args.true_cfg_scale = float(profile["true_cfg_scale"])
    if args.flow_shift is None and "flow_shift" in profile:
        args.flow_shift = float(profile["flow_shift"])
    if args.vae_dtype is None and "vae_dtype" in profile:
        args.vae_dtype = str(profile["vae_dtype"])
    if args.negative_prompt is None and "negative_prompt" in profile:
        args.negative_prompt = str(profile["negative_prompt"])
    if args.seed is None and "seed" in profile:
        args.seed = int(profile["seed"])
    if args.cpu_offload is None and "cpu_offload" in profile:
        args.cpu_offload = bool(profile["cpu_offload"])
    if args.cpu_offload_mode is None and "cpu_offload_mode" in profile:
        args.cpu_offload_mode = str(profile["cpu_offload_mode"])
    if args.vae_tiling is None and "vae_tiling" in profile:
        args.vae_tiling = bool(profile["vae_tiling"])
    if args.vae_slicing is None and "vae_slicing" in profile:
        args.vae_slicing = bool(profile["vae_slicing"])
    if args.vae_decoder_chunk_size is None and "vae_decoder_chunk_size" in profile:
        args.vae_decoder_chunk_size = int(profile["vae_decoder_chunk_size"])

    _fill_defaults(args)
    args.height = height
    args.width = width
    if args.num_inference_steps is None:
        args.num_inference_steps = steps
    return height, width, fps, num_frames, steps


def finalize_runtime_defaults(args: argparse.Namespace) -> None:
    _fill_defaults(args)


def _fill_defaults(args: argparse.Namespace) -> None:
    if args.seed is None:
        args.seed = DEFAULT_SEED
    if args.cpu_offload is None:
        args.cpu_offload = False
    if args.cpu_offload_mode is None:
        args.cpu_offload_mode = "model"
    if args.vae_tiling is None:
        args.vae_tiling = False
    if args.vae_slicing is None:
        args.vae_slicing = False
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT
