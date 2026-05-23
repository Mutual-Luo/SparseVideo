from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_id(spec, model_root: Path, model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    if spec.local_dir:
        local_path = model_root / spec.local_dir
        if local_path.exists():
            return str(local_path.resolve())
    return spec.hf_id


def _has_component_weight(component_dir: Path) -> bool:
    if not component_dir.exists():
        return False
    for pattern in ("*.safetensors", "*.bin", "*.ckpt", "*.pt", "*.pth", "*.index.json", "*.index.bf16.json"):
        if any(component_dir.glob(pattern)):
            return True
    return False


def _ltx_single_file_checkpoint(model_id: str) -> Optional[Path]:
    path = Path(model_id).expanduser()
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if not path.is_dir():
        return None
    if (path / "transformer" / "config.json").exists():
        return None
    preferred = (
        "ltx-video-2b-v0.9.5.safetensors",
        "ltxv-2b-0.9.6-distilled-04-25.safetensors",
        "ltxv-2b-0.9.6-dev-04-25.safetensors",
        "ltx-video-2b-v0.9.1.safetensors",
        "ltx-video-2b-v0.9.safetensors",
    )
    for name in preferred:
        checkpoint = path / name
        if checkpoint.exists():
            return checkpoint
    return None


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compatible_t5_component_root(candidate: Path, reference_config: Optional[Dict[str, Any]]) -> bool:
    text_encoder_dir = candidate / "text_encoder"
    tokenizer_dir = candidate / "tokenizer"
    if not _has_component_weight(text_encoder_dir) or not (tokenizer_dir / "spiece.model").exists():
        return False
    if reference_config is None:
        return True
    candidate_config = _read_json_file(text_encoder_dir / "config.json")
    if candidate_config is None:
        return False
    for key in ("model_type", "d_model", "num_layers", "vocab_size"):
        if candidate_config.get(key) != reference_config.get(key):
            return False
    return True


def _resolve_ltx_text_component_root(model_id: str) -> Optional[Path]:
    path = Path(model_id).expanduser()
    model_dir = path if path.is_dir() else path.parent
    reference_config = _read_json_file(model_dir / "text_encoder" / "config.json")
    candidates = [model_dir]
    if model_dir.parent.exists():
        candidates.extend(
            model_dir.parent / name
            for name in ("CogVideoX-5b", "CogVideoX-5b-I2V", "allegro", "mochi-1")
        )
    for candidate in candidates:
        if _compatible_t5_component_root(candidate, reference_config):
            return candidate
    return None


def resolve_wan_flow_shift(height: int, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return 5.0 if int(height) >= 720 else 3.0


def resolve_scheduler_flow_shift(spec, height: int, override: Optional[float]) -> Optional[float]:
    if spec.family == "wan":
        return resolve_wan_flow_shift(height, override)
    if spec.family == "hunyuan_video" and override is not None:
        return float(override)
    return None


def parse_dtype(torch_module, dtype: str):
    if dtype == "bf16":
        return torch_module.bfloat16
    if dtype == "fp16":
        return torch_module.float16
    return torch_module.float32


def seed_everything(torch_module, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def configure_method_runtime_env(method: str) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    python_prefix = Path(sys.prefix)
    nvcc = python_prefix / "bin" / "nvcc"
    if "CUDA_HOME" not in os.environ and nvcc.exists():
        os.environ["CUDA_HOME"] = str(python_prefix)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_path = Path(cuda_home)
        os.environ.setdefault("CUDA_PATH", str(cuda_path))
        nvcc_path = cuda_path / "bin" / "nvcc"
        if nvcc_path.exists():
            os.environ.setdefault("CUDACXX", str(nvcc_path))
        bin_path = str(cuda_path / "bin")
        if bin_path not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        lib_paths = [
            str(cuda_path / "lib"),
            str(cuda_path / "targets" / "x86_64-linux" / "lib"),
        ]
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        for lib_path in reversed(lib_paths):
            if lib_path not in ld_library_path.split(os.pathsep):
                ld_library_path = lib_path + os.pathsep + ld_library_path
        os.environ["LD_LIBRARY_PATH"] = ld_library_path.rstrip(os.pathsep)
        os.environ.setdefault(
            "FLASHINFER_EXTRA_LDFLAGS",
            "-L{0}/lib -L{0}/targets/x86_64-linux/lib "
            "-L{0}/lib/stubs -L{0}/targets/x86_64-linux/lib/stubs".format(cuda_home),
        )

    if method != "svoo":
        return

    cache_root = Path(os.environ.get("SVOO_CACHE_ROOT", REPO_ROOT / ".triton_cache"))
    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", cache_root))
    torchinductor_cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", cache_root / "torchinductor"))
    flashinfer_workspace = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", cache_root / "flashinfer"))

    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(flashinfer_workspace))
    os.environ.setdefault("SVOO_ENABLE_MEM_SAVE", "1")

    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    flashinfer_workspace.mkdir(parents=True, exist_ok=True)


def infer_hunyuan_prompt_length(pipe, prompt: str, max_sequence_length: int = 256) -> int:
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE

    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        raise RuntimeError("Cannot infer Hunyuan prompt_length because the pipeline has no tokenizer")

    prompts = [prompt] if isinstance(prompt, str) else prompt
    template = DEFAULT_PROMPT_TEMPLATE
    templated_prompts = [template["template"].format(item) for item in prompts]
    crop_start = template.get("crop_start", None)
    if crop_start is None:
        template_input = pipe.tokenizer(
            template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = template_input["input_ids"].shape[-1] - 2

    text_inputs = pipe.tokenizer(
        templated_prompts,
        max_length=max_sequence_length + crop_start,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    prompt_attention_mask = text_inputs.attention_mask
    if crop_start > 0:
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]
    return int(prompt_attention_mask.sum().item())


def load_pipeline(spec, model_id: str, torch_dtype, vae_dtype, local_files_only: bool,
                  height: int, flow_shift: Optional[float]):
    import tempfile

    def pipeline_load_kwargs(**kwargs):
        temp_root = REPO_ROOT / ".tmp_offload"
        temp_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(temp_root))
        tempfile.tempdir = str(temp_root)
        offload_folder = temp_root / "diffusers_state_dict"
        offload_folder.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("low_cpu_mem_usage", True)
        if os.environ.get("SPARSEVIDEO_OFFLOAD_STATE_DICT") == "1":
            kwargs.setdefault("offload_state_dict", True)
            kwargs.setdefault("offload_folder", str(offload_folder))
        device_map = os.environ.get("SPARSEVIDEO_DEVICE_MAP")
        if device_map:
            kwargs.setdefault("device_map", device_map)
        if local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    if spec.pipeline_class == "UnavailablePipeline":
        raise RuntimeError(spec.unsupported_reason or f"{spec.key} has no configured pipeline class")

    # Wan-family pipelines share the same VAE/scheduler setup pattern.
    if spec.pipeline_class in (
        "WanPipeline", "WanImageToVideoPipeline", "WanAnimatePipeline", "WanVACEPipeline",
        "SkyReelsV2Pipeline", "SkyReelsV2ImageToVideoPipeline",
    ):
        import torch
        from diffusers import AutoencoderKLWan
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        if vae_dtype is None:
            vae_dtype = torch.float32
        local_kwargs = {"local_files_only": True} if local_files_only else {}
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=vae_dtype, **local_kwargs)
        cls = getattr(__import__("diffusers", fromlist=[spec.pipeline_class]), spec.pipeline_class)
        pipe = cls.from_pretrained(model_id, vae=vae, **pipeline_load_kwargs(torch_dtype=torch_dtype))
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe

    cls_map = {
        "HunyuanVideoPipeline": lambda: __import__("diffusers", fromlist=["HunyuanVideoPipeline"]).HunyuanVideoPipeline,
        "HunyuanVideoImageToVideoPipeline": lambda: __import__("diffusers", fromlist=["HunyuanVideoImageToVideoPipeline"]).HunyuanVideoImageToVideoPipeline,
        "CogVideoXPipeline": lambda: __import__("diffusers", fromlist=["CogVideoXPipeline"]).CogVideoXPipeline,
        "CogVideoXImageToVideoPipeline": lambda: __import__("diffusers", fromlist=["CogVideoXImageToVideoPipeline"]).CogVideoXImageToVideoPipeline,
        "LTXPipeline": lambda: __import__("diffusers", fromlist=["LTXPipeline"]).LTXPipeline,
        "LTXImageToVideoPipeline": lambda: __import__("diffusers", fromlist=["LTXImageToVideoPipeline"]).LTXImageToVideoPipeline,
        "AllegroPipeline": lambda: __import__("diffusers", fromlist=["AllegroPipeline"]).AllegroPipeline,
        "MochiPipeline": lambda: __import__("diffusers", fromlist=["MochiPipeline"]).MochiPipeline,
        "EasyAnimatePipeline": lambda: __import__("diffusers", fromlist=["EasyAnimatePipeline"]).EasyAnimatePipeline,
        "SanaVideoPipeline": lambda: __import__("diffusers", fromlist=["SanaVideoPipeline"]).SanaVideoPipeline,
        "Kandinsky5T2VPipeline": lambda: __import__("diffusers", fromlist=["Kandinsky5T2VPipeline"]).Kandinsky5T2VPipeline,
    }
    if spec.pipeline_class not in cls_map:
        raise ValueError(f"Unknown pipeline class: {spec.pipeline_class}")
    cls = cls_map[spec.pipeline_class]()

    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if local_files_only:
        kwargs["local_files_only"] = True
    if spec.pipeline_class == "HunyuanVideoPipeline" and model_id == "tencent/HunyuanVideo":
        kwargs["revision"] = "refs/pr/18"
    if spec.pipeline_class in ("HunyuanVideoPipeline", "HunyuanVideoImageToVideoPipeline") and flow_shift is not None:
        from diffusers import FlowMatchEulerDiscreteScheduler
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=float(flow_shift))
    if spec.pipeline_class in ("LTXPipeline", "LTXImageToVideoPipeline"):
        checkpoint = _ltx_single_file_checkpoint(model_id)
        if checkpoint is not None:
            component_root = _resolve_ltx_text_component_root(model_id)
            if component_root is None:
                raise RuntimeError(
                    "LTX local single-file checkpoint requires a compatible local T5 text_encoder "
                    "and tokenizer because the checkpoint does not contain those weights."
                )
            from transformers import T5EncoderModel, T5Tokenizer
            text_encoder = T5EncoderModel.from_pretrained(
                component_root / "text_encoder", torch_dtype=torch_dtype, local_files_only=local_files_only,
            )
            tokenizer = T5Tokenizer.from_pretrained(component_root / "tokenizer", local_files_only=local_files_only)
            return cls.from_single_file(str(checkpoint), text_encoder=text_encoder, tokenizer=tokenizer, **kwargs)
    return cls.from_pretrained(model_id, **pipeline_load_kwargs(**kwargs))


def prepare_pipeline(
    pipe, device: str, cpu_offload: bool, vae_tiling: bool, vae_slicing: bool,
    cpu_offload_mode: str = "model", vae_decoder_chunk_size: Optional[int] = None,
) -> None:
    if hasattr(pipe, "vae") and pipe.vae is not None:
        if vae_tiling and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if vae_slicing and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if vae_decoder_chunk_size is not None:
            pipe.vae.decoder_chunk_size = int(vae_decoder_chunk_size)
    if cpu_offload:
        if cpu_offload_mode == "sequential":
            if not hasattr(pipe, "enable_sequential_cpu_offload"):
                raise RuntimeError("This pipeline does not expose enable_sequential_cpu_offload()")
            pipe.enable_sequential_cpu_offload()
        elif cpu_offload_mode == "model":
            if not hasattr(pipe, "enable_model_cpu_offload"):
                raise RuntimeError("This pipeline does not expose enable_model_cpu_offload()")
            try:
                pipe.enable_model_cpu_offload(device=device)
            except TypeError:
                pipe.enable_model_cpu_offload()
        else:
            raise RuntimeError(f"Unsupported cpu_offload_mode={cpu_offload_mode!r}")
    elif getattr(pipe, "hf_device_map", None) is not None:
        return
    else:
        pipe.to(device)


def build_call_kwargs(args, spec, prompt: str, negative_prompt: str, generator, num_frames: int, fps: int) -> Dict[str, Any]:
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else spec.guidance_scale
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    output_type = "latent" if getattr(args, "skip_decode", False) else spec.output_type
    if spec.pipeline_class == "WanAnimatePipeline":
        if any(getattr(args, k, None) is None for k in ("image", "pose_video", "face_video")):
            raise RuntimeError("WanAnimate real inference requires image, pose_video, and face_video inputs")
    if spec.pipeline_class == "WanVACEPipeline":
        if getattr(args, "reference_video", None) is None or getattr(args, "mask_video", None) is None:
            raise RuntimeError("WanVACE real inference requires video and mask inputs")
    if spec.pipeline_class == "SanaVideoPipeline":
        frame_key = "frames"
    elif spec.pipeline_class == "WanAnimatePipeline":
        frame_key = "segment_frame_length"
    else:
        frame_key = "num_frames"
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": output_type,
    }
    kwargs[frame_key] = num_frames
    if spec.pipeline_class in (
        "WanImageToVideoPipeline", "SkyReelsV2ImageToVideoPipeline",
        "HunyuanVideoImageToVideoPipeline", "CogVideoXImageToVideoPipeline", "LTXImageToVideoPipeline",
    ):
        kwargs["image"] = _load_i2v_image(args.image)
    if spec.pipeline_class == "WanAnimatePipeline":
        kwargs["image"] = _load_i2v_image(args.image)
        kwargs["pose_video"] = _load_video_frames(args.pose_video)
        kwargs["face_video"] = _load_video_frames(args.face_video)
    if spec.pipeline_class == "WanVACEPipeline":
        if args.reference_video is not None:
            kwargs["video"] = _load_video_frames(args.reference_video)
        if args.mask_video is not None:
            kwargs["mask"] = _load_video_frames(args.mask_video)
    if spec.key in ("wan22-t2v-a14b", "wan22-i2v-a14b"):
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    if spec.family == "hunyuan_video":
        kwargs["true_cfg_scale"] = args.true_cfg_scale
    if spec.family == "cogvideox":
        kwargs["use_dynamic_cfg"] = False
    if spec.family == "ltx_video":
        kwargs["frame_rate"] = fps
    return kwargs


def apply_hunyuan_i2v_prompt_template_compat(pipe, call_kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video_image2video import DEFAULT_PROMPT_TEMPLATE

    template = dict(DEFAULT_PROMPT_TEMPLATE)
    tokenizer = getattr(pipe, "tokenizer", None)
    prompt = call_kwargs.get("prompt")
    status: Dict[str, Any] = {
        "default_double_return_token_id": template.get("double_return_token_id"),
        "selected_double_return_token_id": template.get("double_return_token_id"),
        "override": False,
    }
    if tokenizer is None or prompt is None:
        call_kwargs["prompt_template"] = template
        return status
    prompt_item = prompt[0] if isinstance(prompt, list) and prompt else prompt
    if not isinstance(prompt_item, str):
        call_kwargs["prompt_template"] = template
        return status
    rendered_prompt = template["template"].format(prompt_item)
    max_length = int(template.get("crop_start", 0) or 0) + 256
    text_inputs = tokenizer(
        rendered_prompt, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt", return_attention_mask=False,
    )
    token_ids = text_inputs.input_ids[0].tolist()
    default_token_id = int(template.get("double_return_token_id", 271))
    default_count = token_ids.count(default_token_id)
    status["default_token_count"] = default_count
    if default_count == 0:
        assistant_header_end_token_id = 128007
        assistant_header_end_count = token_ids.count(assistant_header_end_token_id)
        status["assistant_header_end_token_count"] = assistant_header_end_count
        if assistant_header_end_count > 0:
            template["double_return_token_id"] = assistant_header_end_token_id
            status["selected_double_return_token_id"] = assistant_header_end_token_id
            status["override"] = True
    call_kwargs["prompt_template"] = template
    return status


def call_pipeline_with_model_compat(pipe, call_kwargs: Dict[str, Any], torch_module, spec, device: str):
    if spec.pipeline_class != "HunyuanVideoImageToVideoPipeline" or not device.startswith("cuda"):
        return pipe(**call_kwargs)
    previous_default_device = None
    if hasattr(torch_module, "get_default_device"):
        previous_default_device = torch_module.get_default_device()
    torch_module.set_default_device(device)
    try:
        return pipe(**call_kwargs)
    finally:
        torch_module.set_default_device(previous_default_device or "cpu")


def _load_i2v_image(image_path: Optional[str]):
    if image_path is None:
        raise ValueError("I2V models require --image <path>. Provide a path to the conditioning image.")
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def _load_video_frames(video_path: str):
    from diffusers.utils import load_video
    return load_video(video_path)


def should_preload_fused_native_kernels(spec, method: str) -> bool:
    if method not in ("svg1", "svg2", "svoo"):
        return False
    if spec.family in ("wan", "hunyuan_video"):
        return True
    return os.environ.get("SPARSEVIDEO_FUSED_KERNEL_BACKEND") == "native"


def should_defer_fused_native_kernel_load(spec, method: str, *, dry_run: bool) -> bool:
    return (not dry_run) and spec.family == "hunyuan_video" and method in ("svg2", "svoo")
