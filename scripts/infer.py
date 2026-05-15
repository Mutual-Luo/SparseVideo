#!/usr/bin/env python3
#
# SparseVideo inference entrypoint.
#
# This is the only inference script you need. Select the model and sparse
# attention method by command-line arguments:
#
#   python scripts/infer.py --model wan1.3b --method dense
#   python scripts/infer.py --model wan1.3b --method svoo
#   python scripts/infer.py --model hunyuan --method sta
#
# Supported models:
#   wan1.3b    Wan2.1 T2V 1.3B Diffusers
#   wan14b     Wan2.1 T2V 14B Diffusers
#   wan22      Wan2.2 T2V A14B Diffusers
#   hunyuan    HunyuanVideo T2V
#   cogvideox  CogVideoX dense baseline only
#
# Supported methods:
#   dense       Original dense attention baseline.
#   svg1        Sparse-VideoGen SVG-style method.
#   svg2        Sparse-VideoGen second method.
#   spargeattn  SpargeAttn method.
#   radial      radial-attention method.
#   sta         FastVideo Sliding Tile Attention.
#   draft       draft-attention method.
#   adacluster  AdaCluster method.
#   flashomni   FlashOmni method.
#   svoo        SVOO method.
#
# Current sparse support:
#   Wan and Hunyuan support the sparse methods above.
#   CogVideoX is included only for the dense baseline until processors are added.
#
# Common options:
#
#   --dry-run                 Show resolved settings without loading model.
#                             Also reports optional kernel availability.
#   --num-frames 81           Override exact frame count.
#   --num-inference-steps 10  Run fewer steps for a smoke test.
#   --prompt "..."            Override prompt from the command line.
#   --prompt-file prompt.txt  Read prompt from a file.
#   --cpu-offload             Use pipeline CPU offload if available.
#   --strict-kernels          Fail before model load if the selected method
#                             would use a slower fallback instead of its
#                             configured native/package kernel path.
#
# SVOO runtime env defaults:
#   SVOO_CACHE_ROOT           Base compiler cache, default .triton_cache.
#   TRITON_CACHE_DIR          Triton JIT cache.
#   TORCHINDUCTOR_CACHE_DIR   TorchInductor cache.
#   FLASHINFER_WORKSPACE_BASE FlashInfer workspace cache.
#   SVOO_TRITON_TUNE=auto     Optional SVOO Triton autotuning.
#   SVOO_ENABLE_MEM_SAVE=0|1  Release SVOO intermediates earlier; default 1.
#   SPARSEVIDEO_FUSED_KERNEL_BACKEND=auto|native|triton|pytorch
#                            auto uses SparseVideo-owned _kernels when found.
#
# Method config names follow the upstream repositories where possible:
#   svg1:        first_layers_fp, first_times_fp, num_sampled_rows,
#                sample_mse_max_row, sparsity.
#   svg2:        first_layers_fp, first_times_fp, num_q_centroids,
#                num_k_centroids, top_p_kmeans, min_kc_ratio,
#                kmeans_iter_init, kmeans_iter_step, zero_step_kmeans_init.
#   spargeattn:  mode=full|cdfthreshd|topk, value, tune,
#                parallel_tune, l1, pv_l1, tune_pv, verbose.
#                Upstream default mode is full; set mode/value to enable sparse.
#   radial:      dense_layers, dense_timesteps, decay_factor,
#                use_sage_attention.
#   sta:         tile_size, window_size, seq_shape, has_text.
#                FastVideo native STA uses tile_size=6,8,8 and supports
#                seq_shape=18x48x80, 30x48x80, or 36x48x48. Other 720p shapes
#                still run through SparseVideo's Triton STA path.
#   draft:       pool_h, pool_w, sparsity_ratio, block_sparse_attention.
#   adacluster:  topk_num, q_kernel_num, kv_kernel_num,
#                kmeans_iter_init, kmeans_iter_step, late_layer_start,
#                late_topk_num, late_q_kernel_num, late_kv_kernel_num.
#   flashomni:   implementation=upstream|flex, backend, workspace_bytes,
#                sparse_block_size_for_q, sparse_block_size_for_kv,
#                sparse_info, sparse_kv_info, sparse_info_indptr,
#                sparse_kv_info_indptr, is_full, sparse_kv_budget.
#   svoo:        first_times_fp, first_layers_fp, num_q_centroids,
#                num_k_centroids, top_p_kmeans, min_kc_ratio,
#                kmeans_iter_init, kmeans_iter_step, zero_step_kmeans_init,
#                start_reuse_step, reuse_interval, use_dynamic_min_kc_ratio,
#                sparsity_csv_path, dynamic_min_kc_ratio_min,
#                dynamic_min_kc_ratio_max, context_length, prompt_length,
#                implementation=native, sparse_backend=flashinfer|triton.
# Old local names like budget/kernel_size are compatibility aliases only.
#
# Example SVOO overrides:
#   python scripts/infer.py --model wan1.3b --method svoo \
#     --method-config top_p_kmeans=0.9 --method-config min_kc_ratio=0.1
#
# Defaults:
#   Wan uses 81 frames at 16 fps.
#   Hunyuan uses 129 frames at 24 fps.
#   Method configs use the reference repositories' public names where possible.
#   For SVG2 and SVOO, this script uses the 720p inference-shell defaults for
#   Wan/Hunyuan instead of the weak parser defaults, because those are the
#   settings used for quality/speed comparison.
#   Wan loads the VAE in fp32 and uses flow_shift=5.0 at 720p, matching the
#   Diffusers/Wan reference example. Override with --flow-shift if needed.
#   SVOO dynamic CSV sparsity is enabled by those defaults; this script resolves
#   the default profile to src/sparsevideo/methods/svoo/sparsity_profiles/.
#   Outputs go to result/inference/<model>/<method>/.
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

METHODS = (
    "adacluster",
    "dense",
    "draft",
    "flashomni",
    "radial",
    "spargeattn",
    "sta",
    "svg1",
    "svg2",
    "svoo",
)

STA_NATIVE_SEQ_SHAPES = {"18x48x80", "30x48x80", "36x48x48"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    pipeline_class: str
    hf_id: str
    local_dir: Optional[str]
    fps: int
    default_frames: int
    default_steps: int
    guidance_scale: float
    output_type: str
    sparse_supported: bool = True


MODEL_SPECS: Dict[str, ModelSpec] = {
    "wan21-t2v-1.3b": ModelSpec(
        key="wan21-t2v-1.3b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        local_dir="Wan2.1-T2V-1.3B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan21-t2v-14b": ModelSpec(
        key="wan21-t2v-14b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        local_dir="Wan2.1-T2V-14B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=50,
        guidance_scale=5.0,
        output_type="np",
    ),
    "wan22-t2v-a14b": ModelSpec(
        key="wan22-t2v-a14b",
        family="wan",
        pipeline_class="WanPipeline",
        hf_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        local_dir="Wan2.2-T2V-A14B-Diffusers",
        fps=16,
        default_frames=81,
        default_steps=40,
        guidance_scale=5.0,
        output_type="np",
    ),
    "hunyuan-t2v": ModelSpec(
        key="hunyuan-t2v",
        family="hunyuan_video",
        pipeline_class="HunyuanVideoPipeline",
        hf_id="tencent/HunyuanVideo",
        local_dir="HunyuanVideo",
        fps=24,
        default_frames=129,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
    ),
    "cogvideox-t2v": ModelSpec(
        key="cogvideox-t2v",
        family="cogvideox",
        pipeline_class="CogVideoXPipeline",
        hf_id="THUDM/CogVideoX-5b",
        local_dir="CogVideoX-5b",
        fps=8,
        default_frames=49,
        default_steps=50,
        guidance_scale=6.0,
        output_type="pil",
        sparse_supported=False,
    ),
}


MODEL_ALIASES = {
    "wan1.3b": "wan21-t2v-1.3b",
    "wan21-1.3b": "wan21-t2v-1.3b",
    "wan21-t2v-1.3b": "wan21-t2v-1.3b",
    "wan14b": "wan21-t2v-14b",
    "wan21-14b": "wan21-t2v-14b",
    "wan21-t2v-14b": "wan21-t2v-14b",
    "wan22": "wan22-t2v-a14b",
    "wan22-a14b": "wan22-t2v-a14b",
    "wan22-t2v-a14b": "wan22-t2v-a14b",
    "hunyuan": "hunyuan-t2v",
    "hunyuan-t2v": "hunyuan-t2v",
    "cog": "cogvideox-t2v",
    "cogvideox": "cogvideox-t2v",
    "cogvideox-t2v": "cogvideox-t2v",
}


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one SparseVideo inference job and write timing metrics.",
        epilog=(
            "Examples:\n"
            "  python scripts/infer.py --model wan1.3b --method dense\n"
            "  python scripts/infer.py --model wan1.3b --method svoo --num-inference-steps 10\n"
            "  python scripts/infer.py --model hunyuan --method radial --prompt-file prompt.txt\n"
            "  python scripts/infer.py --model wan1.3b --method sta --num-frames 81\n"
            "\n"
            "Models: wan1.3b, wan14b, wan22, hunyuan, cogvideox\n"
            "Methods: dense, svg1, svg2, spargeattn, radial, sta, draft, "
            "adacluster, flashomni, svoo\n"
            "Note: sparse methods currently support Wan and Hunyuan; CogVideoX is dense-only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), required=True)
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument("--model-root", type=Path, default=Path("/home/dataset-assist-0/luojy/models"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a red sports car driving along a coastal road at sunset, detailed, realistic")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--flow-shift", type=float, default=None, help="Wan scheduler flow_shift. Default: 5.0 for 720p, 3.0 below 720p.")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "result" / "inference")
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=REPO_ROOT / "result" / "inference" / "metrics.jsonl")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict-kernels",
        action="store_true",
        help="Fail preflight when the selected method would use a slower fallback kernel path.",
    )
    parser.add_argument("--method-config-json", type=str, default=None)
    parser.add_argument(
        "--method-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra sparse method config, e.g. top_p_kmeans=0.9 for SVOO. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_method_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if args.method_config_json:
        loaded = json.loads(args.method_config_json)
        if not isinstance(loaded, dict):
            raise ValueError("--method-config-json must decode to an object")
        config.update(loaded)
    for item in args.method_config:
        if "=" not in item:
            raise ValueError(f"Invalid --method-config {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        config[key] = parse_json_value(value)
    return config


def default_svoo_sparsity_csv_path(spec: ModelSpec) -> str:
    profile_dir = SRC_ROOT / "sparsevideo" / "methods" / "svoo" / "sparsity_profiles"
    if spec.family == "hunyuan_video":
        sparsity_csv = profile_dir / "sparsity_hunyuan10_13B_t2v.csv"
    elif spec.key == "wan22-t2v-a14b":
        sparsity_csv = profile_dir / "sparsity_wan22_A14B_t2v.csv"
    else:
        filename = (
            "sparsity_wan_14B_t2v.csv"
            if spec.key == "wan21-t2v-14b"
            else "sparsity_wan_1.3B_t2v.csv"
        )
        sparsity_csv = profile_dir / filename

    return str(sparsity_csv)


def validate_method_config(method: str, config: Dict[str, Any]) -> None:
    if method == "spargeattn" and config.get("mode") != "full" and config.get("value") is None:
        raise ValueError("spargeattn requires --method-config value=<float> when mode is cdfthreshd or topk")
    if method == "radial" and config.get("use_sage_attention"):
        raise NotImplementedError("radial use_sage_attention is recognized but not ported in SparseVideo yet")
    if method == "flashomni":
        if config.get("implementation") not in ("upstream", "flex"):
            raise ValueError("flashomni implementation must be upstream or flex")
        tensor_keys = ("sparse_info", "sparse_kv_info", "sparse_info_indptr", "sparse_kv_info_indptr")
        provided = [key for key in tensor_keys if config.get(key) is not None]
        if provided:
            raise NotImplementedError(
                "FlashOmni sparse-info tensor inputs are recognized but not wired "
                f"through this inference script yet: {provided}"
            )
    if method == "draft" and not config.get("block_sparse_attention"):
        raise NotImplementedError(
            "draft block_sparse_attention=False disables the upstream sparse path; "
            "use --method dense for the dense baseline."
        )
    if method == "svoo":
        if config.get("implementation") != "native":
            raise NotImplementedError(
                "svoo implementation must be native; SparseVideo no longer uses training_free runtime bridges"
            )
        if config.get("sparse_backend") not in ("flashinfer", "triton"):
            raise ValueError("svoo sparse_backend must be flashinfer or triton")


def normalize_seq_shape_for_warning(seq_shape: Any) -> Optional[str]:
    if seq_shape is None:
        return None
    if isinstance(seq_shape, str):
        return seq_shape.lower()
    if isinstance(seq_shape, (list, tuple)) and len(seq_shape) == 3:
        return "x".join(str(int(part)) for part in seq_shape)
    return str(seq_shape)


def preflight_runtime(
    method: str,
    config: Dict[str, Any],
    device: str,
    runtime_status: Dict[str, Any],
    strict_kernels: bool = False,
) -> Dict[str, Any]:
    kernels = runtime_status["optional_kernels"]
    torch_status = runtime_status["torch"]
    errors = []
    warnings = []

    if device.startswith("cuda"):
        if not torch_status.get("cuda_available"):
            errors.append(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )
    elif method != "dense":
        errors.append("Sparse methods require --device cuda for fair inference benchmarking.")

    fused = kernels["svg_svoo_fused_kernels"]
    if method in ("svg1", "svg2", "svoo") and fused.get("backend_env") == "native":
        if not fused.get("native_extension"):
            errors.append(
                "SPARSEVIDEO_FUSED_KERNEL_BACKEND=native requires a built SparseVideo _kernels extension. "
                f"Searched: {fused.get('candidate_dirs')}."
            )
    elif (
        method in ("svg1", "svg2", "svoo")
        and fused.get("backend_env") == "auto"
        and not fused.get("native_extension")
    ):
        message = (
            "SparseVideo _kernels extension is not detected; RMSNorm/RoPE will use the Triton/PyTorch "
            "path. Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton to benchmark that path explicitly."
        )
        if strict_kernels:
            errors.append(message)
        else:
            warnings.append(message)

    if method == "spargeattn" and config.get("mode") != "full":
        sparge = kernels["spas_sage_attn"]
        if not (sparge.get("package") and sparge.get("qattn_extension") and sparge.get("fused_extension")):
            errors.append(
                "spargeattn sparse modes require spas_sage_attn with _qattn and _fused extensions built."
            )
    elif method == "spargeattn":
        warnings.append("spargeattn mode=full runs dense attention; set mode/value to benchmark sparse SpargeAttn.")

    if method == "flashomni" and config.get("implementation") == "upstream":
        flashomni = kernels["flashomni"]
        if not flashomni.get("package"):
            errors.append(
                "flashomni implementation=upstream requires the flashomni-python package. "
                "Install/build it with its CUDA/C++ ops before running this method."
            )
        elif not flashomni.get("aot_config"):
            errors.append(
                "flashomni implementation=upstream requires FlashOmni AOT kernels "
                "(install with FLASHOMNI_ENABLE_AOT=1)."
            )
        elif not flashomni.get("native_extension"):
            errors.append(
                "flashomni implementation=upstream requires flashomni_kernels*.so from the AOT build."
            )
    if method == "flashomni" and config.get("is_full"):
        warnings.append("flashomni is_full=true runs dense attention; use is_full=false for sparse benchmarking.")

    if method == "svoo" and config.get("sparse_backend") == "flashinfer":
        if not kernels["flashinfer"].get("package"):
            errors.append("svoo sparse_backend=flashinfer requires the flashinfer package.")
        elif not kernels["flashinfer"].get("sparse_module"):
            errors.append("svoo sparse_backend=flashinfer requires flashinfer.sparse APIs.")

    if method in ("adacluster", "draft", "radial", "svg2") and not kernels["flashinfer"].get("package"):
        message = (
            f"{method} will not use FlashInfer because it is not importable; runtime may fall back to slower kernels."
        )
        if strict_kernels:
            errors.append(message)
        else:
            warnings.append(message)
    if method == "sta" and not kernels["fastvideo_kernel"].get("native_extension"):
        message = "sta fastvideo_kernel native extension is not detected; it may use the slower Triton path."
        if strict_kernels:
            errors.append(message)
        else:
            warnings.append(message)
    if method == "sta":
        seq_shape = normalize_seq_shape_for_warning(config.get("seq_shape"))
        if seq_shape is None:
            message = (
                "sta seq_shape is not set. SparseVideo will infer the video layout from token length; "
                "FastVideo C++ native STA is only used for seq_shape in "
                f"{sorted(STA_NATIVE_SEQ_SHAPES)}."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)
        elif seq_shape not in STA_NATIVE_SEQ_SHAPES:
            message = (
                f"sta seq_shape={seq_shape} is outside FastVideo C++ native STA shapes "
                f"{sorted(STA_NATIVE_SEQ_SHAPES)}; SparseVideo Triton STA will be used."
            )
            if strict_kernels:
                errors.append(message)
            else:
                warnings.append(message)

    return {"errors": errors, "warnings": warnings}


def default_num_frames(duration_seconds: float, fps: int) -> int:
    frames = max(1, int(round(duration_seconds * fps)) + 1)
    remainder = (frames - 1) % 4
    if remainder:
        frames += 4 - remainder
    return frames


def resolve_model_id(spec: ModelSpec, model_root: Path, model_path: Optional[str]) -> str:
    if model_path:
        return model_path
    if spec.local_dir:
        local_path = model_root / spec.local_dir
        if local_path.exists():
            return str(local_path.resolve())
    return spec.hf_id


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


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


def resolve_wan_flow_shift(height: int, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return 5.0 if int(height) >= 720 else 3.0


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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SVOO_ENABLE_MEM_SAVE", "1")

    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    flashinfer_workspace.mkdir(parents=True, exist_ok=True)


def load_pipeline(
    spec: ModelSpec,
    model_id: str,
    torch_dtype,
    local_files_only: bool,
    height: int,
    flow_shift: Optional[float],
):
    if spec.pipeline_class == "WanPipeline":
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        kwargs = {"local_files_only": True} if local_files_only else {}
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **kwargs,
        )
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=resolve_wan_flow_shift(height, flow_shift),
        )
        return pipe
    elif spec.pipeline_class == "HunyuanVideoPipeline":
        from diffusers import HunyuanVideoPipeline

        cls = HunyuanVideoPipeline
    elif spec.pipeline_class == "CogVideoXPipeline":
        from diffusers import CogVideoXPipeline

        cls = CogVideoXPipeline
    else:
        raise ValueError(f"Unknown pipeline class: {spec.pipeline_class}")

    kwargs = {"torch_dtype": torch_dtype}
    if local_files_only:
        kwargs["local_files_only"] = True
    if spec.pipeline_class == "HunyuanVideoPipeline" and model_id == "tencent/HunyuanVideo":
        kwargs["revision"] = "refs/pr/18"
    return cls.from_pretrained(model_id, **kwargs)


def prepare_pipeline(pipe, device: str, cpu_offload: bool, vae_tiling: bool):
    if vae_tiling and hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    if cpu_offload:
        if not hasattr(pipe, "enable_model_cpu_offload"):
            raise RuntimeError("This pipeline does not expose enable_model_cpu_offload()")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)


def build_call_kwargs(
    args: argparse.Namespace,
    spec: ModelSpec,
    prompt: str,
    negative_prompt: str,
    generator,
    num_frames: int,
    fps: int,
) -> Dict[str, Any]:
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else spec.guidance_scale
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "output_type": spec.output_type,
    }
    if spec.key == "wan22-t2v-a14b":
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    if spec.family == "hunyuan_video":
        kwargs["true_cfg_scale"] = args.true_cfg_scale
    if spec.family == "cogvideox":
        kwargs["use_dynamic_cfg"] = False
    return kwargs


def make_output_file(args: argparse.Namespace, model: str, method: str, num_frames: int) -> Path:
    if args.output_file is not None:
        return args.output_file
    filename = f"seed{args.seed}_{args.height}x{args.width}_{num_frames}f.mp4"
    return args.output_dir / model / method / filename


def append_metrics(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def cuda_memory_gb(torch_module) -> Dict[str, float]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024**3),
        "cuda_peak_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024**3),
    }


def sync_if_cuda(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def run(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[MODEL_ALIASES[args.model]]
    fps = args.fps if args.fps is not None else spec.fps
    if args.num_frames is not None:
        num_frames = args.num_frames
    elif args.duration_seconds is not None:
        num_frames = default_num_frames(args.duration_seconds, fps)
    else:
        num_frames = spec.default_frames
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps

    configure_method_runtime_env(args.method)
    import sparsevideo

    method_config = sparsevideo.default_method_config(
        args.method, num_inference_steps=steps, model_family=spec.family, model_key=spec.key,
    )
    method_config.update(
        sparsevideo.normalize_method_config(args.method, parse_method_config(args))
    )
    if (
        args.method == "svoo"
        and method_config.get("use_dynamic_min_kc_ratio")
        and (
            not method_config.get("sparsity_csv_path")
            or method_config.get("sparsity_csv_path") == "sparsity_profiles/sparsity_results.csv"
        )
    ):
        method_config["sparsity_csv_path"] = default_svoo_sparsity_csv_path(spec)
    validate_method_config(args.method, method_config)
    from sparsevideo._runtime import optional_kernel_status, torch_runtime_status

    runtime_status = {
        "optional_kernels": optional_kernel_status(),
        "torch": torch_runtime_status(),
    }
    runtime_status["preflight"] = preflight_runtime(
        args.method,
        method_config,
        args.device,
        runtime_status,
        strict_kernels=args.strict_kernels,
    )
    model_id = resolve_model_id(spec, args.model_root, args.model_path)
    output_file = make_output_file(args, spec.key, args.method, num_frames)
    wan_flow_shift = resolve_wan_flow_shift(args.height, args.flow_shift) if spec.family == "wan" else None

    base_metrics: Dict[str, Any] = {
        "model": spec.key,
        "model_arg": args.model,
        "model_id": model_id,
        "method": args.method,
        "method_config": method_config,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
        "requested_duration_seconds": args.duration_seconds,
        "num_inference_steps": steps,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "strict_kernels": args.strict_kernels,
        "seed": args.seed,
        "output_file": str(output_file),
        "wan_flow_shift": wan_flow_shift,
        "runtime": runtime_status,
    }

    unsupported = args.method != "dense" and not spec.sparse_supported

    if args.dry_run:
        base_metrics.update(status="unsupported_dry_run" if unsupported else "dry_run")
        if unsupported:
            base_metrics["error"] = (
                f"{args.method} is not implemented for {spec.family}; "
                "only dense baseline is available."
            )
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 0

    if unsupported:
        base_metrics.update(
            status="unsupported",
            error=f"{args.method} is not implemented for {spec.family}; only dense baseline is available.",
        )
        append_metrics(args.metrics_file, base_metrics)
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 2

    if runtime_status["preflight"]["errors"]:
        base_metrics.update(
            status="failed",
            failed_stage="preflight",
            timings={},
            error_type="RuntimeError",
            error="; ".join(runtime_status["preflight"]["errors"]),
        )
        append_metrics(args.metrics_file, base_metrics)
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 1

    stage = "start"
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    handle = None

    try:
        stage = "import"
        import torch
        from diffusers.utils import export_to_video
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        seed_everything(torch, args.seed)
        try:
            torch.backends.cuda.preferred_linalg_library(backend="magma")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt = read_prompt(args)
        torch_dtype = parse_dtype(torch, args.dtype)

        stage = "load_pipeline"
        t0 = time.perf_counter()
        pipe = load_pipeline(
            spec,
            model_id,
            torch_dtype,
            args.local_files_only,
            height=args.height,
            flow_shift=args.flow_shift,
        )
        prepare_pipeline(pipe, args.device, args.cpu_offload, args.vae_tiling)
        sync_if_cuda(torch, args.device)
        timings["load_pipeline_sec"] = time.perf_counter() - t0

        stage = "apply_sparse_attention"
        t0 = time.perf_counter()
        if args.method == "svoo" and spec.family == "hunyuan_video":
            if method_config.get("context_length") is None:
                method_config["context_length"] = 256
            if method_config.get("prompt_length") is None:
                method_config["prompt_length"] = infer_hunyuan_prompt_length(
                    pipe, prompt, int(method_config["context_length"]),
                )
        handle = sparsevideo.apply_sparse_attention(pipe, method=args.method, config=method_config)
        sync_if_cuda(torch, args.device)
        timings["apply_sparse_attention_sec"] = time.perf_counter() - t0

        stage = "generate"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and output_file.exists():
            base_metrics.update(status="skipped_existing", timings=timings)
            append_metrics(args.metrics_file, base_metrics)
            print(json.dumps(base_metrics, indent=2, sort_keys=True))
            handle.restore()
            return 0

        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        call_kwargs = build_call_kwargs(
            args=args,
            spec=spec,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_frames=num_frames,
            fps=fps,
        )

        t0 = time.perf_counter()
        result = pipe(**call_kwargs)
        sync_if_cuda(torch, args.device)
        timings["generate_sec"] = time.perf_counter() - t0

        stage = "export_video"
        t0 = time.perf_counter()
        export_to_video(result.frames[0], str(output_file), fps=fps)
        timings["export_video_sec"] = time.perf_counter() - t0
        handle.restore()
        handle = None

        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="ok",
            timings=timings,
            seconds_per_frame=timings["generate_sec"] / max(num_frames, 1),
            **cuda_memory_gb(torch),
        )
        append_metrics(args.metrics_file, base_metrics)
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        restore_error = None
        if handle is not None:
            try:
                handle.restore()
            except Exception as restore_exc:
                restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="failed",
            failed_stage=stage,
            timings=timings,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if restore_error is not None:
            base_metrics["restore_error"] = restore_error
        append_metrics(args.metrics_file, base_metrics)
        traceback.print_exc()
        print(json.dumps(base_metrics, indent=2, sort_keys=True))
        return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
