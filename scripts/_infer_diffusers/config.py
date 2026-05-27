from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from sparsevideo._support import unsupported_method_model_reason

from .models import (
    FLASHOMNI_SPARSE_INFO_KEYS,
    STA_NATIVE_SEQ_SHAPES,
    STA_STRATEGY_SHAPES,
    STA_UNSUPPORTED_STRATEGY_MODELS,
    ModelSpec,
    is_allegro_pipeline,
    is_cogvideox_pipeline,
    is_easyanimate_pipeline,
    is_hunyuan_pipeline,
    is_ltx_pipeline,
    is_mochi_pipeline,
    sparsevideo_model_type,
    supports_sparsevideo_processor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def materialize_method_config_values(method: str, config: Dict[str, Any]) -> None:
    tensor_keys: tuple = ()
    if method == "flashomni":
        tensor_keys = FLASHOMNI_SPARSE_INFO_KEYS
    elif method == "spargeattn":
        tensor_keys = ("mask_id",)
    for key in tensor_keys:
        value = config.get(key)
        if isinstance(value, str):
            config[key] = load_torch_tensor_config_value(method, key, value)


def load_torch_tensor_config_value(method: str, key: str, value: str):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{method} {key} tensor path does not exist: {path}")
    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if torch.is_tensor(loaded):
        return loaded
    if isinstance(loaded, dict) and key in loaded and torch.is_tensor(loaded[key]):
        return loaded[key]
    raise TypeError(
        f"{method} {key} must load to a torch.Tensor, or to a dict containing "
        f"a torch.Tensor under key {key!r}; got {type(loaded).__name__} from {path}"
    )


def sync_flashomni_config_aliases(config: Dict[str, Any]) -> None:
    pairs = (
        ("threshold_q", "tau_q", 0.50),
        ("threshold_kv", "tau_kv", 0.05),
        ("fresh_threshold", "N", 6),
        ("max_order", "D", 1),
        ("saving_threshold_q_for_taylor", "S_q", 0.3),
    )
    for primary, legacy, default in pairs:
        primary_set = primary in config
        legacy_set = legacy in config
        if primary_set and legacy_set and config[primary] != config[legacy]:
            if config[legacy] == default:
                config[legacy] = config[primary]
                continue
            if config[primary] == default:
                config[primary] = config[legacy]
                continue
            raise ValueError(
                f"flashomni config keys {primary!r} and {legacy!r} refer to the same upstream "
                "setting but have different non-default values"
            )
        if primary_set:
            config[legacy] = config[primary]
        elif legacy_set:
            config[primary] = config[legacy]


def apply_flashomni_hunyuan_quality_defaults(
    spec: ModelSpec, config: Dict[str, Any], user_config: Dict[str, Any],
) -> None:
    if not is_hunyuan_pipeline(spec):
        return
    if config.get("sparse_pattern") != "paper_mmdit":
        return
    if not any(key in user_config for key in ("max_order", "D")):
        config["max_order"] = 0
        config["D"] = 0
    if "use_sparse_gemm" not in user_config:
        config["use_sparse_gemm"] = False


def validate_flashomni_hunyuan_quality_lock(config: Dict[str, Any], model_type: Optional[str]) -> None:
    if model_type != "hunyuan_video":
        return
    if config.get("sparse_pattern") != "paper_mmdit":
        return
    if not bool(config.get("use_sparse_gemm", False)):
        return
    # Sparse GEMM projection is kept for source audit, but not an allowed Hunyuan inference path:
    # matched 50-step tests showed visual shining artifacts and slower runtime.
    raise NotImplementedError(
        "flashomni Hunyuan paper_mmdit only supports the quality-safe inference path "
        "with use_sparse_gemm=false. Sparse GEMM projection (use_sparse_gemm=true) is "
        "currently disabled because measured Hunyuan runs showed quality degradation "
        "and performance regression. The code is retained for audit/future repair, "
        "but this path is not supported for inference."
    )


def default_svoo_sparsity_csv_path(spec: ModelSpec) -> str:
    sparsity_dir = SRC_ROOT / "sparsevideo" / "methods" / "svoo" / "sparsity_profiles"
    if spec.key == "hunyuan-i2v":
        sparsity_csv = sparsity_dir / "sparsity_hunyuan10_13B_i2v.csv"
    elif is_hunyuan_pipeline(spec):
        sparsity_csv = sparsity_dir / "sparsity_hunyuan10_13B_t2v.csv"
    elif spec.key == "wan22-i2v-a14b":
        sparsity_csv = sparsity_dir / "sparsity_wan22_A14B_i2v.csv"
    elif spec.key == "wan22-t2v-a14b":
        sparsity_csv = sparsity_dir / "sparsity_wan22_A14B_t2v.csv"
    elif spec.key == "wan21-i2v-14b":
        sparsity_csv = sparsity_dir / "sparsity_wan_14B_i2v.csv"
    elif spec.key == "wan21-t2v-14b":
        sparsity_csv = sparsity_dir / "sparsity_wan_14B_t2v.csv"
    elif spec.key == "wan21-t2v-1.3b":
        sparsity_csv = sparsity_dir / "sparsity_wan_1.3B_t2v.csv"
    else:
        raise ValueError(
            f"SVOO has no owned offline sparsity CSV for {spec.key}; "
            "leave use_dynamic_min_kc_ratio=false to skip the offline sparsity "
            "CSV stage and use online co-clustering with the fixed "
            "min_kc_ratio, or provide an explicit owned sparsity_csv_path."
        )
    return str(sparsity_csv)


def normalize_spargeattn_model_out_path(config: Dict[str, Any], output_file: Path) -> None:
    if not config.get("tune") and not config.get("model_out_path"):
        return
    if config.get("tune") and not config.get("model_out_path"):
        config["model_out_path"] = str(output_file.with_suffix(".spargeattn_state.pt"))
        return
    path = Path(str(config["model_out_path"])).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    config["model_out_path"] = str(path)


def validate_method_config(method: str, config: Dict[str, Any], model_type: Optional[str] = None) -> None:
    if method == "spargeattn":
        if config.get("mode", "topk") not in ("cdfthreshd", "topk", "block_sparse"):
            raise ValueError("spargeattn mode must be cdfthreshd, topk, or block_sparse")
        if config.get("tensor_layout", "HND") != "HND":
            raise ValueError("spargeattn SparseVideo processor uses tensor_layout=HND")
        if config.get("return_sparsity", False):
            raise NotImplementedError("spargeattn return_sparsity=true is not supported inside inference processors")
        if config.get("mode", "topk") == "block_sparse" and config.get("mask_id") is None:
            raise ValueError("spargeattn mode=block_sparse requires --method-config mask_id=<torch tensor path>")
        if config.get("pv_l1", 0.08) <= config.get("l1", 0.07):
            raise ValueError("spargeattn pv_l1 must be greater than l1")
        if config.get("sim_rule", "l1") not in ("l1", "cosine", "rmse"):
            raise ValueError("spargeattn sim_rule must be l1, cosine, or rmse")
        if not isinstance(config.get("rearrange_kwargs", {}), dict):
            raise TypeError("spargeattn rearrange_kwargs must be a JSON object")
        if config.get("model_out_path") and not config.get("tune"):
            path = Path(str(config["model_out_path"])).expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"spargeattn model_out_path does not exist: {path}")
    if method == "radial" and config.get("block_size") not in (64, 128):
        raise ValueError("radial block_size must be 64 or 128")
    if method == "flashomni":
        sync_flashomni_config_aliases(config)
        if config.get("implementation") != "upstream":
            raise ValueError("flashomni implementation must be upstream")
        if config.get("backend") not in ("auto", "fa2", "fa3"):
            raise ValueError("flashomni backend must be auto, fa2, or fa3")
        if config.get("pos_encoding_mode") not in ("NONE", "ROPE_LLAMA", "ALIBI"):
            raise ValueError("flashomni pos_encoding_mode must be NONE, ROPE_LLAMA, or ALIBI")
        if config.get("sparse_pattern") not in ("explicit", "global_random", "paper_mmdit", "local_qk_topk"):
            raise ValueError(
                "flashomni sparse_pattern must be explicit, global_random, paper_mmdit, or local_qk_topk"
            )
        for key in ("threshold_q", "threshold_kv", "saving_threshold_q_for_taylor"):
            value = float(config.get(key, 0.0))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"flashomni {key} must be in [0, 1]")
        if int(config.get("fresh_threshold", 1)) < 1:
            raise ValueError("flashomni fresh_threshold must be >= 1")
        if int(config.get("max_order", 0)) not in (0, 1, 2):
            raise ValueError("flashomni max_order must be 0, 1, or 2")
        if int(config.get("first_enhance", 0)) < 0:
            raise ValueError("flashomni first_enhance must be >= 0")
        if int(config.get("num_inference_steps", 1)) < 1:
            raise ValueError("flashomni num_inference_steps must be >= 1")
        if not isinstance(config.get("use_sparse_gemm", True), bool):
            raise ValueError("flashomni use_sparse_gemm must be a boolean")
        if config.get("sparse_pattern") == "global_random":
            config["sparse_block_size_for_q"] = int(config.get("sparse_size", 128))
            config["sparse_block_size_for_kv"] = int(config.get("sparse_size", 128))
        validate_flashomni_hunyuan_quality_lock(config, model_type)
        from .utils import is_torch_tensor
        bad = [key for key in FLASHOMNI_SPARSE_INFO_KEYS if config.get(key) is not None and not is_torch_tensor(config.get(key))]
        if bad:
            raise TypeError(
                "FlashOmni sparse-info inputs must be torch.Tensor values or CLI paths "
                f"to torch-saved tensors. Bad keys: {bad}"
            )
    if method == "draft" and not config.get("block_sparse_attention"):
        raise NotImplementedError(
            "draft block_sparse_attention=False disables the upstream sparse path; "
            "use --method dense for the dense baseline."
        )
    if method == "sta" and config.get("STA_mode", "STA_inference") not in ("STA_inference", "STA_searching"):
        raise NotImplementedError(
            "sta supports STA_inference in pipelines and STA_searching for mask calibration; "
            "use python -m sparsevideo.methods.sta.search tune for STA_tuning."
        )
    if method == "svoo":
        if config.get("use_dynamic_min_kc_ratio"):
            csv_path = config.get("sparsity_csv_path")
            if not csv_path:
                raise ValueError("svoo use_dynamic_min_kc_ratio requires sparsity_csv_path")
            path = Path(str(csv_path)).expanduser()
            if not path.is_absolute():
                path = REPO_ROOT / path
            resolved_path = path.resolve(strict=False)
            if "training_free" in path.parts or "training_free" in resolved_path.parts:
                raise RuntimeError(
                    "Refusing SVOO sparsity_csv_path inside training_free; "
                    "SparseVideo runtime sparsity profiles must live under src/sparsevideo."
                )
            path = resolved_path
            if not path.exists():
                raise FileNotFoundError(
                    f"svoo use_dynamic_min_kc_ratio requires an existing sparsity_csv_path: {path}"
                )
            config["sparsity_csv_path"] = str(path)
        for key in ("kmeans_iter_init", "kmeans_iter_step"):
            if key in config and int(config.get(key, 0)) <= 0:
                raise ValueError(f"svoo requires {key} > 0")


def normalize_seq_shape_for_warning(seq_shape) -> Optional[str]:
    if seq_shape is None:
        return None
    if isinstance(seq_shape, str):
        return seq_shape.lower()
    if isinstance(seq_shape, (list, tuple)) and len(seq_shape) == 3:
        return "x".join(str(int(part)) for part in seq_shape)
    return str(seq_shape)


def _parse_normalized_seq_shape(seq_shape: str) -> Optional[tuple]:
    parts = seq_shape.split("x")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def _normalize_int_triple(value) -> Optional[tuple]:
    if isinstance(value, str):
        value = value.replace("x", ",").split(",")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(part) for part in value)
        except (TypeError, ValueError):
            return None
    return None


def _sta_mask_strategy_shape(path) -> tuple:
    strategy_path = Path(str(path)).expanduser()
    if not strategy_path.is_absolute():
        strategy_path = (REPO_ROOT / strategy_path).resolve()
    if "training_free" in strategy_path.parts:
        raise RuntimeError(
            "Refusing STA mask_strategy_file_path inside training_free; "
            "SparseVideo runtime mask strategies must live under src/sparsevideo."
        )
    with strategy_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"STA mask strategy must be a non-empty JSON object: {strategy_path}")
    timesteps, layers, heads = [], [], []
    for key in data:
        t_idx, layer_idx, head_idx = (int(part) for part in str(key).split("_"))
        timesteps.append(t_idx)
        layers.append(layer_idx)
        heads.append(head_idx)
    return max(timesteps) + 1, max(layers) + 1, max(heads) + 1


def _radial_estimated_latent_shape(spec: ModelSpec, height: int, width: int, num_frames: int) -> tuple:
    if is_cogvideox_pipeline(spec):
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
        if spec.key == "cogvideox-i2v":
            latent_t = ((num_frames - 1) // 4 + 1) * 2
        return latent_t, latent_h // 2, latent_w // 2
    if is_ltx_pipeline(spec):
        return (num_frames - 1) // 8 + 1, height // 32, width // 32
    if is_allegro_pipeline(spec):
        latent_t = ((num_frames + 3) // 4) if num_frames % 2 == 0 else ((num_frames - 1 + 3) // 4 + 1)
        return latent_t, height // 16, width // 16
    if is_mochi_pipeline(spec):
        return (num_frames - 1) // 6 + 1, height // 16, width // 16
    if is_easyanimate_pipeline(spec):
        return (num_frames - 1) // 4 + 1, height // 16, width // 16
    return (num_frames - 1) // 4 + 1, height // 16, width // 16


def _draft_estimated_latent_shape(spec: ModelSpec, height: int, width: int, num_frames: int) -> tuple:
    if is_cogvideox_pipeline(spec):
        return (num_frames - 1) // 4 + 1, height // 16, width // 16
    if spec.pipeline_class == "WanAnimatePipeline":
        segment_frames = _wananimate_effective_segment_frame_length(num_frames)
        # WanAnimate attends over segment latents plus one prepended reference latent frame.
        return (segment_frames - 1) // 4 + 2, height // 16, width // 16
    return _radial_estimated_latent_shape(spec, height, width, num_frames)


def _wananimate_effective_segment_frame_length(num_frames: int) -> int:
    segment_frame_length = int(num_frames)
    if segment_frame_length % 4 != 1:
        segment_frame_length = segment_frame_length // 4 * 4 + 1
    return max(segment_frame_length, 1)


def apply_draft_runtime_layout_defaults(
    spec: ModelSpec, height: int, width: int, num_frames: int,
    config: Dict[str, Any], user_config: Dict[str, Any],
) -> None:
    latent_t, latent_h, latent_w = _draft_estimated_latent_shape(spec, height, width, num_frames)
    defaults = {"latent_h": latent_h, "latent_w": latent_w, "visual_len": latent_t * latent_h * latent_w}
    for key, value in defaults.items():
        if key not in user_config and config.get(key) is None:
            config[key] = value


def draft_layout_error(
    spec: ModelSpec, height: int, width: int, num_frames: int, config: Dict[str, Any],
) -> Optional[str]:
    if not config.get("block_sparse_attention", True):
        return None
    latent_t, latent_h, latent_w = _draft_estimated_latent_shape(spec, height, width, num_frames)
    pool_h = int(config.get("pool_h", 8))
    pool_w = int(config.get("pool_w", 16))
    video_len = latent_t * latent_h * latent_w
    if pool_h * pool_w != 128:
        return (
            "draft MIT Block-Sparse-Attention backend requires pool_h * pool_w == 128 "
            f"to form 128-token blocks; got pool_h={pool_h}, pool_w={pool_w}."
        )
    for key, actual in (("latent_h", latent_h), ("latent_w", latent_w), ("visual_len", video_len)):
        configured = config.get(key)
        if configured is not None and int(configured) != int(actual):
            return (
                f"draft {key} config expects {int(configured)}, "
                f"but the requested layout has {key}={int(actual)}."
            )
    configured_text_len = config.get("text_len")
    expected_text_len = None
    model_type = sparsevideo_model_type(spec)
    if is_hunyuan_pipeline(spec) and spec.pipeline_class != "HunyuanVideoImageToVideoPipeline":
        expected_text_len = 256
    elif model_type in ("wan", "ltx_video", "allegro"):
        expected_text_len = 0
    if (
        configured_text_len is not None
        and expected_text_len is not None
        and int(configured_text_len) != expected_text_len
    ):
        return (
            f"draft text_len config expects {int(configured_text_len)}, "
            f"but {spec.pipeline_class} expects text_len={expected_text_len}."
        )
    if not supports_sparsevideo_processor(spec):
        return f"draft is not implemented for {spec.pipeline_class}."
    return None


def radial_flashinfer_layout_warning(
    spec: ModelSpec, height: int, width: int, num_frames: int, config: Dict[str, Any],
) -> Optional[str]:
    if not supports_sparsevideo_processor(spec):
        return f"radial is not implemented for {spec.pipeline_class}."
    if height % 16 != 0 or width % 16 != 0:
        return (
            "radial FlashInfer path expects height and width divisible by 16 "
            f"for video patch tokens; got {height}x{width}."
        )
    return None


def sta_layout_preflight_messages(
    spec: ModelSpec, height: int, width: int, num_frames: int, config: Dict[str, Any],
) -> Dict[str, list]:
    errors: list = []
    warnings: list = []
    if not supports_sparsevideo_processor(spec):
        return {"errors": [f"sta is not implemented for {spec.pipeline_class}."], "warnings": warnings}
    model_reason = unsupported_method_model_reason("sta", spec.key)
    if model_reason is not None:
        return {"errors": [model_reason], "warnings": warnings}
    if height % 16 != 0 or width % 16 != 0:
        return {
            "errors": [
                "sta FastVideo path expects height and width divisible by 16 "
                f"for video patch tokens; got {height}x{width}."
            ],
            "warnings": warnings,
        }

    sta_mode = config.get("STA_mode", "STA_inference")
    mask_strategy_path = config.get("mask_strategy_file_path")
    if mask_strategy_path is None and sta_mode != "STA_searching":
        message = (
            "sta STA_inference has no tuned mask_strategy_file_path for this model. "
            "SparseVideo will use the configured window_size for every layer/head as local STA support; "
            "this is not a tuned FastVideo sparse strategy."
        )
        warnings.append(message)
    elif mask_strategy_path is not None:
        try:
            strategy_shape = _sta_mask_strategy_shape(mask_strategy_path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"sta could not read mask_strategy_file_path={mask_strategy_path!r}: {exc}")
        else:
            expected_shape = STA_STRATEGY_SHAPES.get(spec.key)
            if expected_shape is None and spec.key in STA_UNSUPPORTED_STRATEGY_MODELS:
                message = (
                    f"sta has no sparse inference mask strategy for {spec.key}. "
                    f"{STA_UNSUPPORTED_STRATEGY_MODELS[spec.key]} "
                    f"Provided strategy has shape steps/layers/heads={strategy_shape}."
                )
                errors.append(message)
            elif expected_shape is not None and strategy_shape != expected_shape:
                message = (
                    f"sta mask_strategy_file_path shape steps/layers/heads={strategy_shape} does not match "
                    f"the expected {spec.key} strategy shape {expected_shape}."
                )
                errors.append(message)

    latent_t, latent_h, latent_w = _radial_estimated_latent_shape(spec, height, width, num_frames)
    latent_shape = (latent_t, latent_h, latent_w)
    latent_seq_shape = f"{latent_t}x{latent_h}x{latent_w}"

    seq_shape = normalize_seq_shape_for_warning(config.get("seq_shape"))
    if seq_shape is not None:
        parsed_seq_shape = _parse_normalized_seq_shape(seq_shape)
        if parsed_seq_shape != latent_shape:
            errors.append(
                f"sta seq_shape={seq_shape} does not match the current latent layout "
                f"{latent_seq_shape} from {num_frames} frames at {width}x{height}; "
                "runtime would fail before reaching the FastVideo STA path."
            )

    tile_size = _normalize_int_triple(config.get("tile_size", [6, 8, 8])) or (6, 8, 8)
    padded_shape = tuple(
        ((dim + tile - 1) // tile) * tile
        for dim, tile in zip(latent_shape, tile_size)
    )
    padded_seq_shape = "x".join(str(part) for part in padded_shape)
    if padded_seq_shape not in STA_NATIVE_SEQ_SHAPES:
        if sparsevideo_model_type(spec) not in ("wan", "hunyuan_video"):
            warnings.append(
                f"sta will use generalized STA A100 block-sparse CUDA for tile-padded canvas {padded_seq_shape} "
                f"from latent layout {latent_seq_shape}."
            )
            return {"errors": errors, "warnings": warnings}
        message = (
            "sta FastVideo native path only covers latent layouts "
            f"{sorted(STA_NATIVE_SEQ_SHAPES)}. Current latent layout is {latent_seq_shape} "
            f"and tile-padded canvas is {padded_seq_shape}, so SparseVideo will use the "
            "generalized STA A100 block-sparse CUDA path with partial-border valid-token masking for this target shape. "
            "Do not treat this as native FastVideo parity unless the requested target has matching "
            "quality and speed evidence."
        )
        warnings.append(message)

    return {"errors": errors, "warnings": warnings}


def model_quality_warnings(spec: ModelSpec, height: int, width: int) -> list:
    warnings: list = []
    if spec.key == "wan21-t2v-1.3b" and int(height) >= 720:
        warnings.append(
            "Wan2.1 T2V 1.3B is a 480P model in the Wan README; 720P is explicitly less "
            "stable and should be treated as target-shape stress evidence rather than a standalone "
            "model-quality baseline."
        )
    return warnings


def model_shape_preflight_errors(spec: ModelSpec, height: int, width: int) -> list:
    if is_allegro_pipeline(spec) and (int(height) % 16 != 0 or int(width) % 16 != 0):
        return [
            "allegro requires height and width divisible by 16 for the VAE/transformer "
            f"patch grid; got {int(height)}x{int(width)}. Use a nearby shape such as "
            "--height 352 --width 640 or --height 320 --width 576."
        ]
    return []


def default_num_frames(duration_seconds: float, fps: int) -> int:
    frames = max(1, int(round(duration_seconds * fps)) + 1)
    remainder = (frames - 1) % 4
    if remainder:
        frames += 4 - remainder
    return frames
