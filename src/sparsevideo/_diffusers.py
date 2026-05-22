from __future__ import annotations

from typing import Any, Callable, List, Tuple


def discover_diffusers_model(pipe: Any, *, infer_model_key: Callable):
    transformer = getattr(pipe, "transformer", None)
    transformer_2 = getattr(pipe, "transformer_2", None)

    transformers = [t for t in (transformer, transformer_2) if t is not None]
    if not transformers:
        raise ValueError("Pipeline has no .transformer attribute")

    cls_name = type(transformers[0]).__name__

    if "Wan" in cls_name or "SkyReelsV2" in cls_name:
        model_type = "wan"
        attn_paths = _enumerate_wan(transformers)
    elif "HunyuanVideo" in cls_name:
        model_type = "hunyuan_video"
        attn_paths = _enumerate_hunyuan(transformers)
    elif "CogVideoX" in cls_name:
        model_type = "cogvideox"
        attn_paths = _enumerate_cogvideox(transformers)
    elif "Motif" in cls_name:
        raise ValueError(
            "MotifVideo is not available in this Diffusers installation; no "
            "processor-swap path can be verified without the target transformer class."
        )
    elif "LTXVideo2" in cls_name or "LTX2" in cls_name:
        raise ValueError(
            "LTX Video 2 is not available in this Diffusers installation; the "
            "plain LTXVideo processor must not be applied to an unverified audio/video transformer."
        )
    elif "LTXVideo" in cls_name:
        model_type = "ltx_video"
        attn_paths = _enumerate_ltx_video(transformers)
    elif "Allegro" in cls_name:
        model_type = "allegro"
        attn_paths = _enumerate_allegro(transformers)
    elif "Mochi" in cls_name:
        model_type = "mochi"
        attn_paths = _enumerate_mochi(transformers)
    elif "EasyAnimate" in cls_name:
        model_type = "easyanimate"
        attn_paths = _enumerate_easyanimate(transformers)
    elif "SanaVideo" in cls_name:
        raise ValueError(
            "SanaVideo uses Diffusers' SanaLinearAttnProcessor3_0 linear attention, "
            "not softmax QK^T V attention; current SparseVideo sparse-softmax methods "
            "are incompatible."
        )
    elif "Kandinsky5" in cls_name:
        raise ValueError(
            "Kandinsky5 exposes native sparse attention controls through transformer "
            "sparse_params/window parameters, so it is not a processor-swap target."
        )
    else:
        raise ValueError(f"Unsupported transformer type: {cls_name}")

    from ._model_info import ModelInfo

    return ModelInfo(
        model_type=model_type,
        transformers=transformers,
        model_key=infer_model_key(pipe, transformers, model_type),
        pipeline_backend="diffusers",
        _self_attn_paths=attn_paths,
    )


def _enumerate_wan(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for t_idx, transformer in enumerate(transformers):
        prefix = "transformer_2" if t_idx == 1 else "transformer"
        for i, block in enumerate(transformer.blocks):
            path = f"{prefix}.blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_hunyuan(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        if hasattr(transformer, "transformer_blocks"):
            for i, block in enumerate(transformer.transformer_blocks):
                path = f"transformer.transformer_blocks.{i}.attn"
                paths.append((path, block.attn))
        if hasattr(transformer, "single_transformer_blocks"):
            for i, block in enumerate(transformer.single_transformer_blocks):
                path = f"transformer.single_transformer_blocks.{i}.attn"
                paths.append((path, block.attn))
    return paths


def _enumerate_cogvideox(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_ltx_video(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_allegro(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_mochi(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_easyanimate(transformers) -> List[Tuple[str, Any]]:
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths
