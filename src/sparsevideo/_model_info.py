from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple


@dataclass
class ModelInfo:
    model_type: str
    transformers: List[Any]
    model_key: Optional[str] = None
    _self_attn_paths: List[Tuple[str, Any]] = field(default_factory=list)

    @property
    def num_self_attn_layers(self) -> int:
        return len(self._self_attn_paths)

    def iter_self_attn_modules(self) -> Iterator[Tuple[str, Any]]:
        yield from self._self_attn_paths

    def get_attn_module(self, path: str) -> Any:
        for p, m in self._self_attn_paths:
            if p == path:
                return m
        raise KeyError(f"No attn module at path {path}")


def discover_model(pipe) -> ModelInfo:
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

    return ModelInfo(
        model_type=model_type,
        transformers=transformers,
        model_key=_infer_model_key(pipe, transformers, model_type),
        _self_attn_paths=attn_paths,
    )


def _infer_model_key(pipe, transformers, model_type: str) -> Optional[str]:
    if model_type == "hunyuan_video":
        cls_name = type(pipe).__name__
        if "ImageToVideo" in cls_name or "I2V" in cls_name:
            return "hunyuan-i2v"
        return "hunyuan_video"
    if model_type == "cogvideox":
        cls_name = type(pipe).__name__
        return "cogvideox-i2v" if "ImageToVideo" in cls_name or "I2V" in cls_name else "cogvideox-t2v"
    if model_type == "ltx_video":
        cls_name = type(pipe).__name__
        return "ltx-video-i2v" if "ImageToVideo" in cls_name or "I2V" in cls_name else "ltx-video"
    if model_type == "allegro":
        return "allegro"
    if model_type == "mochi":
        return "mochi-1"
    if model_type == "easyanimate":
        return "easyanimate-v5-t2v-12b"
    if model_type != "wan":
        return None

    objects = [pipe, *transformers]
    text = " ".join(_iter_model_identity_strings(objects)).lower()
    class_text = " ".join(type(obj).__name__ for obj in objects).lower()
    text = f"{class_text} {text}"
    text = text.replace("_", "-")
    is_i2v = "i2v" in text or "ImageToVideo" in type(pipe).__name__

    if "wananimate" in text or "wan-animate" in text:
        return "wan22-animate-14b"
    if "vace" in text:
        return "wan21-vace-14b" if "14b" in text else "wan21-vace-1.3b"
    if len(transformers) > 1:
        return "wan22-i2v-a14b" if is_i2v else "wan22-t2v-a14b"
    if "wan2.2" in text or "wan22" in text or "wan-2.2" in text:
        if "a14b" in text or "14b" in text:
            return "wan22-i2v-a14b" if is_i2v else "wan22-t2v-a14b"
    if "skyreels" in text:
        return "skyreels-v2-i2v-14b" if is_i2v else "skyreels-v2-t2v-14b"
    if "1.3b" in text or "1-3b" in text:
        return "wan21-t2v-1.3b"
    if "14b" in text:
        return "wan21-i2v-14b" if is_i2v else "wan21-t2v-14b"
    return None


def _iter_model_identity_strings(objects) -> Iterator[str]:
    fields = (
        "_name_or_path",
        "name_or_path",
        "model_name",
        "model_id",
        "pretrained_model_name_or_path",
        "repo_id",
    )
    for obj in objects:
        for field_name in fields:
            value = getattr(obj, field_name, None)
            if isinstance(value, str):
                yield value
        config = getattr(obj, "config", None)
        if config is None:
            continue
        for field_name in fields:
            value = _config_value(config, field_name)
            if isinstance(value, str):
                yield value


def _config_value(config, field_name: str):
    if isinstance(config, dict):
        return config.get(field_name)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            value = getter(field_name)
        except Exception:
            value = None
        if value is not None:
            return value
    return getattr(config, field_name, None)


def _enumerate_wan(transformers):
    paths = []
    for t_idx, transformer in enumerate(transformers):
        prefix = "transformer_2" if t_idx == 1 else "transformer"
        for i, block in enumerate(transformer.blocks):
            path = f"{prefix}.blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_hunyuan(transformers):
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


def _enumerate_cogvideox(transformers):
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_ltx_video(transformers):
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_allegro(transformers):
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_mochi(transformers):
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths


def _enumerate_easyanimate(transformers):
    paths = []
    for transformer in transformers:
        for i, block in enumerate(transformer.transformer_blocks):
            path = f"transformer.transformer_blocks.{i}.attn1"
            paths.append((path, block.attn1))
    return paths
