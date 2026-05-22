from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple


@dataclass
class ModelInfo:
    model_type: str
    transformers: List[Any]
    model_key: Optional[str] = None
    pipeline_backend: str = "diffusers"
    unpatched_attention_paths: List[str] = field(default_factory=list)
    pipeline_notes: List[str] = field(default_factory=list)
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
    from ._diffusers import discover_diffusers_model
    from ._diffsynth import discover_diffsynth_model

    diffsynth_info = discover_diffsynth_model(pipe, infer_model_key=_infer_model_key)
    if diffsynth_info is not None:
        return diffsynth_info

    return discover_diffusers_model(pipe, infer_model_key=_infer_model_key)


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
        if "LTX2" in cls_name:
            return "ltx2"
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

    if "mova" in text:
        return "mova-720p"
    if "longcat" in text:
        return "longcat-video"
    if "krea" in text:
        return "krea-realtime-video"
    if "video-as-prompt" in text or "vap" in text:
        return "video-as-prompt-wan21-14b"
    if "speedcontrol" in text or "speed-control" in text:
        return "wan21-speedcontrol-1.3b"
    fun_key = _infer_wan_fun_model_key(text)
    if fun_key is not None:
        return fun_key
    if "s2v" in text or "speech-to-video" in text:
        return "wan22-s2v-14b"
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


def _infer_wan_fun_model_key(text: str) -> Optional[str]:
    if "wan2.2-fun" in text or "wan22-fun" in text:
        if "control-camera" in text or "control-camera-a14b" in text:
            return "wan22-fun-a14b-control-camera"
        if "control" in text:
            return "wan22-fun-a14b-control"
        return None
    if "wan2.1-fun" not in text and "wan21-fun" not in text:
        return None

    version = "v11" if "v1.1" in text or "v11" in text else None
    size = "14b" if "14b" in text else "1.3b"
    if "control-camera" in text:
        return f"wan21-fun-{version}-{size}-control-camera" if version else None
    if "control" in text:
        return f"wan21-fun-{version}-{size}-control" if version else f"wan21-fun-{size}-control"
    if "inp" in text:
        return f"wan21-fun-{size}-inp"
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
