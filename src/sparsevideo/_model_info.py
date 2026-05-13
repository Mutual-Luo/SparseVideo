from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple


@dataclass
class ModelInfo:
    model_type: str
    transformers: List[Any]
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

    if "Wan" in cls_name:
        model_type = "wan"
        attn_paths = _enumerate_wan(transformers)
    elif "HunyuanVideo" in cls_name:
        model_type = "hunyuan_video"
        attn_paths = _enumerate_hunyuan(transformers)
    elif "CogVideoX" in cls_name:
        model_type = "cogvideox"
        attn_paths = _enumerate_cogvideox(transformers)
    else:
        raise ValueError(f"Unsupported transformer type: {cls_name}")

    return ModelInfo(
        model_type=model_type,
        transformers=transformers,
        _self_attn_paths=attn_paths,
    )


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
