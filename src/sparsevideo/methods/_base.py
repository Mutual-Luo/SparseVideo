from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .._model_info import ModelInfo
    from .._step_tracker import StepTracker


class SparseMethod(ABC):
    CONFIG_DEFAULTS: Dict[str, Any] = {}
    CONFIG_ALIASES: Dict[str, str] = {}
    CONFIG_COMPAT_KEYS: set[str] = set()

    @classmethod
    def normalize_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized_config: Dict[str, Any] = {}
        for key, value in config.items():
            canonical_key = cls.CONFIG_ALIASES.get(key, key)
            if canonical_key in normalized_config:
                raise ValueError(
                    f"Config key {key!r} conflicts with {canonical_key!r} "
                    f"for {cls.__name__}"
                )
            normalized_config[canonical_key] = value

        unknown = (
            set(normalized_config)
            - set(cls.CONFIG_DEFAULTS)
            - set(cls.CONFIG_COMPAT_KEYS)
        )
        if unknown:
            valid_keys = [*cls.CONFIG_DEFAULTS, *sorted(cls.CONFIG_COMPAT_KEYS)]
            raise ValueError(
                f"Unknown config keys for {cls.__name__}: {unknown}. "
                f"Valid keys: {valid_keys}"
            )
        return normalized_config

    @classmethod
    def default_config(cls, **context: Any) -> Dict[str, Any]:
        config = dict(cls.CONFIG_DEFAULTS)
        num_inference_steps = context.get("num_inference_steps")
        if num_inference_steps is not None and "num_inference_steps" in config:
            config["num_inference_steps"] = num_inference_steps
        return config

    def __init__(self, config: Dict[str, Any], model_info: "ModelInfo"):
        from ._schedule import WarmupNotifier
        normalized_config = self.normalize_config(config)
        self.model_info = model_info
        self.config = {
            **self.default_config(
                model_family=model_info.model_type,
                model_key=getattr(model_info, "model_key", None),
            ),
            **normalized_config,
        }
        self._ensure_runtime_stats()
        self.warmup_notifier = WarmupNotifier(self.__class__.__name__.replace("Method", "").lower())

    def _ensure_runtime_stats(self) -> Dict[str, Any]:
        if not hasattr(self, "_runtime_stats"):
            self._runtime_stats = {
                "total_calls": 0,
                "dispatch_counts": {},
                "backend_counts": {},
                "last_dispatch": None,
            }
        return self._runtime_stats

    def record_runtime_dispatch(
        self,
        dispatch: str,
        *,
        backend: str | None = None,
        layer_idx: int | None = None,
        step: int | None = None,
    ) -> None:
        stats = self._ensure_runtime_stats()
        stats["total_calls"] += 1
        stats["dispatch_counts"][dispatch] = stats["dispatch_counts"].get(dispatch, 0) + 1
        if backend is not None:
            stats["backend_counts"][backend] = stats["backend_counts"].get(backend, 0) + 1
        event = {"dispatch": dispatch, "backend": backend, "layer_idx": layer_idx, "step": step}
        stats["last_dispatch"] = {key: value for key, value in event.items() if value is not None}

    def runtime_summary(self) -> Dict[str, Any]:
        return copy.deepcopy(self._ensure_runtime_stats())

    def install_model_patches(self, model_info: "ModelInfo"):
        """Install model-level forward patches for kernel acceleration.

        Default: installs Wan fast-block patch (Triton LayerNorm + modulate)
        for Wan models, Hunyuan fast-block patch for Hunyuan models.
        Subclasses may override to add additional patches.
        """
        if getattr(model_info, "pipeline_backend", "diffusers") != "diffusers":
            return []
        if model_info.model_type == "wan":
            from ..processors.wan_fast_block import install_wan_fast_block_patch

            return [install_wan_fast_block_patch()]
        if model_info.model_type == "hunyuan_video":
            from ..processors.hunyuan_fast_block import install_hunyuan_fast_block_patch

            return [install_hunyuan_fast_block_patch()]
        return []

    @abstractmethod
    def create_processor(
        self,
        layer_idx: int,
        total_layers: int,
        original_processor: Any,
        step_tracker: "StepTracker",
    ) -> Any:
        ...
