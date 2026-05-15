from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .._model_info import ModelInfo
    from .._step_tracker import StepTracker


class SparseMethod(ABC):
    CONFIG_DEFAULTS: Dict[str, Any] = {}
    CONFIG_ALIASES: Dict[str, str] = {}

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

        unknown = set(normalized_config) - set(cls.CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown config keys for {cls.__name__}: {unknown}. "
                f"Valid keys: {list(cls.CONFIG_DEFAULTS)}"
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
        normalized_config = self.normalize_config(config)
        self.model_info = model_info
        self.config = {
            **self.default_config(model_family=model_info.model_type),
            **normalized_config,
        }

    @abstractmethod
    def create_processor(
        self,
        layer_idx: int,
        total_layers: int,
        original_processor: Any,
        step_tracker: "StepTracker",
    ) -> Any:
        ...
