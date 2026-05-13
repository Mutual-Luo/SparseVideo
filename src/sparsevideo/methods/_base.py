from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .._model_info import ModelInfo
    from .._step_tracker import StepTracker


class SparseMethod(ABC):
    CONFIG_DEFAULTS: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any], model_info: "ModelInfo"):
        unknown = set(config) - set(self.CONFIG_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown config keys for {self.__class__.__name__}: {unknown}. "
                f"Valid keys: {list(self.CONFIG_DEFAULTS)}"
            )
        self.config = {**self.CONFIG_DEFAULTS, **config}
        self.model_info = model_info

    @abstractmethod
    def create_processor(
        self,
        layer_idx: int,
        total_layers: int,
        original_processor: Any,
        step_tracker: "StepTracker",
    ) -> Any:
        ...
