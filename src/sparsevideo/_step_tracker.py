from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ._model_info import ModelInfo


class StepTracker:
    """Tracks the current denoising step via a transformer forward_pre_hook.

    Processors read ``step_tracker.step`` and ``step_tracker.timestep`` to
    decide whether to use sparse or dense attention at each denoising step.
    """

    def __init__(self, model_type: str):
        self.step: int = 0
        self.timestep: float = 0.0
        self._model_type = model_type
        self._prev_timestep: Optional[float] = None

    def _hook(self, module, args, kwargs=None):
        t_val = self._extract_timestep(args, kwargs or {})
        if t_val is None:
            return

        if self._prev_timestep is None or t_val != self._prev_timestep:
            self._prev_timestep = t_val
            self.timestep = t_val
            self.step += 1

    def _extract_timestep(self, args, kwargs) -> Optional[float]:
        for key in ("timestep", "timesteps", "t"):
            if key in kwargs:
                value = self._to_float(kwargs[key])
                if value is not None:
                    return value

        if self._model_type == "wan":
            idx = 1
        elif self._model_type == "hunyuan_video":
            idx = 1
        elif self._model_type == "cogvideox":
            idx = 2
        else:
            idx = 1

        if idx < len(args):
            return self._to_float(args[idx])
        return None

    def _to_float(self, candidate) -> Optional[float]:
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, torch.Tensor) and candidate.numel() > 0:
            # Some Diffusers models can expand one scheduler timestep over many
            # tokens. The denoising step still follows the first scalar value.
            return float(candidate.detach().flatten()[0].item())
        return None

    def reset(self):
        self.step = 0
        self.timestep = 0.0
        self._prev_timestep = None


def install_step_tracker(model_info: ModelInfo):
    tracker = StepTracker(model_type=model_info.model_type)
    hooks = []
    for transformer in model_info.transformers:
        h = transformer.register_forward_pre_hook(tracker._hook, with_kwargs=True)
        hooks.append(h)
    return tracker, hooks
