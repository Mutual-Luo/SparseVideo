from __future__ import annotations

from math import floor


DENSE_WARMUP_CONFIG_DEFAULTS = {
    "dense_warmup_step_ratio": 0.1,
    "dense_warmup_layer_ratio": 0.03,
}


def configured_dense_warmup_layer_count(config, total_layers):
    if config.get("dense_warmup_layer_ratio") is not None:
        ratio = _validate_ratio("dense_warmup_layer_ratio", config["dense_warmup_layer_ratio"])
        if ratio <= 0:
            return 0
        return max(1, int(floor(ratio * int(total_layers))))
    return 0


def configured_dense_warmup_requires_dense(config, num_inference_steps, step, timestep=None):
    if config.get("dense_warmup_step_ratio") is not None:
        ratio = _validate_ratio("dense_warmup_step_ratio", config["dense_warmup_step_ratio"])
        if ratio >= 1.0:
            return True
        if step is None or num_inference_steps is None:
            return False
        return int(step) <= int(floor(ratio * int(num_inference_steps)))
    return False


def runtime_num_inference_steps(step_tracker):
    getter = getattr(step_tracker, "num_inference_steps", None)
    if callable(getter):
        return getter()
    return None


def scheduler_timestep_from_tracker(step_tracker, kwargs):
    raw_timestep = kwargs.get("timestep")
    tracked_timestep = getattr(step_tracker, "timestep", None)
    if tracked_timestep is None:
        return raw_timestep
    if raw_timestep is None:
        return tracked_timestep
    if isinstance(raw_timestep, (int, float)):
        return float(raw_timestep)
    if getattr(step_tracker, "step", 0) > 0 and float(tracked_timestep) != 0.0:
        return tracked_timestep
    return raw_timestep


def _scalar_timestep(timestep):
    if timestep is None:
        return None
    if isinstance(timestep, (int, float)):
        return float(timestep)
    if hasattr(timestep, "numel") and timestep.numel() > 0:
        return float(timestep.detach().flatten()[0].item())
    return None


def _validate_ratio(name, value):
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return ratio
