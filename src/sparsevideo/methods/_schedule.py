from __future__ import annotations

from math import floor


def resolve_first_layers(first_layers_fp, total_layers):
    value = float(first_layers_fp or 0)
    if value <= 0:
        return 0
    if value < 1:
        return int(floor(value * total_layers))
    return min(total_layers, int(value))


def resolve_first_steps(first_times_fp, num_inference_steps):
    value = float(first_times_fp or 0)
    if value <= 0:
        return 0
    if value < 1:
        return int(floor(value * int(num_inference_steps)))
    return int(value)


def first_times_fp_requires_dense(first_times_fp, num_inference_steps, step, timestep=None):
    value = float(first_times_fp or 0)
    if value <= 0:
        return False
    if value < 1:
        return int(step) <= resolve_first_steps(value, num_inference_steps)
    timestep_value = _scalar_timestep(timestep)
    return timestep_value is not None and timestep_value > value


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
