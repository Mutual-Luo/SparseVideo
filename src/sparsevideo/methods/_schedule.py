from __future__ import annotations

from math import floor


DENSE_WARMUP_CONFIG_DEFAULTS = {
    "dense_warmup_step_ratio": 0.1,
    "dense_warmup_layer_ratio": 0.03,
}


class WarmupNotifier:
    """Prints a one-time message when warmup begins and when sparse stage begins."""

    def __init__(self, method_name: str):
        self._method_name = method_name
        self._warmup_announced = False
        self._sparse_announced = False

    def announce_config(self, config: dict) -> None:
        """Print warmup summary immediately at apply time if num_inference_steps is in config."""
        if self._warmup_announced:
            return
        num_inference_steps = config.get("num_inference_steps")
        ratio = config.get("dense_warmup_step_ratio")
        if num_inference_steps is None or ratio is None:
            return
        warmup_steps = int(floor(float(ratio) * int(num_inference_steps)))
        if warmup_steps <= 0:
            return
        self._warmup_announced = True
        print(f"[sparsevideo:{self._method_name}] Dense warmup: steps 1-{warmup_steps}/{num_inference_steps} ({warmup_steps / num_inference_steps:.0%})")

    def notify(self, is_warmup: bool, step, num_inference_steps, warmup_steps: int) -> None:
        if is_warmup and not self._warmup_announced:
            self._warmup_announced = True
            print(f"[sparsevideo:{self._method_name}] Dense warmup: steps 1-{warmup_steps}/{num_inference_steps} ({warmup_steps / num_inference_steps:.0%})")
        elif not is_warmup and not self._sparse_announced:
            self._sparse_announced = True
            print(f"[sparsevideo:{self._method_name}] Sparse attention active from step {step}/{num_inference_steps}")


def configured_dense_warmup_layer_count(config, total_layers):
    if config.get("dense_warmup_layer_ratio") is not None:
        ratio = _validate_ratio("dense_warmup_layer_ratio", config["dense_warmup_layer_ratio"])
        if ratio <= 0:
            return 0
        return max(1, int(floor(ratio * int(total_layers))))
    return 0


def configured_dense_warmup_requires_dense(config, num_inference_steps, step, timestep=None, notifier=None):
    if config.get("dense_warmup_step_ratio") is not None:
        ratio = _validate_ratio("dense_warmup_step_ratio", config["dense_warmup_step_ratio"])
        if ratio >= 1.0:
            if notifier is not None and num_inference_steps is not None:
                notifier.notify(True, step, num_inference_steps, int(num_inference_steps))
            return True
        if step is None or num_inference_steps is None:
            return False
        warmup_steps = int(floor(ratio * int(num_inference_steps)))
        is_warmup = int(step) <= warmup_steps
        if notifier is not None:
            notifier.notify(is_warmup, step, int(num_inference_steps), warmup_steps)
        return is_warmup
    return False


def runtime_num_inference_steps(step_tracker):
    getter = getattr(step_tracker, "num_inference_steps", None)
    if callable(getter):
        return getter()
    return None


def runtime_or_config_num_inference_steps(step_tracker, config):
    runtime_steps = runtime_num_inference_steps(step_tracker)
    if runtime_steps is not None:
        return runtime_steps

    config_steps = config.get("num_inference_steps")
    if config_steps is None:
        return None
    try:
        config_steps = int(config_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_inference_steps must be an integer") from exc
    if config_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    return config_steps


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
