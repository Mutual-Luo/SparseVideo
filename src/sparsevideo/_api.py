from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

from ._registry import get_method_class, list_methods as _list_methods
from ._model_info import ModelInfo, discover_model
from ._step_tracker import install_step_tracker


_SCHEDULER_FIRST_TIMES_FP_METHODS = {"svg1", "svg2", "svoo"}


class SparseAttentionHandle:
    def __init__(
        self,
        model_info: ModelInfo,
        original_processors: Dict[str, Any],
        step_tracker_hooks: list,
        restore_callbacks: Optional[list] = None,
        method_instance: Any = None,
    ):
        self._model_info = model_info
        self._original_processors = original_processors
        self._step_tracker_hooks = step_tracker_hooks
        self._restore_callbacks = restore_callbacks or []
        self._method_instance = method_instance
        self._restored = False

    def summary(self) -> Dict[str, Any]:
        processor_classes = {}
        for path in sorted(self._original_processors):
            try:
                processor = self._model_info.get_attn_module(path).get_processor()
                processor_classes[path] = f"{type(processor).__module__}.{type(processor).__name__}"
            except Exception as exc:
                processor_classes[path] = f"<unavailable:{type(exc).__name__}>"

        method_class = None
        method_runtime = None
        if self._method_instance is not None:
            method_class = type(self._method_instance).__name__
            runtime_summary = getattr(self._method_instance, "runtime_summary", None)
            if callable(runtime_summary):
                method_runtime = runtime_summary()

        return {
            "model_type": self._model_info.model_type,
            "model_key": self._model_info.model_key,
            "num_self_attn_layers": self._model_info.num_self_attn_layers,
            "installed_processor_count": len(self._original_processors),
            "installed_processor_paths": sorted(self._original_processors),
            "current_processor_classes": processor_classes,
            "step_tracker_hook_count": len(self._step_tracker_hooks),
            "restore_callback_count": len(self._restore_callbacks),
            "method_class": method_class,
            "method_runtime": method_runtime,
            "restored": self._restored,
        }

    def restore(self):
        if self._restored:
            return
        for path, proc in self._original_processors.items():
            attn_module = self._model_info.get_attn_module(path)
            attn_module.set_processor(proc)
        for hook in self._step_tracker_hooks:
            hook.remove()
        for restore_callback in reversed(self._restore_callbacks):
            restore_callback()
        self._restored = True


def restore_sparse_attention(handle: SparseAttentionHandle) -> None:
    handle.restore()


def apply_sparse_attention(
    pipe,
    method: str = "dense",
    config: Optional[Dict[str, Any]] = None,
) -> SparseAttentionHandle:
    if config is None:
        config = {}

    model_info = discover_model(pipe)
    method_cls = get_method_class(method)
    method_instance = method_cls(config=config, model_info=model_info)
    _resolve_scheduler_first_times_fp(pipe, method, method_instance.config)

    if method == "dense":
        return SparseAttentionHandle(
            model_info=model_info,
            original_processors={},
            step_tracker_hooks=[],
            restore_callbacks=[],
            method_instance=method_instance,
        )

    original_processors = {}
    for path, attn_module in model_info.iter_self_attn_modules():
        original_processors[path] = attn_module.get_processor()

    step_tracker, hooks = install_step_tracker(model_info)
    restore_callbacks = []
    install_model_patches = getattr(method_instance, "install_model_patches", None)
    if callable(install_model_patches):
        restore_callbacks.extend(install_model_patches(model_info))

    try:
        for layer_idx, (path, attn_module) in enumerate(model_info.iter_self_attn_modules()):
            new_processor = method_instance.create_processor(
                layer_idx=layer_idx,
                total_layers=model_info.num_self_attn_layers,
                original_processor=original_processors[path],
                step_tracker=step_tracker,
            )
            attn_module.set_processor(new_processor)
    except Exception:
        for path, proc in original_processors.items():
            model_info.get_attn_module(path).set_processor(proc)
        for hook in hooks:
            hook.remove()
        for restore_callback in reversed(restore_callbacks):
            restore_callback()
        raise

    return SparseAttentionHandle(
        model_info=model_info,
        original_processors=original_processors,
        step_tracker_hooks=hooks,
        restore_callbacks=restore_callbacks,
        method_instance=method_instance,
    )


def _resolve_scheduler_first_times_fp(pipe, method: str, config: Dict[str, Any]) -> None:
    if method not in _SCHEDULER_FIRST_TIMES_FP_METHODS or "first_times_fp" not in config:
        return
    first_times_fp = float(config["first_times_fp"])
    if first_times_fp <= 0 or first_times_fp >= 1:
        return

    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        return
    set_timesteps = getattr(scheduler, "set_timesteps", None)
    if not callable(set_timesteps):
        return

    steps = int(config.get("num_inference_steps", 50))
    ref_scheduler = copy.deepcopy(scheduler)
    ref_scheduler.set_timesteps(steps)
    timesteps = getattr(ref_scheduler, "timesteps", None)
    if timesteps is None:
        return

    num_fp_timesteps = math.floor(first_times_fp * steps)
    if num_fp_timesteps > 0:
        config["first_times_fp"] = _scalar_timestep(timesteps[num_fp_timesteps - 1]) - 1
    else:
        config["first_times_fp"] = 1001.0


def _scalar_timestep(timestep) -> float:
    if hasattr(timestep, "detach"):
        return float(timestep.detach().flatten()[0].item())
    return float(timestep)
