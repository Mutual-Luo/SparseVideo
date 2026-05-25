from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Optional

from ._registry import get_method_class, list_methods as _list_methods
from ._model_info import ModelInfo, discover_model
from ._step_tracker import install_step_tracker
from ._support import (
    LIMITED_METHODS_BY_MODEL_TYPE,
    unvalidated_method_reason,
    unsupported_method_model_reason,
)


_ACTIVE_HANDLE_ATTR = "_sparsevideo_active_handle"


class SparseAttentionHandle:
    def __init__(
        self,
        model_info: ModelInfo,
        original_processors: Dict[str, Any],
        step_tracker_hooks: list,
        restore_callbacks: Optional[list] = None,
        patched_attention_paths: Optional[list] = None,
        method_instance: Any = None,
        pipe: Any = None,
    ):
        self._model_info = model_info
        self._original_processors = original_processors
        self._step_tracker_hooks = step_tracker_hooks
        self._restore_callbacks = restore_callbacks or []
        self._patched_attention_paths = patched_attention_paths or []
        self._method_instance = method_instance
        self._pipe = pipe
        self._restored = False

    def summary(self) -> Dict[str, Any]:
        pipeline_backend = getattr(self._model_info, "pipeline_backend", "diffusers")
        processor_classes = {}
        for path in sorted(self._original_processors):
            try:
                processor = _get_processor(self._model_info.get_attn_module(path))
                processor_classes[path] = f"{type(processor).__module__}.{type(processor).__name__}"
            except Exception as exc:
                processor_classes[path] = f"<unavailable:{type(exc).__name__}>"

        method_class = None
        method_config = None
        method_runtime = None
        if self._method_instance is not None:
            method_class = type(self._method_instance).__name__
            method_config = copy.deepcopy(getattr(self._method_instance, "config", None))
            runtime_summary = getattr(self._method_instance, "runtime_summary", None)
            if callable(runtime_summary):
                method_runtime = runtime_summary()

        return {
            "pipeline_backend": pipeline_backend,
            "diffsynth_version": _backend_package_version(pipeline_backend),
            "model_type": self._model_info.model_type,
            "model_key": self._model_info.model_key,
            "num_self_attn_layers": self._model_info.num_self_attn_layers,
            "installed_processor_count": len(self._original_processors),
            "installed_processor_paths": sorted(self._original_processors),
            "patched_attention_count": len(self._patched_attention_paths),
            "patched_attention_paths": sorted(self._patched_attention_paths),
            "unpatched_attention_paths": sorted(
                getattr(self._model_info, "unpatched_attention_paths", [])
            ),
            "pipeline_notes": list(getattr(self._model_info, "pipeline_notes", [])),
            "current_processor_classes": processor_classes,
            "step_tracker_hook_count": len(self._step_tracker_hooks),
            "restore_callback_count": len(self._restore_callbacks),
            "method_class": method_class,
            "method_config": method_config,
            "method_runtime": method_runtime,
            "restored": self._restored,
        }

    def restore(self):
        if self._restored:
            return
        for path, proc in self._original_processors.items():
            attn_module = self._model_info.get_attn_module(path)
            _set_processor(attn_module, proc)
        for hook in self._step_tracker_hooks:
            hook.remove()
        for restore_callback in reversed(self._restore_callbacks):
            restore_callback()
        self._restored = True
        _clear_active_handle(self._pipe, self)


def restore_sparse_attention(target) -> None:
    if isinstance(target, SparseAttentionHandle):
        target.restore()
        return
    _restore_active_handle(target)


def apply(
    pipe,
    method: str = "dense",
    config: Optional[Dict[str, Any]] = None,
) -> SparseAttentionHandle:
    return apply_sparse_attention(pipe, method=method, config=config)


def replace_attention(
    pipe,
    method: str = "dense",
    config: Optional[Dict[str, Any]] = None,
):
    apply_sparse_attention(pipe, method=method, config=config)
    return pipe


def apply_sparse_attention(
    pipe,
    method: str = "dense",
    config: Optional[Dict[str, Any]] = None,
) -> SparseAttentionHandle:
    if config is None:
        config = {}

    _auto_set_torch_cuda_arch_list()
    method_cls = get_method_class(method)
    _restore_active_handle(pipe)

    model_info = discover_model(pipe)
    _validate_method_support(model_info, method)
    method_instance = method_cls(config=config, model_info=model_info)
    if method == "dense":
        return SparseAttentionHandle(
            model_info=model_info,
            original_processors={},
            step_tracker_hooks=[],
            restore_callbacks=[],
            patched_attention_paths=[],
            method_instance=method_instance,
            pipe=pipe,
        )

    if getattr(model_info, "pipeline_backend", "diffusers") == "diffsynth":
        from ._diffsynth import apply_diffsynth_sparse_attention

        return apply_diffsynth_sparse_attention(
            pipe,
            model_info,
            method_instance,
            handle_cls=SparseAttentionHandle,
            set_active_handle=_set_active_handle,
        )

    original_processors = {}
    for path, attn_module in model_info.iter_self_attn_modules():
        original_processors[path] = _get_processor(attn_module)

    step_tracker, hooks = install_step_tracker(
        model_info,
        num_inference_steps_fn=lambda: _pipeline_num_inference_steps(pipe),
    )
    restore_callbacks = []
    install_model_patches = getattr(method_instance, "install_model_patches", None)
    if callable(install_model_patches):
        restore_callbacks.extend(install_model_patches(model_info))

    try:
        for layer_idx, (path, attn_module) in enumerate(model_info.iter_self_attn_modules()):
            processor_layer_idx, processor_total_layers = model_info.self_attn_layer_context(
                path,
                layer_idx,
                model_info.num_self_attn_layers,
            )
            new_processor = method_instance.create_processor(
                layer_idx=processor_layer_idx,
                total_layers=processor_total_layers,
                original_processor=original_processors[path],
                step_tracker=step_tracker,
            )
            _set_processor(attn_module, new_processor)
    except Exception:
        for path, proc in original_processors.items():
            _set_processor(model_info.get_attn_module(path), proc)
        for hook in hooks:
            hook.remove()
        for restore_callback in reversed(restore_callbacks):
            restore_callback()
        raise

    handle = SparseAttentionHandle(
        model_info=model_info,
        original_processors=original_processors,
        step_tracker_hooks=hooks,
        restore_callbacks=restore_callbacks,
        patched_attention_paths=[],
        method_instance=method_instance,
        pipe=pipe,
    )
    _set_active_handle(pipe, handle)
    return handle


def _auto_set_torch_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return

    try:
        import torch

        if not torch.cuda.is_available():
            return
        archs = sorted(
            {
                f"{major}.{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(index)
                    for index in range(torch.cuda.device_count())
                )
            }
        )
    except Exception:
        return

    if archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)


def _validate_method_support(model_info: ModelInfo, method: str) -> None:
    if method == "dense":
        return
    model_reason = unsupported_method_model_reason(method, model_info.model_key)
    if model_reason is not None:
        print(f"\033[31mError:\033[0m {model_reason}", file=sys.stderr)
        raise NotImplementedError(model_reason)
    supported_methods = LIMITED_METHODS_BY_MODEL_TYPE.get(model_info.model_type)
    if supported_methods is None or method in supported_methods:
        return
    reason = unvalidated_method_reason(method)
    raise NotImplementedError(
        f"{method} is not implemented for {model_info.model_type}; "
        f"supported sparse methods: {list(supported_methods)}. {reason}"
    )


def _restore_active_handle(pipe) -> None:
    handle = getattr(pipe, _ACTIVE_HANDLE_ATTR, None)
    if isinstance(handle, SparseAttentionHandle) and not getattr(handle, "_restored", True):
        handle.restore()


def _set_active_handle(pipe, handle: SparseAttentionHandle) -> None:
    try:
        setattr(pipe, _ACTIVE_HANDLE_ATTR, handle)
    except Exception:
        return


def _pipeline_num_inference_steps(pipe) -> Optional[int]:
    scheduler = getattr(pipe, "scheduler", None)
    timesteps = getattr(scheduler, "timesteps", None)
    if timesteps is None:
        return None
    try:
        return len(timesteps)
    except TypeError:
        return None


def _clear_active_handle(pipe, handle: SparseAttentionHandle) -> None:
    if pipe is None:
        return
    try:
        if getattr(pipe, _ACTIVE_HANDLE_ATTR, None) is handle:
            setattr(pipe, _ACTIVE_HANDLE_ATTR, None)
    except Exception:
        return


def _backend_package_version(pipeline_backend: str) -> str | None:
    if pipeline_backend != "diffsynth":
        return None
    from ._diffsynth import diffsynth_version

    return diffsynth_version()


def _get_processor(attn_module):
    get_processor = getattr(attn_module, "get_processor", None)
    if callable(get_processor):
        return get_processor()
    if hasattr(attn_module, "processor"):
        return attn_module.processor
    raise AttributeError(f"{type(attn_module).__name__} has no get_processor() or .processor")


def _set_processor(attn_module, processor) -> None:
    set_processor = getattr(attn_module, "set_processor", None)
    if callable(set_processor):
        set_processor(processor)
        return
    if hasattr(attn_module, "processor"):
        attn_module.processor = processor
        return
    raise AttributeError(f"{type(attn_module).__name__} has no set_processor() or .processor")
