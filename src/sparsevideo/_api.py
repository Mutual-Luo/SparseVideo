from __future__ import annotations

from typing import Any, Dict, Optional

from ._registry import get_method_class, list_methods as _list_methods
from ._model_info import ModelInfo, discover_model
from ._step_tracker import install_step_tracker


class SparseAttentionHandle:
    def __init__(
        self,
        model_info: ModelInfo,
        original_processors: Dict[str, Any],
        step_tracker_hooks: list,
    ):
        self._model_info = model_info
        self._original_processors = original_processors
        self._step_tracker_hooks = step_tracker_hooks
        self._restored = False

    def restore(self):
        if self._restored:
            return
        for path, proc in self._original_processors.items():
            attn_module = self._model_info.get_attn_module(path)
            attn_module.set_processor(proc)
        for hook in self._step_tracker_hooks:
            hook.remove()
        self._restored = True


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

    original_processors = {}
    for path, attn_module in model_info.iter_self_attn_modules():
        original_processors[path] = attn_module.get_processor()

    step_tracker, hooks = install_step_tracker(model_info)

    for layer_idx, (path, attn_module) in enumerate(model_info.iter_self_attn_modules()):
        new_processor = method_instance.create_processor(
            layer_idx=layer_idx,
            total_layers=model_info.num_self_attn_layers,
            original_processor=original_processors[path],
            step_tracker=step_tracker,
        )
        attn_module.set_processor(new_processor)

    return SparseAttentionHandle(
        model_info=model_info,
        original_processors=original_processors,
        step_tracker_hooks=hooks,
    )
