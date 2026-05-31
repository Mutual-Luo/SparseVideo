from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Optional, Type


@dataclass
class MethodEntry:
    module: str
    class_name: str
    config_module: Optional[str] = None
    loaded_class: Optional[Type] = None


_METHODS: Dict[str, MethodEntry] = {}


def register_method(
    name: str,
    cls: Optional[Type] = None,
    *,
    module: Optional[str] = None,
    class_name: Optional[str] = None,
    config_module: Optional[str] = None,
):
    if cls is not None:
        module = cls.__module__
        class_name = cls.__name__
    if module is None or class_name is None:
        raise ValueError("register_method requires either cls or module/class_name")
    _METHODS[name] = MethodEntry(
        module=module,
        class_name=class_name,
        config_module=config_module,
        loaded_class=cls,
    )


def get_method_class(name: str) -> Type:
    if name not in _METHODS:
        available = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    entry = _METHODS[name]
    if entry.loaded_class is None:
        module = importlib.import_module(entry.module)
        entry.loaded_class = getattr(module, entry.class_name)
    return entry.loaded_class


def list_methods() -> list[str]:
    return sorted(_METHODS)


def default_method_config(name: str, **context: Any) -> Dict[str, Any]:
    entry = _get_entry(name)
    if entry.config_module is not None:
        module = importlib.import_module(entry.config_module)
        if hasattr(module, "default_config"):
            config = dict(module.default_config(**context))
        else:
            config = dict(getattr(module, "CONFIG_DEFAULTS"))
        num_inference_steps = context.get("num_inference_steps")
        compat_keys = _compat_keys_for_defaults(config, getattr(module, "CONFIG_COMPAT_KEYS", ()))
        if num_inference_steps is not None and (
            "num_inference_steps" in config
            or "num_inference_steps" in compat_keys
        ):
            config["num_inference_steps"] = num_inference_steps
        return config
    method_cls = get_method_class(name)
    return method_cls.default_config(**context)


def normalize_method_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    entry = _get_entry(name)
    if entry.config_module is not None:
        module = importlib.import_module(entry.config_module)
        defaults = getattr(module, "CONFIG_DEFAULTS")
        aliases = getattr(module, "CONFIG_ALIASES", {})
        compat_keys = _compat_keys_for_defaults(defaults, getattr(module, "CONFIG_COMPAT_KEYS", ()))
        return _normalize_config(config, defaults, aliases, compat_keys, entry.class_name)
    method_cls = get_method_class(name)
    return method_cls.normalize_config(config)


def _get_entry(name: str) -> MethodEntry:
    if name not in _METHODS:
        available = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return _METHODS[name]


def _compat_keys_for_defaults(defaults: Dict[str, Any], compat_keys) -> set[str]:
    merged = set(compat_keys)
    if "dense_warmup_step_ratio" in defaults:
        merged.add("num_inference_steps")
    return merged


def _normalize_config(
    config: Dict[str, Any],
    defaults: Dict[str, Any],
    aliases: Dict[str, str],
    compat_keys,
    class_name: str,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in config.items():
        canonical_key = aliases.get(key, key)
        if canonical_key in normalized:
            raise ValueError(
                f"Config key {key!r} conflicts with {canonical_key!r} "
                f"for {class_name}"
            )
        normalized[canonical_key] = value

    unknown = set(normalized) - set(defaults) - set(compat_keys)
    if unknown:
        valid_keys = [*defaults, *sorted(compat_keys)]
        raise ValueError(
            f"Unknown config keys for {class_name}: {unknown}. "
            f"Valid keys: {valid_keys}"
        )
    return normalized
