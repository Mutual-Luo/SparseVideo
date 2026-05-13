from __future__ import annotations

from typing import Any, Dict, Type

_METHODS: Dict[str, Type] = {}


def register_method(name: str, cls: Type):
    _METHODS[name] = cls


def get_method_class(name: str) -> Type:
    if name not in _METHODS:
        available = ", ".join(sorted(_METHODS))
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return _METHODS[name]


def list_methods() -> list[str]:
    return sorted(_METHODS)
