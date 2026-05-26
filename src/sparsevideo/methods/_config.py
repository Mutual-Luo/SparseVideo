from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_method_config_yaml(module_file: str) -> Dict[str, Any]:
    path = Path(module_file).with_name("config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for key in ("defaults", "model_defaults"):
        if key not in data or data[key] is None:
            data[key] = {}
        value = data[key]
        if not isinstance(value, dict):
            raise ValueError(f"{path} field {key!r} must be a mapping")
    if "compat_keys" not in data or data["compat_keys"] is None:
        data["compat_keys"] = []
    if not isinstance(data["compat_keys"], list) or not all(
        isinstance(key, str) for key in data["compat_keys"]
    ):
        raise ValueError(f"{path} field 'compat_keys' must be a string list")
    return data


def copy_config_defaults(defaults: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(defaults))


def apply_model_defaults(
    config: Dict[str, Any],
    model_defaults: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
) -> None:
    model_key = context.get("model_key")
    if model_key in model_defaults:
        config.update(copy.deepcopy(dict(model_defaults[model_key])))
