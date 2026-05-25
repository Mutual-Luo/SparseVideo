from __future__ import annotations

from pathlib import Path

from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_MASK_STRATEGY_DIR = Path(__file__).with_name("mask_strategies")
_WAN_MASK_STRATEGY = str(_MASK_STRATEGY_DIR / "mask_strategy_wan.json")
_WAN13_MASK_STRATEGY = str(_MASK_STRATEGY_DIR / "mask_strategy_wan13.json")
_HUNYUAN_MASK_STRATEGY = str(_MASK_STRATEGY_DIR / "mask_strategy_hunyuan.json")

_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
MODEL_DEFAULTS = _YAML_CONFIG["model_defaults"]
CONFIG_ALIASES = _YAML_CONFIG["aliases"]


def _owned_model_mask_strategy(model_key):
    if not model_key:
        return None
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(model_key).lower()).strip("_")
    path = _MASK_STRATEGY_DIR / f"mask_strategy_{safe_key}.json"
    return str(path) if path.exists() else None


# WanVACE 1.3B keeps its separate local default until a matching 720p/81f search is recorded.
_SKIP_OWNED_MASK_STRATEGY = {"wan21-vace-1.3b"}


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    model_key = context.get("model_key")
    apply_model_defaults(config, MODEL_DEFAULTS, context)
    if model_key in _SKIP_OWNED_MASK_STRATEGY:
        pass
    elif (owned_strategy := _owned_model_mask_strategy(model_key)) is not None:
        config["mask_strategy_file_path"] = owned_strategy
    elif model_key == "wan21-t2v-14b":
        config["mask_strategy_file_path"] = _WAN_MASK_STRATEGY
    elif model_key in ("hunyuan-t2v", "hunyuan-i2v"):
        config["mask_strategy_file_path"] = _HUNYUAN_MASK_STRATEGY
    return config
