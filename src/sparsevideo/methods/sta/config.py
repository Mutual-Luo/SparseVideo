from __future__ import annotations

from pathlib import Path


_WAN_MASK_STRATEGY = str(Path(__file__).with_name("mask_strategy_wan.json"))
_HUNYUAN_MASK_STRATEGY = str(Path(__file__).with_name("mask_strategy_hunyuan.json"))

CONFIG_DEFAULTS = {
    "tile_size": [6, 8, 8],
    "window_size": [3, 6, 10],
    "seq_shape": None,
    "has_text": True,
    "STA_mode": "STA_inference",
    "mask_strategy_file_path": None,
}

CONFIG_ALIASES = {}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_key = context.get("model_key")
    if model_key == "wan21-t2v-14b":
        config["has_text"] = False
        config["window_size"] = [3, 6, 10]
        config["mask_strategy_file_path"] = _WAN_MASK_STRATEGY
    elif context.get("model_family") == "wan":
        config["has_text"] = False
        config["window_size"] = [3, 6, 10]
    elif context.get("model_family") == "hunyuan_video":
        config["has_text"] = True
        config["window_size"] = [5, 6, 10]
        config["mask_strategy_file_path"] = _HUNYUAN_MASK_STRATEGY
    return config
