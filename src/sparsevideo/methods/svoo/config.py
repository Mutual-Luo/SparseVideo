import os
from pathlib import Path

from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
T2V_720P_DEFAULTS = _YAML_CONFIG["model_defaults"]

CONFIG_ALIASES = _YAML_CONFIG["aliases"]
PROFILED_MODEL_KEYS = {
    "wan21-t2v-1.3b",
    "wan21-i2v-14b",
    "wan21-t2v-14b",
    "wan22-i2v-a14b",
    "wan22-t2v-a14b",
    "hunyuan_video",
    "hunyuan-t2v",
    "hunyuan-i2v",
}


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "", "false", "False")


def default_sparsity_csv_path(model_key=None):
    profile_dir = Path(__file__).resolve().parent / "sparsity_profiles"
    if model_key == "hunyuan-i2v":
        return str(profile_dir / "sparsity_hunyuan10_13B_i2v.csv")
    if model_key == "wan22-i2v-a14b":
        return str(profile_dir / "sparsity_wan22_A14B_i2v.csv")
    if model_key == "wan22-t2v-a14b":
        return str(profile_dir / "sparsity_wan22_A14B_t2v.csv")
    if model_key == "wan21-i2v-14b":
        return str(profile_dir / "sparsity_wan_14B_i2v.csv")
    if model_key == "wan21-t2v-14b":
        return str(profile_dir / "sparsity_wan_14B_t2v.csv")
    if model_key in ("hunyuan_video", "hunyuan-t2v"):
        return str(profile_dir / "sparsity_hunyuan10_13B_t2v.csv")
    if model_key == "wan21-t2v-1.3b":
        return str(profile_dir / "sparsity_wan_1.3B_t2v.csv")
    return None


def _fallback_model_defaults_key(model_family=None, model_key=None):
    if model_key in T2V_720P_DEFAULTS:
        return model_key
    if model_family == "hunyuan_video":
        return "hunyuan-t2v"
    if model_family == "cogvideox":
        return "cogvideox-t2v"
    if model_family == "ltx_video":
        return "ltx-video"
    if model_family == "allegro":
        return "allegro"
    if model_family == "mochi":
        return "mochi-1"
    if model_family == "easyanimate":
        return "easyanimate-v5-t2v-12b"
    if model_family != "wan":
        return model_key
    if model_key is None:
        return "wan21-t2v-1.3b"

    key = str(model_key).lower()
    if "wan22" in key or "a14b" in key:
        return "wan22-t2v-a14b"
    if "1.3b" in key:
        return "wan21-t2v-1.3b"
    if "14b" in key:
        return "wan21-t2v-14b"
    return "wan21-t2v-1.3b"


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    config["enable_mem_save"] = _env_bool("SVOO_ENABLE_MEM_SAVE", config["enable_mem_save"])
    model_key = context.get("model_key")
    has_exact_profile = model_key in PROFILED_MODEL_KEYS
    model_defaults_key = _fallback_model_defaults_key(
        model_family=context.get("model_family"),
        model_key=model_key,
    )
    apply_model_defaults(config, T2V_720P_DEFAULTS, {**context, "model_key": model_defaults_key})
    if config.get("use_dynamic_min_kc_ratio") and not has_exact_profile:
        # No owned offline sparsity profile is available for this model. Fall
        # back to SVOO's online co-clustering with the fixed min_kc_ratio, which
        # matches the SVG2-style runtime threshold behavior and avoids borrowing
        # another backbone's CSV profile.
        config["use_dynamic_min_kc_ratio"] = False
    if (
        config.get("use_dynamic_min_kc_ratio")
        and config.get("sparsity_csv_path") == CONFIG_DEFAULTS["sparsity_csv_path"]
    ):
        profile = default_sparsity_csv_path(model_key=model_key)
        if profile is not None:
            config["sparsity_csv_path"] = profile
    return config
