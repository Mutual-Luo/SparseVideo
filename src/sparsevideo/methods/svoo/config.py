import os
from pathlib import Path

from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
T2V_720P_DEFAULTS = _YAML_CONFIG["model_defaults"]

CONFIG_ALIASES = {}


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
    if model_key == "hunyuan-t2v":
        return str(profile_dir / "sparsity_hunyuan10_13B_t2v.csv")
    if model_key == "wan21-t2v-1.3b":
        return str(profile_dir / "sparsity_wan_1.3B_t2v.csv")
    return None


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    config["enable_mem_save"] = _env_bool("SVOO_ENABLE_MEM_SAVE", config["enable_mem_save"])
    model_key = context.get("model_key")
    has_profile = default_sparsity_csv_path(model_key=model_key) is not None
    apply_model_defaults(config, T2V_720P_DEFAULTS, context)
    if config.get("use_dynamic_min_kc_ratio") and not has_profile:
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
