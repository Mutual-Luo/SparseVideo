from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
MODEL_DEFAULTS = _YAML_CONFIG["model_defaults"]
CONFIG_ALIASES = {
    "cdfthreshd": "tau_kv",
    "tau_c": "tau_q",
    "cache_interval": "N",
    "cache_order": "D",
    "cache_threshold": "S_q",
}

SPARSE_INFO_KEYS = (
    "sparse_info",
    "sparse_kv_info",
    "sparse_info_indptr",
    "sparse_kv_info_indptr",
)


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    apply_model_defaults(config, MODEL_DEFAULTS, context)
    return config
