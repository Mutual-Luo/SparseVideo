from .._config import copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
CONFIG_ALIASES = _YAML_CONFIG["aliases"]

SPARSE_INFO_KEYS = (
    "sparse_info",
    "sparse_kv_info",
    "sparse_info_indptr",
    "sparse_kv_info_indptr",
)


def default_config(**context):
    return copy_config_defaults(CONFIG_DEFAULTS)
