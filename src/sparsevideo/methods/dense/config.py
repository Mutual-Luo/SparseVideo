from .._config import copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]

CONFIG_ALIASES = {}


def default_config(**context):
    return copy_config_defaults(CONFIG_DEFAULTS)
