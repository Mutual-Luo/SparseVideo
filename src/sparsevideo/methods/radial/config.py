from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
CONFIG_ALIASES = {}
T2V_DEFAULTS = _YAML_CONFIG["model_defaults"]


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    apply_model_defaults(config, T2V_DEFAULTS, context)
    return config
