from .._config import apply_model_defaults, copy_config_defaults, load_method_config_yaml


_YAML_CONFIG = load_method_config_yaml(__file__)

CONFIG_DEFAULTS = _YAML_CONFIG["defaults"]
T2V_720P_DEFAULTS = _YAML_CONFIG["model_defaults"]

CONFIG_ALIASES = {}


def default_config(**context):
    config = copy_config_defaults(CONFIG_DEFAULTS)
    apply_model_defaults(config, T2V_720P_DEFAULTS, context)
    return config
