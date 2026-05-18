CONFIG_DEFAULTS = {
    "dense_layers": 0,
    "dense_timesteps": 0,
    "decay_factor": 1,
    "block_size": 128,
    "use_sage_attention": False,
    "allow_flex_fallback": False,
}

CONFIG_ALIASES = {}

T2V_DEFAULTS = {
    "wan": {
        "dense_layers": 1,
        "dense_timesteps": 12,
        "decay_factor": 0.2,
        "block_size": 128,
    },
    "wan22-t2v-a14b": {
        "dense_layers": 1,
        "dense_timesteps": 11,
        "decay_factor": 0.8,
        "block_size": 64,
    },
    "hunyuan_video": {
        "dense_layers": 0,
        "dense_timesteps": 12,
        "decay_factor": 0.95,
        "block_size": 128,
    },
}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_key = context.get("model_key")
    model_family = context.get("model_family")
    if model_key in T2V_DEFAULTS:
        config.update(T2V_DEFAULTS[model_key])
    elif model_family in T2V_DEFAULTS:
        config.update(T2V_DEFAULTS[model_family])
    return config
