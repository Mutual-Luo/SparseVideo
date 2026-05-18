CONFIG_DEFAULTS = {
    "pool_h": 8,
    "pool_w": 16,
    "latent_h": None,
    "latent_w": None,
    "visual_len": None,
    "text_len": None,
    "sparsity_ratio": 0.9,
    "batch_size": None,
    "block_sparse_attention": True,
    "allow_triton_fallback": False,
}

CONFIG_ALIASES = {}

T2V_DEFAULTS = {
    "wan": {
        "sparsity_ratio": 0.75,
        "text_len": 0,
    },
    "hunyuan_video": {
        "latent_h": 48,
        "latent_w": 80,
        "visual_len": 126_720,
        "text_len": 256,
        "sparsity_ratio": 0.9,
    },
}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_family = context.get("model_family")
    if model_family in T2V_DEFAULTS:
        config.update(T2V_DEFAULTS[model_family])
    return config
