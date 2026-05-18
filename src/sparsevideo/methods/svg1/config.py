CONFIG_DEFAULTS = {
    "first_layers_fp": 0.025,
    "first_times_fp": 0.075,
    "num_inference_steps": 50,
    "num_sampled_rows": 64,
    "sample_mse_max_row": 10000,
    "sparsity": 0.25,
    "context_length": None,
    "prompt_length": None,
}

T2V_720P_DEFAULTS = {
    "wan": {
        "first_times_fp": 0.2,
        "first_layers_fp": 0.03,
        "num_sampled_rows": 64,
        "sparsity": 0.3,
    },
    "hunyuan_video": {
        "first_times_fp": 0.1,
        "first_layers_fp": 0.03,
        "num_sampled_rows": 64,
        "sparsity": 0.25,
        "context_length": 256,
        "prompt_length": None,
    },
}

CONFIG_ALIASES = {}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_family = context.get("model_family")
    if model_family in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_family])
    return config
