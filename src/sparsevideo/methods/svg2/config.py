CONFIG_DEFAULTS = {
    "first_layers_fp": 0.025,
    "first_times_fp": 0.075,
    "num_inference_steps": 50,
    "num_q_centroids": 50,
    "num_k_centroids": 200,
    "top_p_kmeans": 0.9,
    "min_kc_ratio": 0.0,
    "kmeans_iter_init": 0,
    "kmeans_iter_step": 0,
    "zero_step_kmeans_init": False,
}

T2V_720P_DEFAULTS = {
    "wan": {
        "first_times_fp": 0.2,
        "first_layers_fp": 0.03,
        "num_q_centroids": 300,
        "num_k_centroids": 1000,
        "top_p_kmeans": 0.9,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 50,
        "kmeans_iter_step": 2,
    },
    "hunyuan_video": {
        "first_times_fp": 0.1,
        "first_layers_fp": 0.03,
        "num_q_centroids": 400,
        "num_k_centroids": 1000,
        "top_p_kmeans": 0.9,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 50,
        "kmeans_iter_step": 2,
    },
}

CONFIG_ALIASES = {
    "budget": "top_p_kmeans",
    "kmeans_iters": "kmeans_iter_step",
    "skip_first_layers": "first_layers_fp",
    "skip_first_steps": "first_times_fp",
}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_family = context.get("model_family")
    if model_family in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_family])
    return config
