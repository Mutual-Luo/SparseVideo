CONFIG_DEFAULTS = {
    "topk_num": 128,
    "q_kernel_num": 100,
    "kv_kernel_num": 500,
    "kmeans_iter_init": 3,
    "kmeans_iter_step": 1,
    "late_layer_start": 20,
    "late_topk_num": 94,
    "late_q_kernel_num": 250,
    "late_kv_kernel_num": 1243,
}

CONFIG_ALIASES = {
    "num_clusters": "kv_kernel_num",
    "kmeans_iters": "kmeans_iter_step",
}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    if context.get("model_family") == "hunyuan_video":
        config["topk_num"] = 94
    return config
