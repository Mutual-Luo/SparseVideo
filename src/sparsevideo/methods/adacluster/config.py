CONFIG_DEFAULTS = {
    "topk_num": 128,
    "q_kernel_num": 100,
    "kv_kernel_num": 500,
    "kmeans_iter_init": 3,
    "kmeans_iter_step": 1,
    "use_thresholded_kmeans_loop": False,
    "initial_q_kernel_num": 50,
    "initial_kv_kernel_num": 200,
    "q_distance_threshold": 9.0,
    "kv_distance_threshold": 5.5,
    "thresholded_kmeans_iter_time": 3,
    "thresholded_kmeans_max_iterations": 10,
}

CONFIG_ALIASES = {}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    if context.get("model_family") == "hunyuan_video":
        config["topk_num"] = 94
        config["q_kernel_num"] = 250
        config["kv_kernel_num"] = 1243
    return config
