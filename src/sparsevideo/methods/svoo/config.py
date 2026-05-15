CONFIG_DEFAULTS = {
    "implementation": "native",
    "sparse_backend": "flashinfer",
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
    "use_svoo": True,
    "start_reuse_step": None,
    "reuse_interval": 1,
    "use_dynamic_min_kc_ratio": False,
    "sparsity_csv_path": "sparsity_profiles/sparsity_results.csv",
    "dynamic_min_kc_ratio_min": None,
    "dynamic_min_kc_ratio_max": None,
    "context_length": None,
    "prompt_length": None,
    "measure_attention_sparsity": False,
    "sparsity_output_file": "attention_sparsity.txt",
    "sparsity_batch_size": 0,
    "sparsity_query_samples": 0,
    "sparsity_threshold": 0.95,
    "sparsity_start_step": 1,
    "use_global_constraints": False,
    "lambda_schedule": "linear",
    "diverse_top_p_k": 0.0,
    "use_routing_transformer_strategy": False,
    "mq1": None,
    "mk1": None,
    "mq2": None,
    "mk2": None,
}

T2V_720P_DEFAULTS = {
    "wan": {
        "first_times_fp": 0.2,
        "first_layers_fp": 0.03,
        "num_q_centroids": 256,
        "num_k_centroids": 1024,
        "top_p_kmeans": 0.90,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 2,
        "kmeans_iter_step": 2,
        "start_reuse_step": 11,
        "reuse_interval": 20,
        "use_dynamic_min_kc_ratio": True,
        "dynamic_min_kc_ratio_min": 0.05,
        "dynamic_min_kc_ratio_max": 0.10,
    },
    "wan22-t2v-a14b": {
        "first_times_fp": 0.2,
        "first_layers_fp": 0.03,
        "num_q_centroids": 256,
        "num_k_centroids": 1024,
        "top_p_kmeans": 0.90,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 2,
        "kmeans_iter_step": 2,
        "start_reuse_step": 9,
        "reuse_interval": 20,
        "use_dynamic_min_kc_ratio": True,
        "dynamic_min_kc_ratio_min": 0.05,
        "dynamic_min_kc_ratio_max": 0.10,
    },
    "hunyuan_video": {
        "first_times_fp": 0.1,
        "first_layers_fp": 0.03,
        "num_q_centroids": 256,
        "num_k_centroids": 1024,
        "top_p_kmeans": 0.88,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 2,
        "kmeans_iter_step": 2,
        "start_reuse_step": 6,
        "reuse_interval": 50,
        "use_dynamic_min_kc_ratio": True,
        "dynamic_min_kc_ratio_min": 0.05,
        "dynamic_min_kc_ratio_max": 0.10,
    },
}

CONFIG_ALIASES = {
    "budget": "top_p_kmeans",
    "kmeans_iters": "kmeans_iter_step",
    "skip_first_steps": "first_times_fp",
    "skip_first_layers": "first_layers_fp",
}

UNPORTED_OPTION_DEFAULTS = {
    "measure_attention_sparsity": CONFIG_DEFAULTS["measure_attention_sparsity"],
    "sparsity_output_file": CONFIG_DEFAULTS["sparsity_output_file"],
    "sparsity_batch_size": CONFIG_DEFAULTS["sparsity_batch_size"],
    "sparsity_query_samples": CONFIG_DEFAULTS["sparsity_query_samples"],
    "sparsity_threshold": CONFIG_DEFAULTS["sparsity_threshold"],
    "sparsity_start_step": CONFIG_DEFAULTS["sparsity_start_step"],
    "use_global_constraints": CONFIG_DEFAULTS["use_global_constraints"],
    "lambda_schedule": CONFIG_DEFAULTS["lambda_schedule"],
    "diverse_top_p_k": CONFIG_DEFAULTS["diverse_top_p_k"],
    "use_routing_transformer_strategy": CONFIG_DEFAULTS["use_routing_transformer_strategy"],
    "mq1": CONFIG_DEFAULTS["mq1"],
    "mk1": CONFIG_DEFAULTS["mk1"],
    "mq2": CONFIG_DEFAULTS["mq2"],
    "mk2": CONFIG_DEFAULTS["mk2"],
}

UNPORTED_OPTIONS = tuple(UNPORTED_OPTION_DEFAULTS)


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_key = context.get("model_key")
    model_family = context.get("model_family")
    if model_key in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_key])
    elif model_family in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_family])
    return config
