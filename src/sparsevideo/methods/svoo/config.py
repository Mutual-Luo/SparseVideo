import os
from pathlib import Path


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
    "enable_mem_save": True,
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
    "use_fused_rope": True,
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
        "num_inference_steps": 40,
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

CONFIG_ALIASES = {}


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "", "false", "False")

def default_sparsity_csv_path(model_family=None, model_key=None):
    profile_dir = Path(__file__).resolve().parent / "sparsity_profiles"
    if model_key == "wan22-t2v-a14b":
        return str(profile_dir / "sparsity_wan22_A14B_t2v.csv")
    if model_key == "wan21-t2v-14b":
        return str(profile_dir / "sparsity_wan_14B_t2v.csv")
    if model_family == "hunyuan_video" or model_key == "hunyuan_video":
        return str(profile_dir / "sparsity_hunyuan10_13B_t2v.csv")
    if model_family == "wan" or model_key == "wan21-t2v-1.3b":
        return str(profile_dir / "sparsity_wan_1.3B_t2v.csv")
    return None


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    config["enable_mem_save"] = _env_bool("SVOO_ENABLE_MEM_SAVE", config["enable_mem_save"])
    model_key = context.get("model_key")
    model_family = context.get("model_family")
    if model_key in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_key])
    elif model_family in T2V_720P_DEFAULTS:
        config.update(T2V_720P_DEFAULTS[model_family])
    if (
        config.get("use_dynamic_min_kc_ratio")
        and config.get("sparsity_csv_path") == CONFIG_DEFAULTS["sparsity_csv_path"]
    ):
        profile = default_sparsity_csv_path(model_family=model_family, model_key=model_key)
        if profile is not None:
            config["sparsity_csv_path"] = profile
    return config
