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
    "skyreels-v2-t2v-14b": {
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
        "use_dynamic_min_kc_ratio": False,
    },
    "skyreels-v2-i2v-14b": {
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
        "use_dynamic_min_kc_ratio": False,
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
    "cogvideox": {
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
        "use_dynamic_min_kc_ratio": False,
    },
    "ltx_video": {
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
        "use_dynamic_min_kc_ratio": False,
    },
    "allegro": {
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
        "use_dynamic_min_kc_ratio": False,
    },
    "mochi": {
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
        "use_dynamic_min_kc_ratio": False,
    },
    "easyanimate": {
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
        "use_dynamic_min_kc_ratio": False,
    },
}

CONFIG_ALIASES = {}
PROFILED_MODEL_KEYS = {
    "wan21-t2v-1.3b",
    "wan21-i2v-14b",
    "wan21-t2v-14b",
    "wan22-i2v-a14b",
    "wan22-t2v-a14b",
    "hunyuan_video",
    "hunyuan-t2v",
    "hunyuan-i2v",
}


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("0", "", "false", "False")

def default_sparsity_csv_path(model_family=None, model_key=None):
    profile_dir = Path(__file__).resolve().parent / "sparsity_profiles"
    if model_key == "hunyuan-i2v":
        return str(profile_dir / "sparsity_hunyuan10_13B_i2v.csv")
    if model_key == "wan22-i2v-a14b":
        return str(profile_dir / "sparsity_wan22_A14B_i2v.csv")
    if model_key == "wan22-t2v-a14b":
        return str(profile_dir / "sparsity_wan22_A14B_t2v.csv")
    if model_key == "wan21-i2v-14b":
        return str(profile_dir / "sparsity_wan_14B_i2v.csv")
    if model_key == "wan21-t2v-14b":
        return str(profile_dir / "sparsity_wan_14B_t2v.csv")
    if model_family == "hunyuan_video" or model_key in ("hunyuan_video", "hunyuan-t2v"):
        return str(profile_dir / "sparsity_hunyuan10_13B_t2v.csv")
    if model_key == "wan21-t2v-1.3b" or (
        model_key is None and model_family == "wan"
    ):
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
        model_key is not None
        and config.get("use_dynamic_min_kc_ratio")
        and model_key not in PROFILED_MODEL_KEYS
    ):
        # No owned offline sparsity profile is available for this model. Fall
        # back to SVOO's online co-clustering with the fixed min_kc_ratio, which
        # matches the SVG2-style runtime threshold behavior and avoids borrowing
        # another backbone's CSV profile.
        config["use_dynamic_min_kc_ratio"] = False
    if (
        config.get("use_dynamic_min_kc_ratio")
        and config.get("sparsity_csv_path") == CONFIG_DEFAULTS["sparsity_csv_path"]
    ):
        profile = default_sparsity_csv_path(model_family=model_family, model_key=model_key)
        if profile is not None:
            config["sparsity_csv_path"] = profile
    return config
