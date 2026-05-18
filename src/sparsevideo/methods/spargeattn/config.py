CONFIG_DEFAULTS = {
    "mode": "topk",
    "value": None,
    "topk": 0.5,
    "cdfthreshd": 0.98,
    "simthreshd1": None,
    "pvthreshd": 50,
    "attention_sink": False,
    "smooth_k": True,
    "dropout_p": 0.0,
    "scale": None,
    "tensor_layout": "HND",
    "output_dtype": "float16",
    "return_sparsity": False,
    "mask_id": None,
    "tune": False,
    "parallel_tune": False,
    "sim_rule": "l1",
    "l1": 0.07,
    "pv_l1": 0.08,
    "cos_sim": 0.98,
    "rmse": 0.07,
    "rearrange_kwargs": {},
    "tune_pv": True,
    "verbose": False,
    "model_out_path": None,
    "use_fused_qk_norm_rope": True,
}

CONFIG_ALIASES = {}


def default_config(**context):
    config = dict(CONFIG_DEFAULTS)
    model_family = context.get("model_family")
    if model_family == "hunyuan_video":
        config["l1"] = 0.07
        config["pv_l1"] = 0.08
        config["tune_pv"] = True
    return config
