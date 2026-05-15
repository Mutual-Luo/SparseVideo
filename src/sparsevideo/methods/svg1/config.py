CONFIG_DEFAULTS = {
    "first_layers_fp": 0.025,
    "first_times_fp": 0.075,
    "num_inference_steps": 50,
    "num_sampled_rows": 64,
    "sample_mse_max_row": 10000,
    "sparsity": 0.25,
}

CONFIG_ALIASES = {
    "skip_first_layers": "first_layers_fp",
    "skip_first_steps": "first_times_fp",
}
