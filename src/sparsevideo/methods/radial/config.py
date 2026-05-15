CONFIG_DEFAULTS = {
    "dense_layers": 0,
    "dense_timesteps": 0,
    "decay_factor": 1,
    "use_sage_attention": False,
}

CONFIG_ALIASES = {
    "skip_first_layers": "dense_layers",
    "skip_first_steps": "dense_timesteps",
}
