CONFIG_DEFAULTS = {
    "mode": "full",
    "value": None,
    "tune": False,
    "parallel_tune": False,
    "l1": 0.06,
    "pv_l1": 0.065,
    "tune_pv": False,
    "verbose": False,
}

CONFIG_ALIASES = {
    "budget": "value",
}

UNPORTED_OPTION_DEFAULTS = {
    "tune": CONFIG_DEFAULTS["tune"],
    "parallel_tune": CONFIG_DEFAULTS["parallel_tune"],
    "l1": CONFIG_DEFAULTS["l1"],
    "pv_l1": CONFIG_DEFAULTS["pv_l1"],
    "tune_pv": CONFIG_DEFAULTS["tune_pv"],
    "verbose": CONFIG_DEFAULTS["verbose"],
}
