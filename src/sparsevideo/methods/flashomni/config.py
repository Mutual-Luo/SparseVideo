CONFIG_DEFAULTS = {
    "implementation": "upstream",
    "backend": "auto",
    "workspace_bytes": 268435456,
    "sparse_block_size_for_q": 128,
    "sparse_block_size_for_kv": 128,
    "is_full": False,
    "sparse_info": None,
    "sparse_kv_info": None,
    "sparse_info_indptr": None,
    "sparse_kv_info_indptr": None,
    "sparse_kv_budget": 0.5,
}

CONFIG_ALIASES = {
    "block_size": "sparse_block_size_for_q",
    "budget": "sparse_kv_budget",
}

UNPORTED_OPTION_DEFAULTS = {
    "sparse_info": CONFIG_DEFAULTS["sparse_info"],
    "sparse_kv_info": CONFIG_DEFAULTS["sparse_kv_info"],
    "sparse_info_indptr": CONFIG_DEFAULTS["sparse_info_indptr"],
    "sparse_kv_info_indptr": CONFIG_DEFAULTS["sparse_kv_info_indptr"],
}
