from ._api import apply_sparse_attention, SparseAttentionHandle
from ._registry import default_method_config, list_methods, normalize_method_config

import sparsevideo.methods  # trigger method registration

__all__ = [
    "apply_sparse_attention",
    "SparseAttentionHandle",
    "default_method_config",
    "list_methods",
    "normalize_method_config",
]
