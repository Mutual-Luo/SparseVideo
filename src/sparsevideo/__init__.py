from ._api import apply, apply_sparse_attention, replace_attention, restore_sparse_attention, SparseAttentionHandle
from ._registry import default_method_config, list_methods, normalize_method_config

import sparsevideo.methods  # trigger method registration

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "apply",
    "apply_sparse_attention",
    "replace_attention",
    "restore_sparse_attention",
    "SparseAttentionHandle",
    "default_method_config",
    "list_methods",
    "normalize_method_config",
]
