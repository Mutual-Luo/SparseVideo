from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable


_NATIVE_ROOT = Path(__file__).resolve().parent / "native" / "draft_block_sparse"


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _loaded_module_origin(module_name: str) -> Path | None:
    module = sys.modules.get(module_name)
    origin = getattr(module, "__file__", None)
    if origin:
        return Path(origin)
    return None


def _module_origin(module_name: str) -> Path | None:
    origin = _loaded_module_origin(module_name)
    if origin is not None:
        return origin
    spec = importlib.util.find_spec(module_name)
    if spec is not None and spec.origin:
        return Path(spec.origin)
    return None


def load_block_sparse_attn_func() -> Callable:
    """Load the SparseVideo-owned MIT Block-Sparse-Attention runtime."""

    if not _NATIVE_ROOT.exists():
        raise ImportError(
            "SparseVideo-owned Draft Block-Sparse-Attention source is missing at "
            f"{_NATIVE_ROOT}."
        )

    for module_name in ("block_sparse_attn", "block_sparse_attn_cuda"):
        origin = _loaded_module_origin(module_name)
        if origin is not None and not _path_is_under(origin, _NATIVE_ROOT):
            raise ImportError(
                f"{module_name} is already loaded from {origin}, not the "
                f"SparseVideo-owned Draft backend under {_NATIVE_ROOT}."
            )

    root = str(_NATIVE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        importlib.import_module("block_sparse_attn_cuda")
        module = importlib.import_module("block_sparse_attn")
    except Exception as exc:
        raise ImportError(
            "SparseVideo-owned Draft Block-Sparse-Attention extension is not built "
            f"or failed to load from {_NATIVE_ROOT}."
        ) from exc

    for module_name in ("block_sparse_attn", "block_sparse_attn_cuda"):
        origin = _module_origin(module_name)
        if origin is None or not _path_is_under(origin, _NATIVE_ROOT):
            raise ImportError(
                f"{module_name} resolves to {origin}, not the SparseVideo-owned "
                f"Draft backend under {_NATIVE_ROOT}."
            )

    return module.block_sparse_attn_func
