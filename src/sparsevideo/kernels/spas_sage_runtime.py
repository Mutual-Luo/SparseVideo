from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def _is_training_free_runtime(module) -> bool:
    path = getattr(module, "__file__", "") or ""
    return "training_free" in Path(path).parts


def _local_spargeattn_root() -> Path:
    return Path(__file__).resolve().parent / "native" / "spargeattn"


def _candidate_spas_sage_attn_roots() -> list[Path]:
    local_root = _local_spargeattn_root()
    roots = []
    env_root = os.environ.get("SPARSEVIDEO_SPARGEATTN_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        resolved = root.resolve(strict=False)
        if "training_free" in root.parts or "training_free" in resolved.parts:
            raise ImportError(
                "Refusing SPARSEVIDEO_SPARGEATTN_ROOT inside training_free; "
                "SparseVideo runtime kernels must live under "
                f"{local_root}."
            )
        try:
            resolved.relative_to(local_root.resolve())
        except ValueError:
            raise ImportError(
                "Refusing SPARSEVIDEO_SPARGEATTN_ROOT outside the SparseVideo-owned "
                f"runtime root {local_root}: {resolved}"
            )
        roots.append(resolved)
    roots.append(local_root)
    return roots


def _has_spas_sage_extensions(root: Path) -> bool:
    package = root / "spas_sage_attn"
    return (
        (package / "__init__.py").exists()
        and bool(list(package.glob("_qattn*.so")))
        and bool(list(package.glob("_fused*.so")))
    )


def _clear_spas_sage_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "spas_sage_attn"
            or name.startswith("spas_sage_attn.")
            or name == "tools"
            or name.startswith("tools.")
        ):
            del sys.modules[name]


def _import_spas_sage_attn_from_root(root: Path):
    _clear_spas_sage_modules()
    root_str = str(root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    try:
        return importlib.import_module("spas_sage_attn")
    finally:
        if added and root_str in sys.path:
            sys.path.remove(root_str)


def _import_spas_sage_submodule_from_root(root: Path, name: str):
    root_str = str(root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    try:
        return importlib.import_module(name)
    finally:
        if added and root_str in sys.path:
            sys.path.remove(root_str)


def _is_module_under_root(module, root: Path) -> bool:
    location = getattr(module, "__file__", None)
    if not location:
        return False
    try:
        Path(location).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def load_spas_sage_attn_module():
    spas_sage_attn = None
    selected_root = None
    for root in _candidate_spas_sage_attn_roots():
        if _has_spas_sage_extensions(root):
            spas_sage_attn = _import_spas_sage_attn_from_root(root)
            selected_root = root
            break

    if spas_sage_attn is None:
        raise ImportError(
            "SpargeAttn requires the SparseVideo-owned spas_sage_attn runtime with "
            "_qattn/_fused CUDA extensions built under "
            f"{_local_spargeattn_root()}. Do not rely on training_free/ or "
            "environment spas_sage_attn packages for SparseVideo runtime parity."
        )

    if _is_training_free_runtime(spas_sage_attn):
        raise ImportError(
            "spas_sage_attn resolved from training_free/, which is reference-only. "
            "Build the SparseVideo-owned SpargeAttn kernels under "
            "src/sparsevideo/kernels/native/spargeattn."
        )
    if selected_root is not None and not _is_module_under_root(spas_sage_attn, selected_root):
        raise ImportError(
            "spas_sage_attn resolved outside the selected SparseVideo-owned runtime root "
            f"{selected_root}."
        )
    return spas_sage_attn


def load_spas_sage_attn_functions():
    spas_sage_attn = load_spas_sage_attn_module()
    return (
        spas_sage_attn.spas_sage2_attn_meansim_cuda,
        spas_sage_attn.spas_sage2_attn_meansim_topk_cuda,
    )


def load_block_sparse_sage2_attn_function():
    spas_sage_attn = load_spas_sage_attn_module()
    try:
        return spas_sage_attn.block_sparse_sage2_attn_cuda
    except AttributeError as exc:
        raise ImportError("spas_sage_attn does not expose block_sparse_sage2_attn_cuda") from exc


def load_sparse_attention_meansim_class():
    selected_root = None
    for root in _candidate_spas_sage_attn_roots():
        if _has_spas_sage_extensions(root) and (root / "spas_sage_attn" / "autotune.py").exists():
            module = _import_spas_sage_submodule_from_root(root, "spas_sage_attn.autotune")
            selected_root = root
            break
    else:
        raise ImportError(
            "SpargeAttn tune/model_out_path requires the SparseVideo-owned "
            "spas_sage_attn.autotune runtime under "
            f"{_local_spargeattn_root()}."
        )

    if _is_training_free_runtime(module):
        raise ImportError(
            "spas_sage_attn.autotune resolved from training_free/, which is reference-only. "
            "Use the SparseVideo-owned SpargeAttn runtime under src/sparsevideo/kernels/native/spargeattn."
        )
    if selected_root is not None and not _is_module_under_root(module, selected_root):
        raise ImportError(
            "spas_sage_attn.autotune resolved outside the selected SparseVideo-owned runtime root "
            f"{selected_root}."
        )
    return module.SparseAttentionMeansim
