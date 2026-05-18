from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def _is_training_free_runtime(module) -> bool:
    path = getattr(module, "__file__", "") or ""
    return "training_free" in Path(path).parts


def _local_sageattention_root() -> Path:
    return Path(__file__).resolve().parent / "native" / "sageattention"


def _candidate_sageattention_roots() -> list[Path]:
    local_root = _local_sageattention_root()
    roots = []
    env_root = os.environ.get("SPARSEVIDEO_SAGEATTENTION_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        resolved = root.resolve(strict=False)
        if "training_free" in root.parts or "training_free" in resolved.parts:
            raise ImportError(
                "Refusing SPARSEVIDEO_SAGEATTENTION_ROOT inside training_free; "
                "SparseVideo runtime kernels must live under "
                f"{local_root}."
            )
        try:
            resolved.relative_to(local_root.resolve())
        except ValueError:
            raise ImportError(
                "Refusing SPARSEVIDEO_SAGEATTENTION_ROOT outside the SparseVideo-owned "
                f"runtime root {local_root}: {resolved}"
            )
        roots.append(resolved)
    roots.append(local_root)
    return roots


def _has_sageattention_extensions(root: Path) -> bool:
    package = root / "sageattention"
    return (
        (package / "__init__.py").exists()
        and bool(list(package.glob("_fused*.so")))
        and bool(list(package.glob("_qattn_sm*.so")))
    )


def _clear_sageattention_modules() -> None:
    for name in list(sys.modules):
        if name == "sageattention" or name.startswith("sageattention."):
            del sys.modules[name]


def _import_sageattention_from_root(root: Path):
    _clear_sageattention_modules()
    root_str = str(root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    try:
        return importlib.import_module("sageattention")
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


def load_sageattention_module():
    sageattention = None
    selected_root = None
    for root in _candidate_sageattention_roots():
        if _has_sageattention_extensions(root):
            sageattention = _import_sageattention_from_root(root)
            selected_root = root
            break

    if sageattention is None:
        raise ImportError(
            "Radial use_sage_attention dense warmup requires the SparseVideo-owned "
            "SageAttention runtime with _qattn_sm* and _fused CUDA extensions built under "
            f"{_local_sageattention_root()}. Do not rely on training_free/ or "
            "environment sageattention packages for SparseVideo runtime parity."
        )

    if _is_training_free_runtime(sageattention):
        raise ImportError(
            "sageattention resolved from training_free/, which is reference-only. "
            "Build the SparseVideo-owned SageAttention runtime under "
            "src/sparsevideo/kernels/native/sageattention."
        )
    if selected_root is not None and not _is_module_under_root(sageattention, selected_root):
        raise ImportError(
            "sageattention resolved outside the selected SparseVideo-owned runtime root "
            f"{selected_root}."
        )
    return sageattention


def load_sageattn_function():
    sageattention = load_sageattention_module()
    try:
        return sageattention.sageattn
    except AttributeError as exc:
        raise ImportError("sageattention does not expose sageattn") from exc
