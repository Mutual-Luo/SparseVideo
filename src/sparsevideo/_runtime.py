from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except Exception:
        return None


def _package_locations(name: str) -> List[Path]:
    spec = _find_spec(name)
    if spec is None:
        return []
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        return [Path(path) for path in locations]
    if spec.origin:
        return [Path(spec.origin).parent]
    return []


def _glob_existing(paths: Iterable[Path], pattern: str) -> List[str]:
    files: List[str] = []
    for path in paths:
        if path.exists():
            files.extend(str(file) for file in sorted(path.glob(pattern)))
    return files


def _glob_any_existing(paths: Iterable[Path], patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(_glob_existing(paths, pattern))
    return sorted(set(files))


def _has_submodule_file(locations: Iterable[Path], name: str) -> bool:
    for path in locations:
        if (path / f"{name}.py").exists() or (path / name / "__init__.py").exists():
            return True
        if any(path.glob(f"{name}*.so")):
            return True
    return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_dir_status(path: Path, patterns: Iterable[str]) -> Dict[str, Any]:
    files: List[str] = []
    if path.exists():
        for pattern in patterns:
            files.extend(str(file) for file in sorted(path.glob(pattern)))
    return {
        "path": str(path),
        "exists": path.exists(),
        "source_files": bool(files),
        "sample_files": files[:8],
    }


def _native_kernel_dirs() -> List[Path]:
    dirs: List[Path] = []

    env_root = os.environ.get("SPARSEVIDEO_NATIVE_KERNEL_ROOT")
    if env_root:
        dirs.append(Path(env_root).expanduser())

    repo_root = _repo_root()
    dirs.append(repo_root / "src" / "sparsevideo" / "kernels" / "native" / "build")
    return dirs


def optional_kernel_status() -> Dict[str, Any]:
    """Report optional sparse kernel availability without importing them.

    This is intentionally conservative. It answers whether an optional package
    or compiled artifact appears to be installed; actual CUDA execution still
    must be validated by a real inference smoke on a GPU node.
    """
    spas_locations = _package_locations("spas_sage_attn")
    fastvideo_locations = _package_locations("fastvideo_kernel")
    flashomni_locations = _package_locations("flashomni")
    flashinfer_locations = _package_locations("flashinfer")
    native_kernel_files = _glob_existing(_native_kernel_dirs(), "_kernels*.so")
    repo_root = _repo_root()
    native_source_root = repo_root / "src" / "sparsevideo" / "kernels" / "native"

    return {
        "spas_sage_attn": {
            "package": bool(spas_locations),
            "qattn_extension": bool(_glob_existing(spas_locations, "_qattn*.so")),
            "fused_extension": bool(_glob_existing(spas_locations, "_fused*.so")),
            "methods": ["spargeattn"],
        },
        "fastvideo_kernel": {
            "package": bool(fastvideo_locations),
            "native_extension": bool(
                _glob_any_existing(
                    fastvideo_locations,
                    ("_C/*.so", "**/_C*.so", "**/*fastvideo_kernel_ops*.so"),
                )
            ),
            "package_locations": [str(path) for path in fastvideo_locations],
            "methods": ["sta"],
        },
        "flashomni": {
            "package": bool(flashomni_locations),
            "aot_config": bool(_glob_existing(flashomni_locations, "jit/aot_config.py")),
            "native_extension": bool(_glob_existing(flashomni_locations, "flashomni_kernels*.so")),
            "package_locations": [str(path) for path in flashomni_locations],
            "methods": ["flashomni"],
        },
        "flashinfer": {
            "package": bool(flashinfer_locations),
            "sparse_module": _has_submodule_file(flashinfer_locations, "sparse"),
            "methods": ["adacluster", "draft", "radial", "svg2", "svoo"],
        },
        "svg_svoo_fused_kernels": {
            "native_extension": bool(native_kernel_files),
            "backend_env": os.environ.get("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "auto"),
            "candidate_dirs": [str(path) for path in _native_kernel_dirs()],
            "files": native_kernel_files,
            "source": _source_dir_status(
                native_source_root,
                ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/CMakeLists.txt"),
            ),
            "methods": ["svg1", "svg2", "svoo"],
        },
    }


def torch_runtime_status() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "imported": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return {
            "imported": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": False,
            "cuda_error_type": type(exc).__name__,
            "cuda_error": str(exc),
        }

    device_count = 0
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0

    return {
        "imported": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
    }
