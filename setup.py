"""
SparseVideo unified build entry point.

When torch + CUDA are available, all native CUDA/C++ kernel extensions are
compiled and included in the installed package.  Use:

    pip install . --no-build-isolation

The --no-build-isolation flag lets the build reuse the already-installed torch
(required so nvcc links against the correct libcuda/libtorch ABI).

Without CUDA (or without --no-build-isolation), only the Python package is
installed.  Sparse methods backed by Triton or FlashInfer (adacluster, sta,
svg1, svg2, svoo) remain fully functional.  Methods that require native CUDA
kernels (spargeattn, draft, flashomni, svg/svoo fused ops) raise ImportError
at runtime with a clear build instruction.

Individual kernels can also be rebuilt without reinstalling:
    sparsevideo-build-kernels          # rebuild all
    sparsevideo-build-kernels -e draft # rebuild one
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Expose src/ so _ext.py modules can import sparsevideo.kernels.*
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from setuptools import setup, find_packages

ext_modules: list = []
cmdclass: dict = {}

_cuda_ready = False
try:
    import torch
    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1":
        from torch.utils.cpp_extension import BuildExtension
        _cuda_ready = True
except Exception:
    pass

if _cuda_ready:
    _skipped: list[str] = []
    for _kernel in (
        "svg_svoo_fused",
        "spargeattn",
        "sageattention",
        "draft_block_sparse",
        "flashomni",
    ):
        try:
            _mod = __import__(
                f"sparsevideo.kernels.native.{_kernel}._ext",
                fromlist=["get_extensions"],
            )
            ext_modules += _mod.get_extensions()
        except Exception as _e:
            _skipped.append(f"  {_kernel}: {_e}")

    if _skipped:
        print("[sparsevideo] WARNING: some kernels were skipped:")
        for _msg in _skipped:
            print(_msg)

    if ext_modules:
        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_tag=False)


def _native_package_data() -> list[str]:
    """Collect native kernel files for the wheel, excluding CUTLASS headers.

    CUTLASS is a build-time header-only dependency (~1000 .h files).  Including
    it would bloat every wheel by tens of MB.  It is fetched at build time via
    _cutlass.py and never needs to ship with the installed package.
    """
    native = _SRC / "sparsevideo" / "kernels" / "native"
    pkg_base = _SRC / "sparsevideo"
    collected: set[str] = set()

    # Source files needed for standalone kernel rebuilds (exclude cutlass dirs)
    source_globs = [
        "**/*.cu", "**/*.cuh", "**/*.cpp",
        "**/*.h",   # kernel headers only; cutlass filtered below
        "**/*.json",
        "**/setup.py", "**/setup.sh", "**/build.py",
        "**/CMakeLists.txt", "**/Makefile",
    ]
    for pattern in source_globs:
        for p in native.glob(pattern):
            if "cutlass" in p.parts:
                continue
            collected.add(str(p.relative_to(pkg_base)))

    # Compiled extensions produced by BuildExtension
    for p in native.glob("**/*.so"):
        if "cutlass" not in p.parts:
            collected.add(str(p.relative_to(pkg_base)))

    return sorted(collected)


def _flashinfer_vendor_data() -> list[str]:
    """Collect vendored flashinfer 0.2.x files for the wheel.

    Includes the Python package, csrc/, include/, and spdlog/ headers.
    cutlass is excluded (fetched at build time via _cutlass.py).
    """
    vendor = _SRC / "sparsevideo" / "kernels" / "_flashinfer"
    pkg_base = _SRC / "sparsevideo"
    if not vendor.exists():
        return []
    collected: set[str] = set()
    for p in vendor.rglob("*"):
        if not p.is_file():
            continue
        if "cutlass" in p.parts or "__pycache__" in p.parts:
            continue
        collected.add(str(p.relative_to(pkg_base)))
    return sorted(collected)


setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "sparsevideo": _native_package_data() + _flashinfer_vendor_data() + ["methods/sta/*.json"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
