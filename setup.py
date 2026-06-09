"""
SparseVideo unified build entry point.

When torch + CUDA are available, all native CUDA/C++ kernel extensions are
compiled and included in the installed package.  Use:

    pip install . --no-build-isolation

The --no-build-isolation flag lets the build reuse the already-installed torch
(required so nvcc links against the correct libcuda/libtorch ABI).

Without CUDA (or without --no-build-isolation), only the Python package is
installed.  Sparse methods backed by Triton or sparsevideo_flashinfer (adacluster,
sta, svg1, svg2, svoo) remain functional via JIT.  Methods that require native CUDA
kernels (spargeattn, draft, flashomni, svg/svoo fused ops) raise ImportError at
runtime with a clear build instruction.

Individual kernels can also be rebuilt without reinstalling:
    sparsevideo-build-kernels          # rebuild all
    sparsevideo-build-kernels -e draft # rebuild one
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Expose src/ so _ext.py modules can import sparsevideo.kernels.*
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from setuptools import setup, find_packages


# ---------------------------------------------------------------------------
# nvcc / CUDA_HOME resolution
#
# torch.utils.cpp_extension resolves CUDA_HOME exactly once at import time.
# We must set it *before* that import so the extension build finds the right
# nvcc.  Detection order:
#   1. $CUDA_HOME / $CUDA_PATH already set by the user — respect it.
#   2. nvcc sitting next to the current Python interpreter (covers the common
#      case where the user installed cuda-nvcc into their venv/conda env).
#   3. nvcc somewhere on $PATH.
# ---------------------------------------------------------------------------

def _find_nvcc() -> str | None:
    """Return the path to nvcc, or None if not found."""
    # 1. Explicit env override
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root:
            candidate = Path(root) / "bin" / "nvcc"
            if candidate.is_file():
                return str(candidate)

    # 2. Same bin dir as the current Python interpreter (venv / conda / pyenv)
    interpreter_bin = Path(sys.executable).resolve().parent
    candidate = interpreter_bin / "nvcc"
    if candidate.is_file():
        return str(candidate)

    # 3. PATH
    return shutil.which("nvcc")


def _set_cuda_home_from_nvcc(nvcc: str) -> None:
    """Derive CUDA_HOME from nvcc path and export it so cpp_extension finds it."""
    nvcc_bin = Path(nvcc).resolve().parent          # .../bin/
    cuda_root = nvcc_bin.parent                     # one level up
    os.environ.setdefault("CUDA_HOME", str(cuda_root))
    os.environ.setdefault("CUDA_PATH", str(cuda_root))


def _preflight(nvcc: str) -> None:
    """Validate the build environment and abort early with clear guidance."""
    errors: list[str] = []

    # --- nvcc version ---
    try:
        raw = subprocess.check_output([nvcc, "--version"], text=True)
        m = re.search(r"release (\d+)\.(\d+)", raw)
        nvcc_major = int(m.group(1)) if m else None
        nvcc_minor = int(m.group(2)) if m else None
        nvcc_ver_str = f"{nvcc_major}.{nvcc_minor}" if nvcc_major is not None else "unknown"
    except Exception as exc:
        errors.append(f"Failed to run nvcc at {nvcc}: {exc}")
        nvcc_major = nvcc_minor = None
        nvcc_ver_str = "unknown"

    print(f"[sparsevideo] nvcc   : {nvcc}  (version {nvcc_ver_str})")

    # --- torch CUDA version ---
    try:
        import torch
        torch_cuda = torch.version.cuda or ""
        parts = torch_cuda.split(".")
        torch_cuda_major = int(parts[0]) if parts else None
        torch_cuda_minor = int(parts[1]) if len(parts) > 1 else None
        print(f"[sparsevideo] torch  : {torch.__version__}  (built with CUDA {torch_cuda})")
    except Exception:
        torch_cuda_major = torch_cuda_minor = None
        torch_cuda = "unknown"

    # --- torch must be a CUDA build ---
    if torch_cuda_major is None:
        errors.append(
            "torch is not a CUDA build (torch.version.cuda is empty), so the native "
            "kernels cannot be compiled.\n"
            "  Install a CUDA build of torch matching your toolkit, e.g.\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    # --- major-version mismatch ---
    if nvcc_major is not None and torch_cuda_major is not None:
        if nvcc_major != torch_cuda_major:
            errors.append(
                f"CUDA major version mismatch:\n"
                f"  nvcc   : {nvcc_ver_str}   (at {nvcc})\n"
                f"  torch  : built with CUDA {torch_cuda}\n"
                f"  Fix: install nvcc matching your torch CUDA version, e.g.\n"
                f"    conda install -c nvidia/label/cuda-{torch_cuda} cuda-nvcc\n"
                f"  or rebuild torch for CUDA {nvcc_major}.{nvcc_minor}."
            )
        elif torch_cuda_minor is not None and nvcc_minor is not None:
            if abs(nvcc_minor - torch_cuda_minor) > 2:
                # Minor mismatch: warn but don't fail — usually still compiles fine
                print(
                    f"[sparsevideo] WARNING: nvcc {nvcc_ver_str} vs torch CUDA {torch_cuda} "
                    f"— minor version gap > 2, compilation may fail."
                )

    # --- torch version ---
    try:
        from packaging.version import Version
        torch_ver = Version(torch.__version__.split("+")[0])
        if torch_ver < Version("2.1.0"):
            errors.append(
                f"torch >= 2.1.0 required, found {torch.__version__}.\n"
                f"  pip install 'torch>=2.1.0'"
            )
    except Exception:
        pass

    # --- GPU compute capability ---
    try:
        import torch
        if torch.cuda.is_available():
            caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
            best = max(caps)
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(f"[sparsevideo] GPU(s) : {', '.join(names)}  (best sm_{best[0]}{best[1]})")
            if best < (8, 0):
                errors.append(
                    f"SparseVideo kernels require compute capability >= 8.0 "
                    f"(Ampere A100 / RTX 30xx or newer).\n"
                    f"  Best visible GPU: sm_{best[0]}{best[1]}"
                )
    except Exception:
        pass

    if errors:
        border = "=" * 70
        msg = f"\n{border}\n[sparsevideo] Build preflight FAILED — fix the issues below first\n{border}\n"
        for i, e in enumerate(errors, 1):
            msg += f"\n  [{i}] {e}\n"
        msg += f"\n{border}\n"
        raise SystemExit(msg)

    print("[sparsevideo] Preflight OK — starting kernel compilation.")


# ---------------------------------------------------------------------------
# Detect nvcc and set CUDA_HOME *before* importing torch.utils.cpp_extension
# (cpp_extension resolves CUDA_HOME exactly once at module import time).
# ---------------------------------------------------------------------------

ext_modules: list = []
cmdclass: dict = {}

# This is a CUDA/C++ extension package: by default the native kernels are compiled
# at install time (like flash-attn). The build environment is validated up front
# (torch present, nvcc found, nvcc/torch CUDA versions compatible, GPU arch >= sm80)
# and the install aborts early with actionable guidance if anything is wrong.
# Set SPARSEVIDEO_SKIP_CUDA_BUILD=1 to install the pure-Python layer only (Triton/
# JIT methods still work; native-CUDA methods raise a clear ImportError at runtime).
_skip_cuda = os.environ.get("SPARSEVIDEO_SKIP_CUDA_BUILD") == "1"
_build_cmds = ("build", "build_ext", "bdist_wheel", "install", "develop", "editable_wheel")
_building = any(cmd in sys.argv for cmd in _build_cmds)

if _building and not _skip_cuda:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\n[sparsevideo] torch is required to compile the native kernels but is not "
            "installed.\n"
            "  Install a CUDA build of torch first, then install with build isolation off:\n"
            "    pip install torch\n"
            "    pip install sparsevideo --no-build-isolation\n"
            "  (Or set SPARSEVIDEO_SKIP_CUDA_BUILD=1 to install the pure-Python layer only.)\n"
        )

    _nvcc = _find_nvcc()
    if _nvcc:
        _set_cuda_home_from_nvcc(_nvcc)

    from torch.utils.cpp_extension import BuildExtension

    if _nvcc is None:
        raise SystemExit(
            "\n[sparsevideo] nvcc not found — cannot compile the native kernels.\n"
            "  Install the CUDA compiler into your current Python environment, e.g.:\n"
            "    conda install -c nvidia cuda-nvcc\n"
            "  or set CUDA_HOME to your CUDA toolkit root.\n"
            "  (Or set SPARSEVIDEO_SKIP_CUDA_BUILD=1 to install the pure-Python layer only.)\n"
        )

    _preflight(_nvcc)

    for _kernel in (
        "svg_svoo_fused",
        "spargeattn",
        "sageattention",
        "draft_block_sparse",
        "flashomni",
    ):
        _mod = __import__(
            f"sparsevideo.kernels.native.{_kernel}._ext",
            fromlist=["get_extensions"],
        )
        ext_modules += _mod.get_extensions()

    if ext_modules:
        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_tag=False)


def _native_package_data() -> list[str]:
    """Collect native kernel files for the wheel, excluding CUTLASS headers."""
    native = _SRC / "sparsevideo" / "kernels" / "native"
    pkg_base = _SRC / "sparsevideo"
    collected: set[str] = set()
    source_globs = [
        "**/*.cu", "**/*.cuh", "**/*.cpp", "**/*.cc",
        "**/*.h", "**/*.hpp", "**/*.inc",
        "**/*.json",
        "**/setup.py", "**/setup.sh", "**/build.py",
        "**/CMakeLists.txt", "**/Makefile",
        "**/LICENSE", "**/LICENSE.txt",
    ]
    for pattern in source_globs:
        for p in native.glob(pattern):
            if "cutlass" in p.parts:
                continue
            collected.add(str(p.relative_to(pkg_base)))
    # Prebuilt .so artifacts are platform/CUDA-specific and are never shipped in
    # the source distribution; build_ext produces them at install time.
    return sorted(collected)


def _methods_data() -> list[str]:
    """Collect per-method runtime data files loaded relative to ``__file__``.

    Each method's ``config.py`` reads its ``config.yaml`` at import time, and the
    svoo method reads its sparsity-profile CSVs at runtime. These must ship in the
    wheel or an installed (non-editable) package raises FileNotFoundError on use.
    """
    methods = _SRC / "sparsevideo" / "methods"
    pkg_base = _SRC / "sparsevideo"
    collected: set[str] = set()
    for pattern in ("**/config.yaml", "svoo/sparsity_profiles/*.csv"):
        for p in methods.glob(pattern):
            collected.add(str(p.relative_to(pkg_base)))
    return sorted(collected)


def _flashinfer_vendor_data() -> list[str]:
    """Collect vendored sparsevideo_flashinfer files for the wheel."""
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
        # Never ship prebuilt binaries; they are platform/CUDA-specific.
        if p.suffix in {".so", ".pyd", ".dll", ".dylib", ".o", ".a"}:
            continue
        collected.add(str(p.relative_to(pkg_base)))
    return sorted(collected)


setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "sparsevideo": _native_package_data() + _flashinfer_vendor_data() + _methods_data(),
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
