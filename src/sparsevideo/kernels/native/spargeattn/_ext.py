"""CUDAExtension definitions for spargeattn (spas_sage_attn).

Called by the root setup.py during `pip install . --no-build-isolation`.
Compiled .so files land at:
  sparsevideo/kernels/native/spargeattn/spas_sage_attn/_qattn.<abi>.so
  sparsevideo/kernels/native/spargeattn/spas_sage_attn/_fused.<abi>.so

spas_sage_runtime.py finds them by adding native/spargeattn/ to sys.path
and importing spas_sage_attn by name.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"


def _run_instantiations(subdir: Path) -> None:
    """Run Python generator scripts to create .cu instantiation files."""
    if not subdir.exists():
        return
    for py_file in sorted(subdir.rglob("*.py")):
        subprocess.run(
            [sys.executable, str(py_file)],
            cwd=str(py_file.parent),
            check=False,
        )


def _cu_files(subdir: Path) -> list[str]:
    if not subdir.exists():
        return []
    return sorted(str(f) for f in subdir.rglob("*.cu"))


def get_extensions() -> list:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    from packaging.version import Version

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME not found; cannot build spargeattn")

    # Compute capability detection
    caps: set[str] = set()
    env_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch:
        for a in env_arch.replace(" ", ";").split(";"):
            a = a.replace("+PTX", "").strip()
            if a:
                caps.add(a)
    else:
        for i in range(torch.cuda.device_count()):
            maj, mn = torch.cuda.get_device_capability(i)
            if maj >= 8:
                caps.add(f"{maj}.{mn}")
    if not caps:
        raise RuntimeError("No CUDA device with compute capability >= 8.0")

    # NVCC version check
    nvcc_bin = os.path.join(CUDA_HOME, "bin", "nvcc")
    raw = subprocess.check_output([nvcc_bin, "-V"], text=True)
    m = re.search(r"release (\d+\.\d+),", raw)
    nvcc_ver = Version(m.group(1)) if m else Version("12.0")
    if nvcc_ver < Version("12.0"):
        raise RuntimeError("spargeattn requires CUDA 12.0+")

    has_sm90 = any(c.startswith("9.0") for c in caps)

    # Generate instantiation .cu files before listing sources
    _run_instantiations(CSRC / "qattn" / "instantiations_sm80")
    _run_instantiations(CSRC / "qattn" / "instantiations_sm89")
    if has_sm90:
        _run_instantiations(CSRC / "qattn" / "instantiations_sm90")

    # Arch flags
    arch_flags: list[str] = []
    for cap in sorted(caps):
        num = cap.replace(".", "")
        if num == "90":
            num = "90a"
        arch_flags += ["-gencode", f"arch=compute_{num},code=sm_{num}"]

    abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    cxx_flags = [
        "-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17",
        "-DENABLE_BF16", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]
    nvcc_flags = [
        "-O3", "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math", "--threads=8",
        "-Xptxas=-v", "-diag-suppress=174",
        "-Xcompiler", "-include,cassert",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ] + arch_flags

    qattn_sources = [
        str(CSRC / "qattn" / "pybind.cpp"),
        str(CSRC / "qattn" / "qk_int_sv_f16_cuda_sm80.cu"),
        str(CSRC / "qattn" / "qk_int_sv_f8_cuda_sm89.cu"),
    ]
    qattn_sources += _cu_files(CSRC / "qattn" / "instantiations_sm80")
    qattn_sources += _cu_files(CSRC / "qattn" / "instantiations_sm89")
    if has_sm90:
        sm90_cu = CSRC / "qattn" / "qk_int_sv_f8_cuda_sm90.cu"
        if sm90_cu.exists():
            qattn_sources.append(str(sm90_cu))
        qattn_sources += _cu_files(CSRC / "qattn" / "instantiations_sm90")

    return [
        CUDAExtension(
            name="sparsevideo.kernels.native.spargeattn.spas_sage_attn._qattn",
            sources=qattn_sources,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            extra_link_args=["-lcuda"],
        ),
        CUDAExtension(
            name="sparsevideo.kernels.native.spargeattn.spas_sage_attn._fused",
            sources=[
                str(CSRC / "fused" / "pybind.cpp"),
                str(CSRC / "fused" / "fused.cu"),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
    ]
