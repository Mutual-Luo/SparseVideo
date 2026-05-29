"""CUDAExtension definitions for sageattention.

Called by the root setup.py during `pip install . --no-build-isolation`.
Compiled .so files land at:
  sparsevideo/kernels/native/sageattention/sageattention/_qattn_sm80.<abi>.so
  sparsevideo/kernels/native/sageattention/sageattention/_qattn_sm89.<abi>.so  (SM89+)
  sparsevideo/kernels/native/sageattention/sageattention/_qattn_sm90.<abi>.so  (SM90)
  sparsevideo/kernels/native/sageattention/sageattention/_fused.<abi>.so

sageattention_runtime.py finds them by adding native/sageattention/ to sys.path
and importing sageattention by name.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"


def get_extensions() -> list:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME
    from packaging.version import Version

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME not found; cannot build sageattention")

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

    nvcc_bin = os.path.join(CUDA_HOME, "bin", "nvcc")
    raw = subprocess.check_output([nvcc_bin, "-V"], text=True)
    m = re.search(r"release (\d+\.\d+),", raw)
    nvcc_ver = Version(m.group(1)) if m else Version("12.0")
    if nvcc_ver < Version("12.0"):
        raise RuntimeError("sageattention requires CUDA 12.0+")

    has_sm80 = any(c.startswith(("8.0", "8.6", "8.7")) for c in caps)
    has_sm89 = any(c.startswith(("8.9", "12.0")) for c in caps)
    has_sm90 = any(c.startswith("9.0") for c in caps)

    if nvcc_ver < Version("12.4") and has_sm89:
        raise RuntimeError("sageattention SM89 requires CUDA 12.4+")
    if nvcc_ver < Version("12.3") and has_sm90:
        raise RuntimeError("sageattention SM90 requires CUDA 12.3+")

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
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ] + arch_flags

    exts = []

    if has_sm80 or has_sm89 or has_sm90:
        exts.append(CUDAExtension(
            name="sparsevideo.kernels.native.sageattention.sageattention._qattn_sm80",
            sources=[
                str(CSRC / "qattn" / "pybind_sm80.cpp"),
                str(CSRC / "qattn" / "qk_int_sv_f16_cuda_sm80.cu"),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ))

    if has_sm89:
        exts.append(CUDAExtension(
            name="sparsevideo.kernels.native.sageattention.sageattention._qattn_sm89",
            sources=[
                str(CSRC / "qattn" / "pybind_sm89.cpp"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f32_attn.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu"),
                str(CSRC / "qattn" / "sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ))

    if has_sm90:
        exts.append(CUDAExtension(
            name="sparsevideo.kernels.native.sageattention.sageattention._qattn_sm90",
            sources=[
                str(CSRC / "qattn" / "pybind_sm90.cpp"),
                str(CSRC / "qattn" / "qk_int_sv_f8_cuda_sm90.cu"),
            ],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
            extra_link_args=["-lcuda"],
        ))

    exts.append(CUDAExtension(
        name="sparsevideo.kernels.native.sageattention.sageattention._fused",
        sources=[
            str(CSRC / "fused" / "pybind.cpp"),
            str(CSRC / "fused" / "fused.cu"),
        ],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ))

    return exts
