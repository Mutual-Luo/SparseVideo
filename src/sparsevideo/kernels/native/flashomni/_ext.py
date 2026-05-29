"""CUDAExtension definition for flashomni (flashomni_kernels).

Called by the root setup.py during `pip install . --no-build-isolation`.
Compiled .so lands at:
  sparsevideo/kernels/native/flashomni/flashomni/flashomni_kernels.abi3.so

flashomni/method.py finds it by adding native/flashomni/ to sys.path and
importing flashomni (which loads flashomni_kernels internally).

AOT source generation runs here before returning extensions. This mirrors
what flashomni/setup.py does when FLASHOMNI_ENABLE_AOT=1.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
GEN_DIR = ROOT / "csrc" / "generated"


def _write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _generate_aot_sources() -> None:
    """Run flashomni AOT code generation (produces .cu files and headers)."""
    _added = str(ROOT) not in sys.path
    if _added:
        sys.path.insert(0, str(ROOT))
    try:
        from aot_build_utils import generate_dispatch_inc
        from aot_build_utils.generate import get_instantiation_cu
        from aot_build_utils.generate_aot_default_additional_params_header import (
            get_aot_default_additional_params_header_str,
        )
    finally:
        if _added and str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))

    head_dims = list(map(int, os.environ.get("FLASHOMNI_HEAD_DIMS", "64,128,256").split(",")))
    mask_modes = [0]

    def _flag(key: str) -> bool:
        return os.environ.get(key, "1") == "1"

    _write_if_different(
        GEN_DIR / "dispatch.inc",
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=head_dims,
                head_dims_sm90=[],
                pos_encoding_modes=[0],
                use_fp16_qk_reductions=[0],
                mask_modes=mask_modes,
            )
        ),
    )

    aot_kernel_uris = get_instantiation_cu(
        argparse.Namespace(
            path=GEN_DIR,
            head_dims=head_dims,
            pos_encoding_modes=[0],
            use_fp16_qk_reductions=[0],
            mask_modes=mask_modes,
            enable_f16=_flag("FLASHOMNI_ENABLE_F16"),
            enable_bf16=_flag("FLASHOMNI_ENABLE_BF16"),
            enable_int8=_flag("FLASHOMNI_ENABLE_INT8"),
            enable_uint8=_flag("FLASHOMNI_ENABLE_UINT8"),
            enable_fp8_e4m3=_flag("FLASHOMNI_ENABLE_FP8_E4M3"),
            enable_fp8_e5m2=_flag("FLASHOMNI_ENABLE_FP8_E5M2"),
        )
    )

    _write_if_different(
        ROOT / "flashomni" / "jit" / "aot_config.py",
        f"prebuilt_ops_uri = set({aot_kernel_uris})",
    )
    _write_if_different(
        ROOT / "csrc" / "aot_default_additional_params.h",
        get_aot_default_additional_params_header_str(),
    )


def get_extensions() -> list:
    import torch.utils.cpp_extension as torch_cpp_ext
    from torch.utils.cpp_extension import CUDAExtension

    # CUTLASS is required (v4.3.0)
    from sparsevideo.kernels._cutlass import cutlass_root
    cutlass = cutlass_root("flashomni")
    include_dirs = [
        str(ROOT / "include"),
        str(cutlass / "include"),
        str(cutlass / "tools" / "util" / "include"),
    ]

    # Run AOT code generation: creates csrc/generated/*.cu and dispatch.inc
    _generate_aot_sources()

    sparseFA_sources = [
        str(f) for f in GEN_DIR.glob("*sparseFA_head*.cu") if "_sm90" not in f.name
    ]
    kernel_sources = [
        str(ROOT / "csrc" / "batch_sparseFA.cu"),
        str(ROOT / "csrc" / "gemm.cu"),
        str(ROOT / "csrc" / "gemm_reduction.cu"),
        str(ROOT / "csrc" / "quantization.cu"),
        str(ROOT / "csrc" / "flashomni_ops.cu"),
    ]

    def _flag(key: str) -> bool:
        return os.environ.get(key, "1") == "1"

    feature_flags = [
        f for f, en in [
            ("-DFLASHOMNI_ENABLE_F16",     _flag("FLASHOMNI_ENABLE_F16")),
            ("-DFLASHOMNI_ENABLE_BF16",    _flag("FLASHOMNI_ENABLE_BF16")),
            ("-DFLASHOMNI_ENABLE_INT8",    _flag("FLASHOMNI_ENABLE_INT8")),
            ("-DFLASHOMNI_ENABLE_UINT8",   _flag("FLASHOMNI_ENABLE_UINT8")),
            ("-DFLASHOMNI_ENABLE_FP8_E4M3", _flag("FLASHOMNI_ENABLE_FP8_E4M3")),
            ("-DFLASHOMNI_ENABLE_FP8_E5M2", _flag("FLASHOMNI_ENABLE_FP8_E5M2")),
        ] if en
    ]

    return [
        CUDAExtension(
            name="sparsevideo.kernels.native.flashomni.flashomni.flashomni_kernels",
            sources=kernel_sources + sparseFA_sources,
            include_dirs=include_dirs,
            libraries=["cublas", "cublasLt"],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-switch-bool", "-DPy_LIMITED_API=0x03080000"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    # Undo torch's default half-operator disable flags
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--threads=1",
                    "-Xfatbin", "-compress-all",
                    "-use_fast_math", "-DNDEBUG",
                    "-DPy_LIMITED_API=0x03080000",
                ] + feature_flags,
            },
            py_limited_api=True,
        )
    ]
