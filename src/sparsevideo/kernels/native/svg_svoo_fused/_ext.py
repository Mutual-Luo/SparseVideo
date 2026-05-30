"""CUDAExtension definition for svg_svoo_fused (_kernels).

Called by the root setup.py during `pip install . --no-build-isolation`.
The compiled .so lands at:
  sparsevideo/kernels/native/svg_svoo_fused/_kernels.<abi>.so

fused_norm_rope.py finds it by adding that directory to sys.path and
importing _kernels by name.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# flashinfer headers (include/) live next to our vendored sparsevideo_flashinfer package
_FLASHINFER_INCLUDE = (
    Path(__file__).resolve().parents[3] / "_flashinfer" / "include"
)


def get_extensions() -> list:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from sparsevideo.kernels._cutlass import cutlass_root
    cutlass = cutlass_root("svg_svoo_fused")

    from torch.utils.cpp_extension import CUDAExtension

    return [
        CUDAExtension(
            name="sparsevideo.kernels.native.svg_svoo_fused._kernels",
            sources=[str(ROOT / "csrc" / "ops.cu")],
            include_dirs=[
                str(ROOT / "csrc"),
                str(ROOT / "include"),
                str(_FLASHINFER_INCLUDE),
                str(cutlass / "include"),
            ],
            extra_compile_args={
                "cxx": ["-w"],
                "nvcc": [
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-extended-lambda",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                    "--disable-warnings",
                ],
            },
        )
    ]
