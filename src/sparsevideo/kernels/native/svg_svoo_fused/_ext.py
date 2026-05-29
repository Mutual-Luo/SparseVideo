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


def get_extensions() -> list:
    from torch.utils.cpp_extension import CUDAExtension

    try:
        import flashinfer
        fi_dir = Path(flashinfer.__file__).parent
    except ImportError as exc:
        raise ImportError(
            "svg_svoo_fused requires flashinfer-python. "
            "Install with: pip install sparsevideo[flashinfer]"
        ) from exc

    cutlass_include = fi_dir / "data" / "cutlass" / "include"
    if not cutlass_include.exists():
        raise RuntimeError(
            f"flashinfer CUTLASS headers not found at {cutlass_include}. "
            "Ensure flashinfer-python is fully installed."
        )

    return [
        CUDAExtension(
            name="sparsevideo.kernels.native.svg_svoo_fused._kernels",
            sources=[str(ROOT / "csrc" / "ops.cu")],
            include_dirs=[
                str(ROOT / "csrc"),
                str(ROOT / "include"),
                str(fi_dir / "data" / "include"),
                str(cutlass_include),
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
