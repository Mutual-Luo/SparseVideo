"""CUDAExtension definition for draft_block_sparse (block_sparse_attn_cuda).

Called by the root setup.py during `pip install . --no-build-isolation`.
Compiled .so lands at:
  sparsevideo/kernels/native/draft_block_sparse/block_sparse_attn_cuda.<abi>.so

draft_block_sparse_runtime.py finds it by adding native/draft_block_sparse/ to
sys.path and importing block_sparse_attn_cuda by name.

Always builds in draft_inference mode: only the hdim=128 forward block-sparse
.cu files are compiled (the only path ported to CUTLASS 4.x that SparseVideo
uses).
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
        raise RuntimeError("CUDA_HOME not found; cannot build draft_block_sparse")

    # CUTLASS is required: resolved by _cutlass (env / vendored / fetched)
    from sparsevideo.kernels._cutlass import cutlass_root
    cutlass = cutlass_root("draft_block_sparse")
    cutlass_include = cutlass / "include"
    if not cutlass_include.exists():
        raise RuntimeError(
            f"draft_block_sparse CUTLASS headers not found at {cutlass_include}. "
            "Run sparsevideo-build-kernels or set SPARSEVIDEO_CUTLASS_DIR."
        )

    # NVCC version check
    nvcc_bin = os.path.join(CUDA_HOME, "bin", "nvcc")
    raw = subprocess.check_output([nvcc_bin, "-V"], text=True)
    m = re.search(r"release (\d+\.\d+),", raw)
    nvcc_ver = Version(m.group(1)) if m else Version("12.0")
    if nvcc_ver < Version("11.6"):
        raise RuntimeError("draft_block_sparse requires CUDA 11.6+")

    # Check for legacy ATen generator path
    torch_dir = torch.__path__[0]
    generator_flag = []
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    # Arch flags (SM80 default; SM90 opt-in via env var, requires CUDA 11.8+)
    archs = set(os.getenv("BLOCK_SPARSE_ATTN_CUDA_ARCHS", "80").split(";"))
    cc_flags: list[str] = []
    if "80" in archs:
        cc_flags += ["-gencode", "arch=compute_80,code=sm_80"]
    if "90" in archs and nvcc_ver >= Version("11.8"):
        cc_flags += ["-gencode", "arch=compute_90,code=sm_90"]

    draft_flags = ["-DSPARSEVIDEO_DRAFT_INFERENCE_ONLY"]
    nvcc_threads = os.environ.get("NVCC_THREADS", "4")

    # draft_inference mode: only the two hdim=128 forward block-sparse files
    src_dir = CSRC / "block_sparse_attn" / "src"
    sources = [
        str(CSRC / "block_sparse_attn" / "flash_api.cpp"),
        str(src_dir / "flash_fwd_block_hdim128_fp16_sm80.cu"),
        str(src_dir / "flash_fwd_block_hdim128_bf16_sm80.cu"),
    ]

    return [
        CUDAExtension(
            name="sparsevideo.kernels.native.draft_block_sparse.block_sparse_attn_cuda",
            sources=sources,
            # flash_api.cpp does #include "flash.h", which lives in src/; the .cu
            # sources include their sibling headers there too.
            include_dirs=[str(src_dir), str(cutlass_include)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag + draft_flags,
                "nvcc": [
                    "-O3", "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-lineinfo",
                    f"--threads={nvcc_threads}",
                ] + generator_flag + cc_flags + draft_flags,
            },
        )
    ]
