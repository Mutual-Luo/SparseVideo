from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = Path(os.environ.get("BUILD_DIR", ROOT.parent / "build")).resolve()


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"{name} is required to build SparseVideo native fused kernels")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def main() -> None:
    path_entries = [str(Path(sys.executable).resolve().parent)]
    try:
        import ninja

        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir:
            path_entries.append(str(bin_dir))
    except Exception:
        pass
    os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])

    flashinfer_dir = _package_dir("flashinfer")
    include_paths = [
        ROOT / "csrc",
        ROOT / "include",
        flashinfer_dir / "data" / "include",
        flashinfer_dir / "data" / "cutlass" / "include",
    ]
    for include_path in include_paths:
        if not include_path.exists():
            raise RuntimeError(f"required include path does not exist: {include_path}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    module = load(
        name="_kernels",
        sources=[str(ROOT / "csrc" / "ops.cu")],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cuda_cflags=[
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "--disable-warnings",
        ],
        extra_cflags=["-w"],
        build_directory=str(BUILD_DIR),
        verbose=True,
        keep_intermediates=True,
    )
    print(module.__file__)


if __name__ == "__main__":
    main()
