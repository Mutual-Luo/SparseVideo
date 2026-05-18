"""Unified build entry point for SparseVideo native CUDA/C++ extensions.

Builds all native kernel extensions and caches compiled .so files.
Can be invoked as:
  - CLI: sparsevideo-build-kernels
  - Python: from sparsevideo.kernels._build import main; main()

Each sub-extension has its own setup.py or build.py. This module orchestrates
them and reports per-kernel build status.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


NATIVE_ROOT = Path(__file__).resolve().parent / "native"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sparsevideo" / "kernels"

EXTENSIONS = {
    "svg_svoo_fused": {
        "description": "Fused LayerNorm + RoPE kernels (SVG/SVOO)",
        "build_script": "build.py",
        "requires": ["flashinfer-python"],
    },
    "spargeattn": {
        "description": "SpargeAttn quantized sparse attention (CUDA + Triton)",
        "build_script": "setup.py",
        "setup_cmd": ["install", "--prefix", "{build_dir}"],
    },
    "sageattention": {
        "description": "SageAttention quantized attention (CUDA)",
        "build_script": "setup.py",
        "setup_cmd": ["install", "--prefix", "{build_dir}"],
    },
    "draft_block_sparse": {
        "description": "MIT Block-Sparse-Attention for Draft method (CUDA)",
        "build_script": "setup.py",
        "setup_cmd": ["install", "--prefix", "{build_dir}"],
    },
    "flashomni": {
        "description": "FlashOmni sparse attention + GEMM (CUDA)",
        "build_script": "setup.py",
        "setup_cmd": ["install", "--prefix", "{build_dir}"],
    },
    "sta_h100": {
        "description": "FastVideo STA Hopper/H100 C++ extension",
        "build_script": None,
        "note": "Requires H100 hardware; build manually with cmake in sta_h100/",
    },
}


def _check_cuda_available() -> tuple[bool, str]:
    """Check if CUDA toolkit is available for building extensions."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cuda_home:
            nvcc = subprocess.run(
                ["which", "nvcc"], capture_output=True, text=True
            )
            if nvcc.returncode != 0:
                return False, "CUDA_HOME not set and nvcc not found in PATH"
        return True, "ok"
    except ImportError:
        return False, "torch not installed"


def _build_svg_svoo_fused(ext_dir: Path, build_dir: Path, verbose: bool) -> tuple[bool, str]:
    """Build svg_svoo_fused via its build.py (uses torch.utils.cpp_extension.load)."""
    build_script = ext_dir / "build.py"
    env = os.environ.copy()
    env["BUILD_DIR"] = str(build_dir / "svg_svoo_fused")
    result = subprocess.run(
        [sys.executable, str(build_script)],
        cwd=str(ext_dir),
        env=env,
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr if not verbose else ""
        return False, f"build.py failed (rc={result.returncode}): {err[:500]}"
    return True, "ok"


def _build_setup_py(ext_dir: Path, build_dir: Path, verbose: bool) -> tuple[bool, str]:
    """Build a kernel via its setup.py install."""
    setup_script = ext_dir / "setup.py"
    target_dir = build_dir / ext_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MAX_JOBS", str(os.cpu_count() or 4))

    result = subprocess.run(
        [
            sys.executable, str(setup_script),
            "build_ext", "--inplace",
        ],
        cwd=str(ext_dir),
        env=env,
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr if not verbose else ""
        return False, f"setup.py build_ext failed (rc={result.returncode}): {err[:500]}"
    return True, "ok"


def build_extension(
    name: str,
    build_dir: Path | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Build a single native extension by name."""
    if name not in EXTENSIONS:
        return False, f"unknown extension: {name}. Available: {list(EXTENSIONS.keys())}"

    ext_info = EXTENSIONS[name]
    ext_dir = NATIVE_ROOT / name

    if not ext_dir.exists():
        return False, f"source directory not found: {ext_dir}"

    if ext_info.get("build_script") is None:
        return False, ext_info.get("note", "no build script available")

    if build_dir is None:
        build_dir = NATIVE_ROOT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    if name == "svg_svoo_fused":
        return _build_svg_svoo_fused(ext_dir, build_dir, verbose)
    elif ext_info.get("build_script") == "setup.py":
        return _build_setup_py(ext_dir, build_dir, verbose)
    else:
        return False, f"unsupported build_script: {ext_info['build_script']}"


def build_all(
    build_dir: Path | None = None,
    verbose: bool = False,
    skip: list[str] | None = None,
) -> dict[str, tuple[bool, str]]:
    """Build all native extensions. Returns {name: (success, message)}."""
    results = {}
    skip = skip or []

    cuda_ok, cuda_msg = _check_cuda_available()
    if not cuda_ok:
        print(f"CUDA not available: {cuda_msg}")
        print("Skipping all native kernel builds.")
        for name in EXTENSIONS:
            results[name] = (False, f"skipped: {cuda_msg}")
        return results

    for name in EXTENSIONS:
        if name in skip:
            results[name] = (False, "skipped by user")
            continue

        ext_info = EXTENSIONS[name]
        if ext_info.get("build_script") is None:
            results[name] = (False, ext_info.get("note", "no build script"))
            continue

        print(f"Building {name}: {ext_info['description']}...")
        ok, msg = build_extension(name, build_dir=build_dir, verbose=verbose)
        results[name] = (ok, msg)
        status = "OK" if ok else f"FAILED: {msg}"
        print(f"  {name}: {status}")

    return results


def status() -> dict[str, str]:
    """Report which native extensions are already built and loadable."""
    report = {}

    build_dir = NATIVE_ROOT / "build"
    if build_dir.exists():
        kernels_so = list(build_dir.glob("_kernels*.so"))
        report["svg_svoo_fused"] = "built" if kernels_so else "not built"
    else:
        report["svg_svoo_fused"] = "not built"

    spargeattn_dir = NATIVE_ROOT / "spargeattn" / "spas_sage_attn"
    if spargeattn_dir.exists() and list(spargeattn_dir.glob("_qattn*.so")):
        report["spargeattn"] = "built"
    else:
        report["spargeattn"] = "not built"

    sage_dir = NATIVE_ROOT / "sageattention"
    if sage_dir.exists() and list(sage_dir.glob("**/*.so")):
        report["sageattention"] = "built"
    else:
        report["sageattention"] = "not built"

    draft_dir = NATIVE_ROOT / "draft_block_sparse"
    if draft_dir.exists() and list(draft_dir.glob("**/block_sparse_attn_cuda*.so")):
        report["draft_block_sparse"] = "built"
    else:
        report["draft_block_sparse"] = "not built"

    flashomni_dir = NATIVE_ROOT / "flashomni"
    if flashomni_dir.exists() and list(flashomni_dir.glob("**/*.so")):
        report["flashomni"] = "built"
    else:
        report["flashomni"] = "not built"

    sta_dir = NATIVE_ROOT / "sta_h100"
    if sta_dir.exists() and list(sta_dir.glob("**/fastvideo_kernel_ops*.so")):
        report["sta_h100"] = "built"
    else:
        report["sta_h100"] = "not built (requires H100)"

    return report


def main():
    """CLI entry point for sparsevideo-build-kernels."""
    parser = argparse.ArgumentParser(
        description="Build SparseVideo native CUDA/C++ kernel extensions",
    )
    parser.add_argument(
        "--extension", "-e",
        choices=list(EXTENSIONS.keys()),
        help="Build only this extension (default: build all)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help=f"Output directory for compiled extensions (default: {NATIVE_ROOT / 'build'})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full build output",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show build status of all extensions and exit",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Extensions to skip",
    )
    args = parser.parse_args()

    if args.status:
        print("SparseVideo Native Kernel Status:")
        print("-" * 40)
        for name, st in status().items():
            print(f"  {name:20s} {st}")
        return

    if args.extension:
        ok, msg = build_extension(args.extension, build_dir=args.build_dir, verbose=args.verbose)
        if ok:
            print(f"Successfully built: {args.extension}")
        else:
            print(f"Failed to build {args.extension}: {msg}", file=sys.stderr)
            sys.exit(1)
    else:
        results = build_all(build_dir=args.build_dir, verbose=args.verbose, skip=args.skip)
        print("\n" + "=" * 40)
        print("Build Summary:")
        print("=" * 40)
        any_failed = False
        for name, (ok, msg) in results.items():
            symbol = "+" if ok else "-"
            print(f"  [{symbol}] {name:20s} {msg}")
            if not ok and "skipped" not in msg.lower():
                any_failed = True
        if any_failed:
            print("\nSome extensions failed. Use --verbose for details.")
            print("Triton-based methods (adacluster, sta A100) work without native builds.")


if __name__ == "__main__":
    main()
