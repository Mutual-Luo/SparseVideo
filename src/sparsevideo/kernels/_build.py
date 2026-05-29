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


def _inject_cutlass(name: str, env: dict[str, str]) -> None:
    """Resolve the pinned CUTLASS root for ``name`` and expose it to the build.

    No-op for kernels that don't use CUTLASS (e.g. svg_svoo_fused uses flashinfer's
    bundled copy) or when the caller already set ``SPARSEVIDEO_CUTLASS_DIR``.
    """
    from . import _cutlass

    if name not in _cutlass.CUTLASS_PINS or "SPARSEVIDEO_CUTLASS_DIR" in env:
        return
    env["SPARSEVIDEO_CUTLASS_DIR"] = str(_cutlass.cutlass_root(name))

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
        # SparseVideo's draft only uses the forward block-sparse hdim=128 path, which is
        # also the only path ported to CUTLASS 4.x. Build inference-only by default.
        "build_env": {"BLOCK_SPARSE_ATTN_BUILD_MODE": "draft_inference"},
        "artifact": ("", "block_sparse_attn_cuda*.so"),
    },
    "flashomni": {
        "description": "FlashOmni sparse attention + GEMM (CUDA)",
        "build_script": "setup.py",
        "setup_cmd": ["install", "--prefix", "{build_dir}"],
        # flashomni is flashinfer-derived: it JITs at runtime unless AOT is requested.
        # We must build the .so ahead of time so benchmark/quality runs do not pay a
        # first-call compile and so missing-kernel states fail loudly.
        "build_env": {"FLASHOMNI_ENABLE_AOT": "1"},
        "artifact": ("flashomni", "**/flashomni_kernels*.so"),
    },
}


def _nvcc_path() -> str | None:
    """Locate nvcc: prefer $CUDA_HOME/bin/nvcc, then PATH."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidate = Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    import shutil

    return shutil.which("nvcc")


def _nvcc_major(nvcc: str) -> str | None:
    """Parse the CUDA major version reported by ``nvcc --version`` (e.g. '12')."""
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    import re

    m = re.search(r"release (\d+)\.", out.stdout)
    return m.group(1) if m else None


def _check_cuda_available() -> tuple[bool, str]:
    """Preflight the build toolchain, failing loudly with actionable guidance.

    Verifies a CUDA torch, that nvcc is found, and that nvcc's CUDA major matches
    torch's. A mismatch (a common trap: a stray system nvcc shadowing the conda
    env's) is the usual cause of confusing compile errors, so we stop early.
    """
    try:
        import torch
    except ImportError:
        return False, "torch not installed; install torch with CUDA support"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False; install a CUDA build of torch"

    nvcc = _nvcc_path()
    if not nvcc:
        return False, (
            "nvcc not found. Set CUDA_HOME to your CUDA toolkit (e.g. your conda env), "
            "or put nvcc on PATH."
        )

    torch_cuda = (torch.version.cuda or "").split(".")[0]
    nvcc_cuda = _nvcc_major(nvcc)
    if torch_cuda and nvcc_cuda and torch_cuda != nvcc_cuda:
        return False, (
            f"CUDA major mismatch: torch is built for CUDA {torch.version.cuda} but "
            f"nvcc ({nvcc}) is CUDA {nvcc_cuda}.x. Use the matching nvcc (e.g. set "
            f"CUDA_HOME to your conda env) before building."
        )
    return True, f"ok (nvcc={nvcc}, CUDA {nvcc_cuda}.x, torch CUDA {torch.version.cuda})"


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
    # Parallel build jobs for ninja. Cap at 32: CUTLASS-heavy nvcc TUs each use
    # several GB of RAM, so using all cores (e.g. 128) risks OOM/thrashing.
    env.setdefault("MAX_JOBS", str(min(32, os.cpu_count() or 4)))
    _inject_cutlass(ext_dir.name, env)
    for key, value in EXTENSIONS.get(ext_dir.name, {}).get("build_env", {}).items():
        env.setdefault(key, value)

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
        ok, msg = _build_svg_svoo_fused(ext_dir, build_dir, verbose)
    elif ext_info.get("build_script") == "setup.py":
        ok, msg = _build_setup_py(ext_dir, build_dir, verbose)
    else:
        return False, f"unsupported build_script: {ext_info['build_script']}"

    # Guard against silent no-op builds (e.g. a setup.py that exits 0 without
    # producing its extension): a reported success must leave the artifact behind.
    artifact = ext_info.get("artifact")
    if ok and artifact:
        subdir, pattern = artifact
        if not list((ext_dir / subdir).glob(pattern)):
            return False, (
                f"build exited 0 but produced no artifact matching "
                f"{subdir}/{pattern}; check the build flags for {name}"
            )
    return ok, msg


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

    # Check both locations: pip-install path and sparsevideo-build-kernels path
    _svg_locations = [
        NATIVE_ROOT / "svg_svoo_fused",   # pip install --no-build-isolation
        NATIVE_ROOT / "build",            # sparsevideo-build-kernels
    ]
    report["svg_svoo_fused"] = "not built"
    for _d in _svg_locations:
        if _d.exists() and list(_d.glob("_kernels*.so")):
            report["svg_svoo_fused"] = "built"
            break

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
