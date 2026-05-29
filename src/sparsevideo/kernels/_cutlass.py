"""Resolve (and, if needed, fetch) the pinned CUTLASS headers for native builds.

CUTLASS is a build-time, header-only dependency. SparseVideo does not ship it in
the wheel; instead each cutlass-using kernel is pinned to the exact CUTLASS ref its
upstream validated, and that ref is resolved at build time.

Resolution order for a kernel (see ``cutlass_root``):

1. ``$SPARSEVIDEO_CUTLASS_DIR`` -- explicit global override (advanced / offline).
   Must satisfy the kernel being built; note flashomni needs CUTLASS 4.x
   while draft needs 3.x, so a single override cannot serve both at once.
2. Vendored copy in the repo, if present -- correct per-kernel version, offline.
3. Previously-fetched copy cached under ``~/.cache/sparsevideo/third_party``.
4. Fetch the pinned-ref source tarball from GitHub and cache it. Honors
   ``$SPARSEVIDEO_GITHUB_PROXY`` / ``$HTTPS_PROXY`` (this is how the build reaches
   GitHub on proxied networks).

A "cutlass root" is a directory containing ``include/`` (and, for flashomni,
``tools/util/include/``). Callers build their ``-I`` paths from the returned root.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

NATIVE_ROOT = Path(__file__).resolve().parent / "native"

_CACHE_ROOT = (
    Path(os.environ.get("SPARSEVIDEO_CACHE_DIR", Path.home() / ".cache" / "sparsevideo"))
    / "third_party"
)

# Pinned CUTLASS ref per kernel. All three are unified on the v4.3.0 tag: draft
# (MIT Block-Sparse-Attention) was ported from its original 3.x CuTe API to 4.x
# (TiledMMA PermutationMNK / tile_shape / prefetch include hygiene), validated
# bit-identical to the 3.x build with no speed regression.
CUTLASS_PINS: dict[str, str] = {
    "flashomni": "v4.3.0",
    "draft_block_sparse": "v4.3.0",
}

# Vendored cutlass root per kernel (each contains an ``include/`` subdir). Used as the
# offline fallback while the vendored copies are still present in the tree. draft now
# shares the flashomni 4.3.0 copy (its own 3.x checkout is obsolete after the port).
_VENDORED_ROOT: dict[str, Path] = {
    "flashomni": NATIVE_ROOT / "flashomni" / "3rdparty" / "cutlass",
    "draft_block_sparse": NATIVE_ROOT / "flashomni" / "3rdparty" / "cutlass",
}

_TARBALL_URL = "https://github.com/NVIDIA/cutlass/archive/{ref}.tar.gz"


def _is_cutlass_root(path: Path) -> bool:
    """A usable cutlass root has cutlass.h, the version-independent sentinel header."""
    return (path / "include" / "cutlass" / "cutlass.h").is_file()


def _build_opener() -> urllib.request.OpenerDirector:
    """urllib opener honoring an explicit GitHub proxy or the standard env proxies."""
    proxy = os.environ.get("SPARSEVIDEO_GITHUB_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        return urllib.request.build_opener(handler)
    return urllib.request.build_opener()  # respects getproxies() defaults


def _fetch(ref: str, dest: Path) -> Path:
    """Download the cutlass ``ref`` source tarball and extract its tree into ``dest``.

    ``ref`` is a tag (``v4.3.0``) or a full commit SHA. Extraction is atomic: we
    unpack into a temp dir and rename into place only after validating the headers.
    """
    if _is_cutlass_root(dest):
        return dest

    url = _TARBALL_URL.format(ref=ref)
    opener = _build_opener()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tmp_path = Path(tmp)
        tarball = tmp_path / "cutlass.tar.gz"
        print(f"[sparsevideo] fetching CUTLASS {ref} from {url}", flush=True)
        with opener.open(url, timeout=60) as resp, open(tarball, "wb") as fh:
            shutil.copyfileobj(resp, fh)

        with tarfile.open(tarball, "r:gz") as tar:
            # The archive has a single top-level dir, e.g. cutlass-4.3.0/ or cutlass-<sha>/.
            tar.extractall(tmp_path)  # noqa: S202 - trusted NVIDIA release tarball
        roots = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("cutlass")]
        if not roots or not _is_cutlass_root(roots[0]):
            raise RuntimeError(
                f"fetched CUTLASS {ref} is missing include/cutlass/cutlass.h "
                f"(corrupt/truncated download from {url})"
            )
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(roots[0]), str(dest))
    return dest


def cutlass_root(kernel: str, *, allow_fetch: bool = True) -> Path:
    """Return a CUTLASS root dir (containing ``include/``) for ``kernel``.

    Raises ``RuntimeError`` with actionable guidance if CUTLASS cannot be resolved
    and ``allow_fetch`` is False or the fetch fails.
    """
    if kernel not in CUTLASS_PINS:
        raise KeyError(f"unknown cutlass-using kernel: {kernel}. Known: {list(CUTLASS_PINS)}")

    override = os.environ.get("SPARSEVIDEO_CUTLASS_DIR")
    if override:
        root = Path(override)
        if _is_cutlass_root(root):
            return root
        raise RuntimeError(
            f"$SPARSEVIDEO_CUTLASS_DIR={override} is not a CUTLASS root "
            f"(expected include/cutlass/cutlass.h)."
        )

    vendored = _VENDORED_ROOT[kernel]
    if _is_cutlass_root(vendored):
        return vendored

    ref = CUTLASS_PINS[kernel]
    cached = _CACHE_ROOT / f"cutlass-{ref}"
    if _is_cutlass_root(cached):
        return cached

    if not allow_fetch:
        raise RuntimeError(
            f"CUTLASS {ref} for {kernel} is not available offline. Provide it via "
            f"$SPARSEVIDEO_CUTLASS_DIR or allow network fetch."
        )

    try:
        return _fetch(ref, cached)
    except Exception as exc:  # noqa: BLE001 - turn any fetch failure into guidance
        raise RuntimeError(
            f"failed to fetch CUTLASS {ref} for {kernel}: {exc}\n"
            f"Fixes: if GitHub needs a proxy on your network, set $SPARSEVIDEO_GITHUB_PROXY "
            f"(or $HTTPS_PROXY) to it; or point $SPARSEVIDEO_CUTLASS_DIR at a local "
            f"CUTLASS {ref} checkout."
        ) from exc
