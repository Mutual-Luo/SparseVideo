from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS_ROOT = REPO_ROOT / "src" / "sparsevideo" / "methods"
PUBLIC_METHODS = [
    "adacluster",
    "dense",
    "draft",
    "flashomni",
    "radial",
    "spargeattn",
    "sta",
    "svg1",
    "svg2",
    "svoo",
]


def test_methods_are_packages_not_flat_modules():
    for method in PUBLIC_METHODS:
        method_dir = METHODS_ROOT / method
        assert method_dir.is_dir(), method
        assert (method_dir / "__init__.py").is_file(), method
        assert (method_dir / "config.py").is_file(), method
        assert (method_dir / "method.py").is_file(), method
        assert not (METHODS_ROOT / f"{method}.py").exists(), method
