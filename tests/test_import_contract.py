from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def test_sparsevideo_import_does_not_import_optional_svoo_runtime():
    code = """
import json
import sys
sys.path.insert(0, {src!r})
import sparsevideo
print(json.dumps({{
    "methods": sparsevideo.list_methods(),
    "svoo_imported": "svoo" in sys.modules,
    "co_clustering_imported": "svoo.co_clustering" in sys.modules,
    "flashinfer_imported": "flashinfer" in sys.modules,
    "flashomni_imported": "flashomni" in sys.modules,
    "native_kernels_imported": "_kernels" in sys.modules,
}}))
""".format(src=str(SRC_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["methods"] == [
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
    assert payload["svoo_imported"] is False
    assert payload["co_clustering_imported"] is False
    assert payload["flashinfer_imported"] is False
    assert payload["flashomni_imported"] is False
    assert payload["native_kernels_imported"] is False


def test_native_fused_kernel_candidates_are_sparsevideo_owned():
    from sparsevideo.kernels.fused_norm_rope import _candidate_native_kernel_dirs

    paths = [str(path) for path in _candidate_native_kernel_dirs()]
    assert any("src/sparsevideo/kernels/native/build" in path for path in paths)
    assert all("training_free" not in path for path in paths)


def test_native_rmsnorm_support_boundary_matches_upstream_narrow_kernel():
    from sparsevideo.kernels.fused_norm_rope import _native_rmsnorm_supported

    assert _native_rmsnorm_supported(32)
    assert _native_rmsnorm_supported(64)
    assert _native_rmsnorm_supported(128)
    assert _native_rmsnorm_supported(256)
    assert not _native_rmsnorm_supported(1536)


def test_required_native_fused_kernel_reports_build_hint(monkeypatch, tmp_path):
    import sparsevideo.kernels.fused_norm_rope as kernels

    monkeypatch.setenv("SPARSEVIDEO_NATIVE_KERNEL_ROOT", str(tmp_path))
    monkeypatch.setattr(kernels.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)))
    kernels._NATIVE_KERNELS_CHECKED = False
    kernels._NATIVE_KERNELS = None
    kernels._NATIVE_KERNELS_ERROR = None

    try:
        try:
            kernels._load_native_kernels(required=True)
        except ImportError as exc:
            assert "src/sparsevideo/kernels/native/build" in str(exc)
        else:
            raise AssertionError("expected missing native kernel import to fail")
    finally:
        kernels._NATIVE_KERNELS_CHECKED = False
        kernels._NATIVE_KERNELS = None
        kernels._NATIVE_KERNELS_ERROR = None


def test_native_fused_kernel_search_preserves_candidate_priority(monkeypatch, tmp_path):
    import sys

    import sparsevideo.kernels.fused_norm_rope as kernels

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    inserted = []

    def fake_import_module(name):
        inserted.extend(path for path in (str(first), str(second)) if path in sys.path)
        raise ImportError(name)

    monkeypatch.setattr(kernels, "_candidate_native_kernel_dirs", lambda: iter((first, second)))
    monkeypatch.setattr(kernels.importlib, "import_module", fake_import_module)
    kernels._NATIVE_KERNELS_CHECKED = False
    kernels._NATIVE_KERNELS = None
    kernels._NATIVE_KERNELS_ERROR = None

    try:
        kernels._load_native_kernels(required=False)
        assert inserted[:2] == [str(first), str(second)]
    finally:
        for path in (str(first), str(second)):
            if path in sys.path:
                sys.path.remove(path)
        kernels._NATIVE_KERNELS_CHECKED = False
        kernels._NATIVE_KERNELS = None
        kernels._NATIVE_KERNELS_ERROR = None


def test_optional_kernel_status_does_not_import_optional_packages():
    code = """
import json
import sys
sys.path.insert(0, {src!r})
from sparsevideo._runtime import optional_kernel_status
status = optional_kernel_status()
print(json.dumps({{
    "keys": sorted(status),
    "flashomni_imported": "flashomni" in sys.modules,
    "flashinfer_imported": "flashinfer" in sys.modules,
    "native_kernels_imported": "_kernels" in sys.modules,
}}))
""".format(src=str(SRC_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["keys"] == [
        "fastvideo_kernel",
        "flashinfer",
        "flashomni",
        "spas_sage_attn",
        "svg_svoo_fused_kernels",
    ]
    assert payload["flashomni_imported"] is False
    assert payload["flashinfer_imported"] is False
    assert payload["native_kernels_imported"] is False


def test_fastvideo_native_extension_detection_accepts_c_extension_dir(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package_root = tmp_path / "fastvideo_kernel"
    extension_dir = package_root / "_C"
    extension_dir.mkdir(parents=True)
    (extension_dir / "fastvideo_kernel_ops.cpython-312-x86_64-linux-gnu.so").write_text("", encoding="utf-8")

    def fake_locations(name):
        return [package_root] if name == "fastvideo_kernel" else []

    monkeypatch.setattr(_runtime, "_package_locations", fake_locations)

    status = _runtime.optional_kernel_status()
    assert status["fastvideo_kernel"]["native_extension"] is True
