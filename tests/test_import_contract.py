from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace


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


def test_sparsevideo_public_api_exports_apply_alias():
    import inspect

    import sparsevideo

    assert callable(sparsevideo.apply)
    assert callable(sparsevideo.replace_attention)
    assert inspect.signature(sparsevideo.apply) == inspect.signature(sparsevideo.apply_sparse_attention)
    assert inspect.signature(sparsevideo.replace_attention) == inspect.signature(sparsevideo.apply_sparse_attention)
    assert "apply" in sparsevideo.__all__
    assert "replace_attention" in sparsevideo.__all__


def test_unvalidated_method_reasons_are_shared_by_api_and_infer():
    import importlib.util

    from sparsevideo._support import unvalidated_method_reason

    script = REPO_ROOT / "scripts" / "infer_diffusers.py"
    spec = importlib.util.spec_from_file_location("sparsevideo_infer_for_support_test", script)
    infer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = infer
    spec.loader.exec_module(infer)

    spec_obj = replace(infer.MODEL_SPECS["ltx-video"], sparse_methods=("svg2",))
    message = infer.unsupported_sparse_method_message(spec_obj, "draft")

    assert unvalidated_method_reason("draft") in message
    assert "Current smoke coverage is dense/svg2 only." in message


def test_native_fused_kernel_candidates_are_sparsevideo_owned():
    from sparsevideo.kernels.fused_norm_rope import _candidate_native_kernel_dirs

    paths = [str(path) for path in _candidate_native_kernel_dirs()]
    assert any("src/sparsevideo/kernels/native/build" in path for path in paths)
    assert all("training_free" not in path for path in paths)


def test_native_fused_kernel_candidates_reject_training_free_env_root(monkeypatch, tmp_path):
    import pytest

    from sparsevideo.kernels.fused_norm_rope import _candidate_native_kernel_dirs

    monkeypatch.setenv(
        "SPARSEVIDEO_NATIVE_KERNEL_ROOT",
        str(tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"),
    )

    with pytest.raises(RuntimeError, match="Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free"):
        list(_candidate_native_kernel_dirs())


def test_native_rmsnorm_support_boundary_matches_upstream_narrow_kernel():
    from sparsevideo.kernels.fused_norm_rope import _native_rmsnorm_supported

    assert _native_rmsnorm_supported(32)
    assert _native_rmsnorm_supported(64)
    assert _native_rmsnorm_supported(128)
    assert _native_rmsnorm_supported(256)
    assert not _native_rmsnorm_supported(1536)


def test_explicit_native_rmsnorm_rejects_unsupported_dim_without_fallback(monkeypatch):
    import pytest
    import torch

    import sparsevideo.kernels.fused_norm_rope as kernels

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "native")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(kernels, "_load_native_kernels", lambda required=False: SimpleNamespace())

    x = torch.zeros(1, 1536)
    weight = torch.ones(1536)
    with pytest.raises(RuntimeError, match="does not support hidden_dim=1536"):
        kernels.triton_rmsnorm_inplace(x, weight, 1e-6)


def test_explicit_fused_kernel_backends_reject_cpu_fallback(monkeypatch):
    import pytest
    import torch

    import sparsevideo.kernels.fused_norm_rope as kernels

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "native")
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        kernels.triton_rmsnorm_inplace(torch.zeros(1, 64), torch.ones(64), 1e-6)

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    q = torch.zeros(1, 4, 1, 8)
    k = torch.zeros_like(q)
    cos = torch.ones(4, 8)
    sin = torch.zeros(4, 8)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        kernels.triton_rope_wan_inplace(q, k, cos, sin)


def test_fused_norm_rope_accepts_cpu_offload_constants(monkeypatch):
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-only CPU-offload constant device check")

    import sparsevideo.kernels.fused_norm_rope as kernels

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "auto")
    x = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.float16)
    weight = torch.ones(64)
    out = kernels.triton_rmsnorm_inplace(x.clone(), weight, 1e-6)
    assert out.is_cuda

    monkeypatch.setenv("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "triton")
    q = torch.randn(1, 4, 1, 8, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    cos = torch.ones(4, 8)
    sin = torch.zeros(4, 8)
    q_out, k_out = kernels.triton_rope_wan_inplace(q, k, cos, sin)
    assert q_out.is_cuda
    assert k_out.is_cuda


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


def test_required_native_fused_kernel_rejects_training_free_env_root(monkeypatch, tmp_path):
    import sparsevideo.kernels.fused_norm_rope as kernels

    monkeypatch.setenv(
        "SPARSEVIDEO_NATIVE_KERNEL_ROOT",
        str(tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"),
    )
    kernels._NATIVE_KERNELS_CHECKED = False
    kernels._NATIVE_KERNELS = None
    kernels._NATIVE_KERNELS_ERROR = None

    try:
        try:
            kernels._load_native_kernels(required=True)
        except ImportError as exc:
            assert "Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free" in str(exc)
        else:
            raise AssertionError("expected training_free native root to fail")
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


def test_native_fused_kernel_loader_rejects_non_sparsevideo_module(monkeypatch, tmp_path):
    import sparsevideo.kernels.fused_norm_rope as kernels

    build_dir = tmp_path / "native" / "build"
    build_dir.mkdir(parents=True)
    external_dir = tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"
    external_dir.mkdir(parents=True)
    external_module = SimpleNamespace(__file__=str(external_dir / "_kernels.so"))

    def fake_import_module(name):
        if name == "_kernels":
            return external_module
        return __import__(name)

    monkeypatch.setattr(kernels, "_candidate_native_kernel_dirs", lambda: iter((build_dir,)))
    monkeypatch.setattr(kernels.importlib, "import_module", fake_import_module)
    kernels._NATIVE_KERNELS_CHECKED = False
    kernels._NATIVE_KERNELS = None
    kernels._NATIVE_KERNELS_ERROR = None

    try:
        try:
            kernels._load_native_kernels(required=True)
        except ImportError as exc:
            assert "outside SparseVideo native dirs" in str(exc)
        else:
            raise AssertionError("expected non-SparseVideo native kernel import to fail")
    finally:
        kernels._NATIVE_KERNELS_CHECKED = False
        kernels._NATIVE_KERNELS = None
        kernels._NATIVE_KERNELS_ERROR = None


def test_native_fused_kernel_loader_rejects_missing_expected_ops(monkeypatch, tmp_path):
    import sparsevideo.kernels.fused_norm_rope as kernels

    build_dir = tmp_path / "native" / "build"
    build_dir.mkdir(parents=True)
    local_module = SimpleNamespace(__file__=str(build_dir / "_kernels.so"), rms_norm_forward=object())

    def fake_import_module(name):
        if name == "_kernels":
            return local_module
        return __import__(name)

    monkeypatch.setattr(kernels, "_candidate_native_kernel_dirs", lambda: iter((build_dir,)))
    monkeypatch.setattr(kernels.importlib, "import_module", fake_import_module)
    kernels._NATIVE_KERNELS_CHECKED = False
    kernels._NATIVE_KERNELS = None
    kernels._NATIVE_KERNELS_ERROR = None

    try:
        try:
            kernels._load_native_kernels(required=True)
        except ImportError as exc:
            assert "missing expected fused ops" in str(exc)
        else:
            raise AssertionError("expected incomplete native kernel import to fail")
    finally:
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
    "flash_attn_imported": "flash_attn" in sys.modules,
    "flashomni_imported": "flashomni" in sys.modules,
    "flashinfer_imported": "flashinfer" in sys.modules,
    "fastvideo_imported": "fastvideo_kernel" in sys.modules,
    "sageattention_imported": "sageattention" in sys.modules,
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
        "adacluster_kernels",
        "draft_kernels",
        "flash_attn",
        "flashinfer",
        "flashomni",
        "flex_attention",
        "radial_kernels",
        "sageattention",
        "spas_sage_attn",
        "sta_kernels",
        "svg1_kernels",
        "svg2_kernels",
        "svg_svoo_fused_kernels",
        "svoo_kernels",
    ]
    assert payload["flash_attn_imported"] is False
    assert payload["flashomni_imported"] is False
    assert payload["flashinfer_imported"] is False
    assert payload["fastvideo_imported"] is False
    assert payload["sageattention_imported"] is False
    assert payload["native_kernels_imported"] is False


def test_native_kernel_load_status_reports_import_failure(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    build_dir = tmp_path / "native" / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "_kernels.so").write_bytes(b"not a real extension")

    def fake_import_module(name):
        if name == "_kernels":
            raise ImportError("libc10.so: cannot open shared object file")
        return __import__(name)

    monkeypatch.setattr(_runtime, "_native_kernel_dirs", lambda: [build_dir])
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.native_kernel_load_status()

    assert status["built_extension"] is True
    assert status["native_load_checked"] is True
    assert status["native_extension"] is False
    assert status["native_import_error_type"] == "ImportError"
    assert "libc10.so" in status["native_import_error"]


def test_native_kernel_load_status_rejects_non_sparsevideo_module(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    build_dir = tmp_path / "native" / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "_kernels.so").write_bytes(b"placeholder")
    external_dir = tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"
    external_dir.mkdir(parents=True)
    external_module = SimpleNamespace(__file__=str(external_dir / "_kernels.so"), rms_norm_forward=object())

    def fake_import_module(name):
        if name == "_kernels":
            return external_module
        return __import__(name)

    monkeypatch.setattr(_runtime, "_native_kernel_dirs", lambda: [build_dir])
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.native_kernel_load_status()

    assert status["built_extension"] is True
    assert status["native_extension"] is False
    assert "outside SparseVideo native dirs" in status["native_import_error"]


def test_native_kernel_load_status_rejects_missing_expected_ops(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    build_dir = tmp_path / "native" / "build"
    build_dir.mkdir(parents=True)
    (build_dir / "_kernels.so").write_bytes(b"placeholder")
    local_module = SimpleNamespace(__file__=str(build_dir / "_kernels.so"), rms_norm_forward=object())

    def fake_import_module(name):
        if name == "_kernels":
            return local_module
        return __import__(name)

    monkeypatch.setattr(_runtime, "_native_kernel_dirs", lambda: [build_dir])
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.native_kernel_load_status()

    assert status["built_extension"] is True
    assert status["native_extension"] is False
    assert "missing expected fused ops" in status["native_import_error"]
    assert "apply_qk_rope_inplace_cossin_txtlast" in status["missing_ops"]


def test_native_kernel_load_status_rejects_training_free_env_root(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    monkeypatch.setenv(
        "SPARSEVIDEO_NATIVE_KERNEL_ROOT",
        str(tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"),
    )

    status = _runtime.native_kernel_load_status()

    assert status["native_extension"] is False
    assert status["native_import_error_type"] == "RuntimeError"
    assert "Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free" in status["native_import_error"]


def test_flash_attn_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "flash_attn":
            raise ImportError("bad flash_attn abi")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flash_attn_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad flash_attn abi" in status["import_error"]


def test_flash_attn_load_status_detects_required_functions(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "flash_attn" / "__init__.py"),
        flash_attn_func=lambda *args, **kwargs: None,
    )
    interface = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "flash_attn" / "flash_attn_interface.py"),
        flash_attn_varlen_func=lambda *args, **kwargs: None,
    )

    def fake_import_module(name):
        if name == "flash_attn":
            return package
        if name == "flash_attn.flash_attn_interface":
            return interface
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flash_attn_load_status()

    assert status["imported"] is True
    assert status["flash_attn_func"] is True
    assert status["flash_attn_varlen_func"] is True
    assert status["import_error"] is None


def test_flash_attn_load_status_rejects_training_free_package(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "flash_attn" / "__init__.py"),
        flash_attn_func=lambda *args, **kwargs: None,
    )
    interface = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "flash_attn" / "flash_attn_interface.py"),
        flash_attn_varlen_func=lambda *args, **kwargs: None,
    )

    def fake_import_module(name):
        if name == "flash_attn":
            return package
        if name == "flash_attn.flash_attn_interface":
            return interface
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flash_attn_load_status()

    assert status["imported"] is True
    assert status["training_free_package_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_flashinfer_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "flashinfer":
            raise ImportError("bad flashinfer abi")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flashinfer_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad flashinfer abi" in status["import_error"]


def test_flashinfer_load_status_detects_required_apis(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    class Wrapper:
        pass

    class MaskMode:
        pass

    class PosEncodingMode:
        pass

    package = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "flashinfer" / "__init__.py"),
        BlockSparseAttentionWrapper=Wrapper,
        single_prefill_with_kv_cache=lambda *args, **kwargs: None,
        merge_state=lambda *args, **kwargs: None,
    )
    sparse = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "flashinfer" / "sparse.py"),
        BlockSparseAttentionWrapper=Wrapper,
        VariableBlockSparseAttentionWrapper=Wrapper,
        canonicalize_torch_dtype=lambda dtype: dtype,
        MaskMode=MaskMode,
        PosEncodingMode=PosEncodingMode,
        determine_attention_backend=lambda *args, **kwargs: "fa2",
        get_batch_prefill_module=lambda *args, **kwargs: object(),
    )

    def fake_import_module(name):
        if name == "flashinfer":
            return package
        if name == "flashinfer.sparse":
            return sparse
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flashinfer_load_status()

    assert status["imported"] is True
    assert status["sparse_imported"] is True
    assert status["top_level_block_sparse_attention_wrapper"] is True
    assert status["top_level_single_prefill_with_kv_cache"] is True
    assert status["top_level_merge_state"] is True
    assert status["sparse_variable_block_sparse_attention_wrapper"] is True
    assert status["sparse_get_batch_prefill_module"] is True
    assert status["import_error"] is None


def test_flashinfer_load_status_rejects_training_free_package(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "flashinfer" / "__init__.py"),
    )
    sparse = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "flashinfer" / "sparse.py"),
    )

    def fake_import_module(name):
        if name == "flashinfer":
            return package
        if name == "flashinfer.sparse":
            return sparse
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.flashinfer_load_status()

    assert status["imported"] is True
    assert status["sparse_imported"] is True
    assert status["training_free_package_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_adacluster_load_status_detects_owned_triton_apis(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    native_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "adacluster"
    fast_kmeans = SimpleNamespace(
        __file__=str(native_root / "fast_kmeans_single.py"),
        flash_kmeans_single=lambda *args, **kwargs: None,
        _compute_norm_squal_impl=lambda *args, **kwargs: None,
        _compute_cluster_indices_impl=lambda *args, **kwargs: None,
        _compute_new_kernel_impl=lambda *args, **kwargs: None,
    )
    cluster_attn = SimpleNamespace(
        __file__=str(native_root / "triton_cluster_sparse_attn.py"),
        triton_cluster_sparse_attn=lambda *args, **kwargs: None,
        _cluster_sparse_attn=lambda *args, **kwargs: None,
    )
    cluster_attn_topk = SimpleNamespace(
        __file__=str(native_root / "triton_cluster_sparse_attn_topk.py"),
        triton_cluster_sparse_attn_topk=lambda *args, **kwargs: None,
        _cluster_sparse_attn_topk=lambda *args, **kwargs: None,
    )

    def fake_import_module(name):
        if name == "sparsevideo.kernels.native.adacluster.fast_kmeans_single":
            return fast_kmeans
        if name == "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn":
            return cluster_attn
        if name == "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn_topk":
            return cluster_attn_topk
        return __import__(name)

    monkeypatch.setattr(_runtime, "_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.adacluster_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is True
    assert status["owned_runtime"] is True
    assert status["flash_kmeans_single"] is True
    assert status["triton_cluster_sparse_attn"] is True
    assert status["triton_cluster_sparse_attn_topk"] is True
    assert status["kmeans_jit_kernels"] is True
    assert status["cluster_sparse_attn_jit_kernel"] is True
    assert status["cluster_sparse_attn_topk_jit_kernel"] is True
    assert status["import_error"] is None


def test_adacluster_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "sparsevideo.kernels.native.adacluster.fast_kmeans_single":
            raise ImportError("bad adacluster triton import")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.adacluster_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad adacluster triton import" in status["import_error"]


def test_adacluster_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    fast_kmeans = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "Adacluster" / "triton_kernel" / "fast_kmeans_single.py"),
        flash_kmeans_single=lambda *args, **kwargs: None,
    )
    cluster_attn = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "Adacluster" / "triton_kernel" / "triton_cluster_sparse_attn.py"),
        triton_cluster_sparse_attn=lambda *args, **kwargs: None,
    )
    cluster_attn_topk = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "Adacluster" / "triton_kernel" / "triton_cluster_sparse_attn_topk.py"),
        triton_cluster_sparse_attn_topk=lambda *args, **kwargs: None,
    )

    def fake_import_module(name):
        if name == "sparsevideo.kernels.native.adacluster.fast_kmeans_single":
            return fast_kmeans
        if name == "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn":
            return cluster_attn
        if name == "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn_topk":
            return cluster_attn_topk
        return __import__(name)

    monkeypatch.setattr(_runtime, "_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.adacluster_load_status()

    assert status["training_free_runtime_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_svg2_runtime_load_status_detects_owned_apis():
    from sparsevideo import _runtime

    status = _runtime.svg2_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is True
    assert status["owned_runtime"] is True
    assert status["training_free_runtime_detected"] is False
    assert status["triton_kmeans"] is True
    assert status["euclid_assign_triton"] is True
    assert status["identify_dynamic_map"] is True
    assert status["permute_tensor_by_labels_triton"] is True
    assert status["variable_block_sparse_attn"] is True
    assert status["hunyuan_flashinfer_varlen_attn"] is True
    assert status["import_error"] is None


def test_svg2_runtime_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "sparsevideo.methods.svg2.kmeans":
            raise ImportError("bad svg2 triton module")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.svg2_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad svg2 triton module" in status["import_error"]


def test_svg2_runtime_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    def fake_import_module(name):
        return SimpleNamespace(__file__=str(tmp_path / "training_free" / "Sparse-VideoGen" / f"{name}.py"))

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.svg2_runtime_load_status()

    assert status["training_free_runtime_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_radial_runtime_load_status_detects_owned_apis():
    from sparsevideo import _runtime

    status = _runtime.radial_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is True
    assert status["owned_runtime"] is True
    assert status["training_free_runtime_detected"] is False
    assert status["radial_bsr_mask"] is True
    assert status["shrink_mask_strict"] is True
    assert status["radial_flashinfer_attention"] is True
    assert status["radial_sage_attention"] is True
    assert status["sparge_mask_convert"] is True
    assert status["build_bsr_from_mask"] is True
    assert status["bsr_sparse_attn"] is True
    assert status["import_error"] is None


def test_radial_runtime_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "sparsevideo.methods.radial.method":
            raise ImportError("bad radial module")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.radial_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad radial module" in status["import_error"]


def test_radial_runtime_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    def fake_import_module(name):
        return SimpleNamespace(__file__=str(tmp_path / "training_free" / "radial-attention" / f"{name}.py"))

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.radial_runtime_load_status()

    assert status["training_free_runtime_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_svg1_runtime_load_status_detects_owned_apis():
    from sparsevideo import _runtime

    status = _runtime.svg1_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is True
    assert status["owned_runtime"] is True
    assert status["training_free_runtime_detected"] is False
    assert status["svg_attention"] is True
    assert status["svg_flex_attention"] is True
    assert status["svg1_dense_attention"] is True
    assert status["profile_masks"] is True
    assert status["build_svg_block_mask"] is True
    assert status["place_svg_heads"] is True
    assert status["restore_svg_heads"] is True
    assert status["sparse_head_placement"] is True
    assert status["hidden_states_placement"] is True
    assert status["import_error"] is None


def test_svg1_runtime_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime

    def fake_import_module(name):
        if name == "sparsevideo.methods.svg1.method":
            raise ImportError("bad svg1 method")
        return __import__(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.svg1_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad svg1 method" in status["import_error"]


def test_svg1_runtime_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    def fake_import_module(name):
        return SimpleNamespace(__file__=str(tmp_path / "training_free" / "Sparse-VideoGen" / f"{name}.py"))

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.svg1_runtime_load_status()

    assert status["training_free_runtime_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_svoo_runtime_load_status_detects_owned_apis():
    from sparsevideo import _runtime

    status = _runtime.svoo_runtime_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is True
    assert status["owned_runtime"] is True
    assert status["training_free_runtime_detected"] is False
    assert status["triton_l2norm_forward"] is True
    assert status["triton_layernorm_forward"] is True
    assert status["triton_modulate_shift_forward"] is True
    assert status["co_cluster_tokens"] is True
    assert status["identify_dynamic_map"] is True
    assert status["variable_block_sparse_attn"] is True
    assert status["compute_exact_attention_sparsity"] is True
    assert status["import_error"] is None


def test_svoo_runtime_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    def fake_import_module(name):
        return SimpleNamespace(__file__=str(tmp_path / "training_free" / "SVOO" / f"{name}.py"))

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)

    status = _runtime.svoo_runtime_load_status()

    assert status["training_free_runtime_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_spas_sage_attn_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime
    from sparsevideo.kernels import spas_sage_runtime

    def fake_load_module():
        raise ImportError("bad spas_sage_attn abi")

    monkeypatch.setattr(spas_sage_runtime, "load_spas_sage_attn_module", fake_load_module)

    status = _runtime.spas_sage_attn_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad spas_sage_attn abi" in status["import_error"]


def test_spas_sage_attn_load_status_detects_required_apis(monkeypatch, tmp_path):
    from sparsevideo import _runtime
    from sparsevideo.kernels import spas_sage_runtime

    module = SimpleNamespace(
        __file__=str(tmp_path / "spargeattn" / "spas_sage_attn" / "__init__.py"),
        spas_sage2_attn_meansim_cuda=lambda *args, **kwargs: None,
        spas_sage2_attn_meansim_topk_cuda=lambda *args, **kwargs: None,
        block_sparse_sage2_attn_cuda=lambda *args, **kwargs: None,
    )

    class SparseAttentionMeansim:
        pass

    monkeypatch.setattr(spas_sage_runtime, "load_spas_sage_attn_module", lambda: module)
    monkeypatch.setattr(
        spas_sage_runtime,
        "load_sparse_attention_meansim_class",
        lambda: SparseAttentionMeansim,
    )

    status = _runtime.spas_sage_attn_load_status(require_autotune=True)

    assert status["imported"] is True
    assert status["spas_sage2_attn_meansim_cuda"] is True
    assert status["spas_sage2_attn_meansim_topk_cuda"] is True
    assert status["block_sparse_sage2_attn_cuda"] is True
    assert status["autotune"] is True
    assert status["import_error"] is None


def test_sageattention_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime
    from sparsevideo.kernels import sageattention_runtime

    def fake_load_module():
        raise ImportError("bad sageattention abi")

    monkeypatch.setattr(sageattention_runtime, "load_sageattention_module", fake_load_module)

    status = _runtime.sageattention_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad sageattention abi" in status["import_error"]


def test_sageattention_load_status_detects_sageattn(monkeypatch, tmp_path):
    from sparsevideo import _runtime
    from sparsevideo.kernels import sageattention_runtime

    module = SimpleNamespace(
        __file__=str(tmp_path / "sageattention" / "__init__.py"),
        sageattn=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sageattention_runtime, "load_sageattention_module", lambda: module)

    status = _runtime.sageattention_load_status()

    assert status["imported"] is True
    assert status["sageattn"] is True
    assert status["import_error"] is None


def test_flashomni_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime
    from sparsevideo.methods.flashomni import method as flashomni_method

    def fake_load_module():
        raise ImportError("bad flashomni abi")

    monkeypatch.setattr(flashomni_method, "_flashomni_import", fake_load_module)

    status = _runtime.flashomni_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad flashomni abi" in status["import_error"]


def test_flashomni_load_status_detects_required_apis(monkeypatch, tmp_path):
    import torch

    from sparsevideo import _runtime
    from sparsevideo.methods.flashomni import method as flashomni_method

    class Wrapper:
        pass

    class _Op:
        default = object()

    owned_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "flashomni"
    package = SimpleNamespace(
        __file__=str(owned_root / "flashomni" / "__init__.py"),
        attention=SimpleNamespace(
            __file__=str(owned_root / "flashomni" / "attention.py"),
            BatchFlashOmniFAWithRaggedKVWrapper=Wrapper,
        ),
        segment_packbits=lambda *args, **kwargs: None,
        packbits=lambda *args, **kwargs: None,
    )
    native_module = SimpleNamespace(__file__=str(owned_root / "flashomni" / "flashomni_kernels.abi3.so"))
    jit_module = SimpleNamespace(has_prebuilt_ops=True)
    ops_namespace = SimpleNamespace(
        batch_sparseFA_with_kv_plan=_Op(),
        batch_sparseFA_with_ragged_kv_run=_Op(),
    )
    torch_ops = SimpleNamespace(flashomni_kernels=ops_namespace)

    real_import_module = _runtime.importlib.import_module

    def fake_import_module(name):
        if name == "flashomni.flashomni_kernels":
            return native_module
        if name == "flashomni.jit":
            return jit_module
        return real_import_module(name)

    monkeypatch.setattr(flashomni_method, "_flashomni_import", lambda: package)
    monkeypatch.setattr(flashomni_method, "_local_flashomni_root", lambda: owned_root)
    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(torch, "ops", torch_ops)

    status = _runtime.flashomni_load_status()

    assert status["imported"] is True
    assert status["native_extension_imported"] is True
    assert status["owned_runtime"] is True
    assert status["batch_flashomni_fa_with_ragged_kv_wrapper"] is True
    assert status["segment_packbits"] is True
    assert status["packbits"] is True
    assert status["jit_has_prebuilt_ops"] is True
    assert status["torch_ops_batch_sparseFA_with_kv_plan"] is True
    assert status["torch_ops_batch_sparseFA_with_ragged_kv_run"] is True
    assert status["import_error"] is None


def test_flashomni_load_status_rejects_training_free_package(monkeypatch, tmp_path):
    from sparsevideo import _runtime
    from sparsevideo.methods.flashomni import method as flashomni_method

    package = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "FlashOmni" / "flashomni" / "__init__.py"),
        attention=SimpleNamespace(
            __file__=str(tmp_path / "training_free" / "FlashOmni" / "flashomni" / "attention.py"),
        ),
    )

    monkeypatch.setattr(flashomni_method, "_flashomni_import", lambda: package)

    status = _runtime.flashomni_load_status()

    assert status["imported"] is True
    assert status["training_free_package_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_sta_load_status_detects_h100_and_a100_apis(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    owned_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "sta_h100"
    triton_source = owned_root / "python" / "fastvideo_kernel" / "triton_kernels" / "st_attn_triton.py"

    def triton_fn():
        return None

    triton_fn.__code__ = (lambda: None).__code__.replace(co_filename=str(triton_source))
    sta_ops = SimpleNamespace(_owned_fastvideo_sta_triton=lambda: triton_fn)
    h100_module = SimpleNamespace(
        __file__=str(owned_root / "__init__.py"),
        sta_fwd=lambda *args, **kwargs: None,
    )

    real_import_module = _runtime.importlib.import_module

    def fake_import_module(name):
        if name == "sparsevideo.methods.sta.ops":
            return sta_ops
        if name == "sparsevideo.kernels.native.sta_h100":
            return h100_module
        return real_import_module(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        _runtime,
        "draft_block_sparse_load_status",
        lambda: {
            "imported": True,
            "cuda_extension_imported": True,
            "block_sparse_attn_func": True,
            "cuda_fwd_block": True,
        },
    )

    status = _runtime.sta_load_status()

    assert status["triton_load_checked"] is False
    assert status["triton_imported"] is False
    assert status["triton_sliding_tile_attention_triton"] is False
    assert status["h100_package_imported"] is True
    assert status["h100_native_extension_imported"] is True
    assert status["h100_sta_fwd"] is True
    assert status["a100_block_sparse_ready"] is True
    assert status["triton_import_error"] is None
    assert status["h100_import_error"] is None


def test_sta_load_status_reports_missing_h100_sta_fwd(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    owned_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "sta_h100"
    owned_root.mkdir(parents=True)
    (owned_root / "fastvideo_kernel_ops.cpython-312-x86_64-linux-gnu.so").write_bytes(b"not a real extension")

    def triton_fn():
        return None

    sta_ops = SimpleNamespace(_owned_fastvideo_sta_triton=lambda: triton_fn)
    h100_module = SimpleNamespace(__file__=str(owned_root / "__init__.py"), sta_fwd=None)

    real_import_module = _runtime.importlib.import_module

    def fake_import_module(name):
        if name == "sparsevideo.methods.sta.ops":
            return sta_ops
        if name == "sparsevideo.kernels.native.sta_h100":
            return h100_module
        return real_import_module(name)

    monkeypatch.setattr(_runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        _runtime,
        "draft_block_sparse_load_status",
        lambda: {
            "imported": False,
            "cuda_extension_imported": False,
            "block_sparse_attn_func": False,
            "cuda_fwd_block": False,
            "import_error_type": "ImportError",
            "import_error": "missing block_sparse_attn_cuda",
        },
    )

    status = _runtime.sta_load_status()

    assert status["h100_package_imported"] is True
    assert status["h100_native_extension_imported"] is False
    assert status["h100_sta_fwd"] is False
    assert status["h100_import_error_type"] == "ImportError"
    assert "fastvideo_kernel_ops" in status["h100_import_error"]
    assert status["a100_block_sparse_ready"] is False
    assert "block_sparse_attn_cuda" in status["a100_import_error"]


def test_draft_block_sparse_load_status_detects_required_apis(monkeypatch, tmp_path):
    from sparsevideo import _runtime
    from sparsevideo.kernels import draft_block_sparse_runtime

    native_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "draft_block_sparse"
    package = SimpleNamespace(__file__=str(native_root / "block_sparse_attn" / "__init__.py"))
    cuda = SimpleNamespace(
        __file__=str(native_root / "block_sparse_attn_cuda.cpython-312-x86_64-linux-gnu.so"),
        fwd_block=lambda *args, **kwargs: None,
        bwd_block=lambda *args, **kwargs: None,
    )

    def block_sparse_attn_func():
        return None

    monkeypatch.setattr(draft_block_sparse_runtime, "_NATIVE_ROOT", native_root)
    monkeypatch.setattr(draft_block_sparse_runtime, "load_block_sparse_attn_func", lambda: block_sparse_attn_func)
    monkeypatch.setitem(sys.modules, "block_sparse_attn", package)
    monkeypatch.setitem(sys.modules, "block_sparse_attn_cuda", cuda)

    status = _runtime.draft_block_sparse_load_status()

    assert status["imported"] is True
    assert status["cuda_extension_imported"] is True
    assert status["owned_runtime"] is True
    assert status["block_sparse_attn_func"] is True
    assert status["cuda_fwd_block"] is True
    assert status["cuda_bwd_block"] is True
    assert status["import_error"] is None


def test_draft_block_sparse_load_status_reports_import_failure(monkeypatch):
    from sparsevideo import _runtime
    from sparsevideo.kernels import draft_block_sparse_runtime

    def fail_load():
        raise ImportError("bad block sparse abi")

    monkeypatch.setattr(draft_block_sparse_runtime, "load_block_sparse_attn_func", fail_load)

    status = _runtime.draft_block_sparse_load_status()

    assert status["load_checked"] is True
    assert status["imported"] is False
    assert status["import_error_type"] == "ImportError"
    assert "bad block sparse abi" in status["import_error"]


def test_draft_block_sparse_load_status_rejects_training_free_runtime(monkeypatch, tmp_path):
    from sparsevideo import _runtime
    from sparsevideo.kernels import draft_block_sparse_runtime

    native_root = tmp_path / "repo" / "src" / "sparsevideo" / "kernels" / "native" / "draft_block_sparse"
    package = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "draft-attention" / "block_sparse_attn" / "__init__.py")
    )
    cuda = SimpleNamespace(
        __file__=str(tmp_path / "training_free" / "draft-attention" / "block_sparse_attn_cuda.so"),
        fwd_block=lambda *args, **kwargs: None,
        bwd_block=lambda *args, **kwargs: None,
    )

    def block_sparse_attn_func():
        return None

    monkeypatch.setattr(draft_block_sparse_runtime, "_NATIVE_ROOT", native_root)
    monkeypatch.setattr(draft_block_sparse_runtime, "load_block_sparse_attn_func", lambda: block_sparse_attn_func)
    monkeypatch.setitem(sys.modules, "block_sparse_attn", package)
    monkeypatch.setitem(sys.modules, "block_sparse_attn_cuda", cuda)

    status = _runtime.draft_block_sparse_load_status()

    assert status["training_free_package_detected"] is True
    assert status["import_error_type"] == "ImportError"
    assert "training_free" in status["import_error"]


def test_optional_kernel_status_reports_training_free_native_root(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    monkeypatch.setenv(
        "SPARSEVIDEO_NATIVE_KERNEL_ROOT",
        str(tmp_path / "training_free" / "SVOO" / "svoo" / "kernels" / "build"),
    )

    status = _runtime.optional_kernel_status()
    fused = status["svg_svoo_fused_kernels"]

    assert fused["candidate_dirs"] == []
    assert "Refusing SPARSEVIDEO_NATIVE_KERNEL_ROOT inside training_free" in fused["candidate_dirs_error"]
    assert fused["native_extension"] is False


def test_sta_kernel_status_detects_external_c_extension_without_sta_op(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package_root = tmp_path / "fastvideo_kernel"
    extension_dir = package_root / "_C"
    extension_dir.mkdir(parents=True)
    (extension_dir / "fastvideo_kernel_ops.cpython-312-x86_64-linux-gnu.so").write_bytes(b"not_sta")

    def fake_locations(name):
        return [package_root] if name == "fastvideo_kernel" else []

    monkeypatch.setattr(_runtime, "_package_locations", fake_locations)

    status = _runtime.optional_kernel_status()
    external = status["sta_kernels"]["external_fastvideo_kernel"]
    assert external["native_extension"] is True
    assert external["sta_fwd_op"] is False


def test_sta_kernel_status_detects_external_sta_op_without_import(monkeypatch, tmp_path):
    from sparsevideo import _runtime

    package_root = tmp_path / "fastvideo_kernel"
    extension_dir = package_root / "_C"
    extension_dir.mkdir(parents=True)
    (extension_dir / "fastvideo_kernel_ops.cpython-312-x86_64-linux-gnu.so").write_bytes(b"...sta_fwd...")

    def fake_locations(name):
        return [package_root] if name == "fastvideo_kernel" else []

    monkeypatch.setattr(_runtime, "_package_locations", fake_locations)

    status = _runtime.optional_kernel_status()
    external = status["sta_kernels"]["external_fastvideo_kernel"]
    assert external["native_extension"] is True
    assert external["sta_fwd_op"] is True
