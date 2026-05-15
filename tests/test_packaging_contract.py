from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "sparsevideo"

METHODS = (
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
)
SPARSE_METHODS = tuple(name for name in METHODS if name != "dense")


def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_wheel_only_packages_sparsevideo_layer():
    config = _pyproject()

    wheel = config["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/sparsevideo"]
    assert "training_free" not in wheel["packages"]


def test_optional_all_extra_is_explicit_not_self_referential():
    extras = _pyproject()["project"]["optional-dependencies"]

    for name in (*SPARSE_METHODS, "all", "dev"):
        assert name in extras
    assert all(not dep.startswith("sparsevideo[") for dep in extras["all"])
    assert set(extras["all"]) == {
        "flashinfer-python",
        "flashomni-python",
        "spas-sage-attn",
        "triton",
    }


def test_methods_are_packages_not_flat_method_files():
    methods_root = SRC_ROOT / "methods"

    for name in METHODS:
        assert (methods_root / name / "__init__.py").exists()
        assert not (methods_root / f"{name}.py").exists()
    assert not (methods_root / "sap.py").exists()
    assert not (methods_root / "svg.py").exists()


def test_runtime_status_uses_sparsevideo_owned_paths_not_training_free():
    from sparsevideo._runtime import optional_kernel_status

    status = optional_kernel_status()

    fused = status["svg_svoo_fused_kernels"]
    assert all("training_free" not in path for path in fused["candidate_dirs"])
    assert "src/sparsevideo/kernels/native" in fused["source"]["path"]
    assert "svoo" in status["svg_svoo_fused_kernels"]["methods"]
    assert "svg1" in status["svg_svoo_fused_kernels"]["methods"]
    assert "svg2" in status["svg_svoo_fused_kernels"]["methods"]
    assert "build_requirements" not in fused
    assert "missing_build_requirements" not in fused

    assert status["fastvideo_kernel"]["methods"] == ["sta"]
    assert "source" not in status["fastvideo_kernel"]

    assert status["flashomni"]["methods"] == ["flashomni"]
    assert "native_extension" in status["flashomni"]
    assert "source" not in status["flashomni"]


def test_svoo_sparsity_profiles_are_owned_package_data():
    profile_root = SRC_ROOT / "methods" / "svoo" / "sparsity_profiles"

    assert (profile_root / "sparsity_wan_1.3B_t2v.csv").exists()
    assert (profile_root / "sparsity_wan_14B_t2v.csv").exists()
    assert (profile_root / "sparsity_wan22_A14B_t2v.csv").exists()
    assert (profile_root / "sparsity_hunyuan10_13B_t2v.csv").exists()
