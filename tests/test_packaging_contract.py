from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile


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


def test_wheel_includes_owned_native_sources_and_assets(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    wheel_path = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    required = {
        "sparsevideo/methods/svoo/sparsity_profiles/sparsity_wan_1.3B_t2v.csv",
        "sparsevideo/methods/sta/mask_strategies/mask_strategy_wan.json",
        "sparsevideo/kernels/native/svg_svoo_fused/setup.sh",
        "sparsevideo/kernels/native/svg_svoo_fused/csrc/ops.cu",
        "sparsevideo/kernels/native/spargeattn/setup.sh",
        "sparsevideo/kernels/native/flashomni/setup.sh",
        "sparsevideo/kernels/native/sta_h100/setup.sh",
        "sparsevideo/kernels/native/draft_block_sparse/setup.py",
    }
    assert required <= names
    for method in METHODS:
        assert f"sparsevideo/methods/{method}/config.yaml" in names
    assert not any(name.startswith("training_free/") for name in names)
    assert any(name.startswith("sparsevideo/kernels/native/flashomni/csrc/") for name in names)
    assert any(name.startswith("sparsevideo/kernels/native/spargeattn/csrc/") for name in names)
    assert any(name.startswith("sparsevideo/kernels/native/sta_h100/csrc/") for name in names)
    assert any(name.startswith("sparsevideo/kernels/native/draft_block_sparse/csrc/") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any(name.endswith((".so", ".pyd", ".dll", ".dylib")) for name in names)
    assert not any("/build/" in name for name in names)


def test_optional_all_extra_is_explicit_not_self_referential():
    extras = _pyproject()["project"]["optional-dependencies"]

    for name in (*SPARSE_METHODS, "all", "dev"):
        assert name in extras
    assert all(not dep.startswith("sparsevideo[") for dep in extras["all"])
    all_pkg_names = {dep.split(">")[0].split("<")[0].split("=")[0].split("!")[0] for dep in extras["all"]}
    assert all_pkg_names == {"diffsynth", "flashinfer-python", "triton"}


def test_owned_native_runtime_extras_do_not_install_rejected_runtime_packages():
    extras = _pyproject()["project"]["optional-dependencies"]

    spargeattn_pkgs = {dep.split(">")[0].split("<")[0].split("=")[0] for dep in extras["spargeattn"]}
    assert spargeattn_pkgs == {"triton"}
    assert extras["flashomni"] == []
    for deps in extras.values():
        assert not any("spas-sage-attn" in d for d in deps)
        assert not any("flashomni-python" in d for d in deps)


def test_methods_are_packages_not_flat_method_files():
    methods_root = SRC_ROOT / "methods"

    for name in METHODS:
        assert (methods_root / name / "__init__.py").exists()
        assert not (methods_root / f"{name}.py").exists()
    assert not (methods_root / "sap.py").exists()
    assert not (methods_root / "svg.py").exists()


def test_diffsynth_inference_orchestration_stays_out_of_package():
    script_only_modules = {
        "_diffsynth_infer.py",
        "infer_diffsynth.py",
    }

    assert not any(path.name in script_only_modules for path in SRC_ROOT.rglob("*.py"))
    assert (REPO_ROOT / "scripts" / "_infer_diffsynth" / "models.py").exists()
    assert (REPO_ROOT / "scripts" / "infer_diffsynth.py").exists()

    forbidden_snippets = (
        "argparse",
        "DEFAULT_MODEL_ROOT",
        "load_diffsynth_pipeline",
        "resolve_diffsynth_model_paths",
        "save_diffsynth_output",
        "ModelConfig(",
    )
    for module in SRC_ROOT.glob("*.py"):
        text = module.read_text(encoding="utf-8")
        for snippet in forbidden_snippets:
            assert snippet not in text, f"{module.relative_to(REPO_ROOT)} contains script-only DiffSynth code: {snippet}"


def test_runtime_status_uses_sparsevideo_owned_paths_not_training_free():
    from sparsevideo._runtime import optional_kernel_status

    status = optional_kernel_status()

    adacluster = status["adacluster_kernels"]
    assert adacluster["methods"] == ["adacluster"]
    assert "training_free" not in adacluster["fast_kmeans_single"]["path"]
    assert "src/sparsevideo/kernels/native/adacluster" in adacluster["fast_kmeans_single"]["path"]
    assert "training_free" not in adacluster["triton_cluster_sparse_attn"]["path"]
    assert "src/sparsevideo/kernels/native/adacluster" in adacluster["triton_cluster_sparse_attn"]["path"]
    assert "training_free" not in adacluster["triton_cluster_sparse_attn_topk"]["path"]
    assert "src/sparsevideo/kernels/native/adacluster" in adacluster["triton_cluster_sparse_attn_topk"]["path"]

    fused = status["svg_svoo_fused_kernels"]
    assert all("training_free" not in path for path in fused["candidate_dirs"])
    assert "src/sparsevideo/kernels/native" in fused["source"]["path"]
    assert "svoo" in status["svg_svoo_fused_kernels"]["methods"]
    assert "svg1" in status["svg_svoo_fused_kernels"]["methods"]

    flex = status["flex_attention"]
    assert flex["methods"] == ["svg1"]

    flash_attn = status["flash_attn"]
    assert flash_attn["methods"] == ["svg1", "draft", "adacluster"]
    assert all("training_free" not in path for path in flash_attn["package_locations"])
    assert "svg2" in status["svg_svoo_fused_kernels"]["methods"]
    assert "build_requirements" not in fused
    assert "missing_build_requirements" not in fused

    svg2 = status["svg2_kernels"]
    assert svg2["methods"] == ["svg2"]
    assert "training_free" not in svg2["triton_kmeans"]["path"]
    assert "src/sparsevideo/methods/svg2" in svg2["triton_kmeans"]["path"]
    assert svg2["triton_kmeans"]["source_files"] is True

    svoo = status["svoo_kernels"]
    assert "training_free" not in svoo["triton_layernorm"]["path"]
    assert "src/sparsevideo/kernels" in svoo["triton_layernorm"]["path"]
    assert svoo["triton_layernorm"]["source_files"] is True
    assert "training_free" not in svoo["triton_modulate"]["path"]
    assert "src/sparsevideo/kernels" in svoo["triton_modulate"]["path"]
    assert svoo["triton_modulate"]["source_files"] is True
    assert "training_free" not in svoo["wan_fast_block_patch"]["path"]
    assert "src/sparsevideo/processors" in svoo["wan_fast_block_patch"]["path"]
    assert svoo["wan_fast_block_patch"]["source_files"] is True
    assert "training_free" not in svoo["hunyuan_sparse_forward_patch"]["path"]
    assert "src/sparsevideo/processors" in svoo["hunyuan_sparse_forward_patch"]["path"]
    assert svoo["hunyuan_sparse_forward_patch"]["source_files"] is True

    sparge = status["spas_sage_attn"]
    assert sparge["methods"] == ["spargeattn", "radial"]
    assert "training_free" not in sparge["sparsevideo_owned_source"]["path"]
    assert sparge["sparsevideo_owned_source"]["source_files"] is True
    assert "src/sparsevideo/kernels/native/spargeattn" in sparge["sparsevideo_runtime"]["path"]
    assert sparge["sparsevideo_runtime"]["package"] is True
    assert sparge["sparsevideo_runtime"]["block_sparse_sage2_attn_cuda"] is True
    assert sparge["sparsevideo_runtime"]["autotune"] is True
    assert sparge["sparsevideo_runtime"]["gpu_process_pool"] is True
    assert "training_free" not in sparge["sparsevideo_runtime"]["hunyuan_forward_patch"]["path"]
    assert "src/sparsevideo/methods/spargeattn" in sparge["sparsevideo_runtime"]["hunyuan_forward_patch"]["path"]
    assert sparge["sparsevideo_runtime"]["hunyuan_forward_patch"]["source_files"] is True
    if sparge["sparsevideo_runtime"].get("ready"):
        assert sparge["selected_runtime"] == "sparsevideo"
        assert sparge["training_free_runtime"] is False

    sta = status["sta_kernels"]
    assert sta["methods"] == ["sta"]
    assert "training_free" not in sta["sparsevideo_fastvideo_triton"]["path"]
    assert "src/sparsevideo/kernels/native/sta_h100/python/fastvideo_kernel/triton_kernels" in sta["sparsevideo_fastvideo_triton"]["path"]
    assert sta["sparsevideo_fastvideo_triton"]["source_files"] is True
    assert "sparsevideo_triton" not in sta
    assert "training_free" not in sta["sparsevideo_h100"]["source"]["path"]
    assert "src/sparsevideo/kernels/native/sta_h100" in sta["sparsevideo_h100"]["source"]["path"]
    assert sta["sparsevideo_h100"]["source"]["source_files"] is True
    assert "training_free" not in sta["sparsevideo_a100_block_sparse"]["source"]["path"]
    assert "src/sparsevideo/kernels/native/draft_block_sparse" in sta["sparsevideo_a100_block_sparse"]["source"]["path"]
    assert sta["sparsevideo_a100_block_sparse"]["source"]["source_files"] is True

    assert status["flashomni"]["methods"] == ["flashomni"]
    assert "native_extension" in status["flashomni"]
    assert "training_free" not in status["flashomni"]["sparsevideo_owned_source"]["path"]
    assert "src/sparsevideo/kernels/native/flashomni" in status["flashomni"]["sparsevideo_owned_source"]["path"]
    assert status["flashomni"]["sparsevideo_owned_source"]["source_files"] is True
    assert status["flashomni"]["sparsevideo_runtime"]["package"] is True
    if status["flashomni"]["sparsevideo_runtime"].get("ready"):
        assert status["flashomni"]["selected_runtime"] == "sparsevideo"
        assert status["flashomni"]["training_free_runtime"] is False

    draft = status["draft_kernels"]
    assert draft["methods"] == ["draft"]
    assert draft["upstream_backend"] == "mit-han-lab/Block-Sparse-Attention"
    assert "training_free" not in draft["mit_block_sparse_attn"]["path"]
    assert "src/sparsevideo/kernels/native/draft_block_sparse" in draft["mit_block_sparse_attn"]["path"]
    assert draft["mit_block_sparse_attn"]["selected_runtime"] in {"missing", "sparsevideo"}


def test_optional_kernel_status_reports_invalid_owned_runtime_env_roots(monkeypatch, tmp_path):
    from sparsevideo._runtime import optional_kernel_status

    monkeypatch.setenv("SPARSEVIDEO_SPARGEATTN_ROOT", str(tmp_path / "training_free" / "SpargeAttn"))
    monkeypatch.setenv("SPARSEVIDEO_SAGEATTENTION_ROOT", str(tmp_path / "external" / "sageattention"))
    monkeypatch.setenv("SPARSEVIDEO_FLASHOMNI_ROOT", str(tmp_path / "training_free" / "FlashOmni"))

    status = optional_kernel_status()

    assert "inside training_free" in status["spas_sage_attn"]["env_root"]["error"]
    assert "outside the SparseVideo-owned runtime root" in status["sageattention"]["env_root"]["error"]
    assert "inside training_free" in status["flashomni"]["env_root"]["error"]


def test_svoo_sparsity_profiles_are_owned_package_data():
    profile_root = SRC_ROOT / "methods" / "svoo" / "sparsity_profiles"

    assert (profile_root / "sparsity_wan_1.3B_t2v.csv").exists()
    assert (profile_root / "sparsity_wan_14B_t2v.csv").exists()
    assert (profile_root / "sparsity_wan22_A14B_t2v.csv").exists()
    assert (profile_root / "sparsity_hunyuan10_13B_t2v.csv").exists()


def test_sta_mask_strategies_are_owned_package_data():
    strategy_root = SRC_ROOT / "methods" / "sta" / "mask_strategies"

    assert (strategy_root / "mask_strategy_wan.json").exists()
    assert (strategy_root / "mask_strategy_hunyuan.json").exists()
