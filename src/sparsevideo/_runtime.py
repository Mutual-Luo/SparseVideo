from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List


_EXPECTED_FUSED_NATIVE_OPS = {
    "apply_qk_rope_inplace_cossin",
    "apply_qk_rope_inplace_cossin_complex",
    "apply_qk_rope_inplace_cossin_txtlast",
    "layer_norm_forward",
    "rms_norm_forward",
}


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except Exception:
        return None


def _package_locations(name: str) -> List[Path]:
    spec = _find_spec(name)
    if spec is None:
        return []
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        return [Path(path) for path in locations]
    if spec.origin:
        return [Path(spec.origin).parent]
    return []


def _glob_existing(paths: Iterable[Path], pattern: str) -> List[str]:
    files: List[str] = []
    for path in paths:
        if path.exists():
            files.extend(str(file) for file in sorted(path.glob(pattern)))
    return files


def _glob_any_existing(paths: Iterable[Path], patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(_glob_existing(paths, pattern))
    return sorted(set(files))


def _files_contain(files: Iterable[str], marker: bytes) -> bool:
    for file in files:
        try:
            with open(file, "rb") as handle:
                if marker in handle.read():
                    return True
        except OSError:
            continue
    return False


def _has_submodule_file(locations: Iterable[Path], name: str) -> bool:
    for path in locations:
        if (path / f"{name}.py").exists() or (path / name / "__init__.py").exists():
            return True
        if any(path.glob(f"{name}*.so")):
            return True
    return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_dir_status(path: Path, patterns: Iterable[str]) -> Dict[str, Any]:
    files: List[str] = []
    if path.exists():
        for pattern in patterns:
            files.extend(str(file) for file in sorted(path.glob(pattern)))
    return {
        "path": str(path),
        "exists": path.exists(),
        "source_files": bool(files),
        "sample_files": files[:8],
    }


def _has_training_free_path(paths: Iterable[Path]) -> bool:
    for path in paths:
        if "training_free" in path.parts:
            return True
    return False


def _reject_training_free_path(path: Path, *, env_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if "training_free" in resolved.parts:
        raise RuntimeError(
            f"Refusing {env_name} inside training_free; SparseVideo native kernels "
            "must be built under src/sparsevideo."
        )
    return resolved


def _owned_runtime_env_status(env_name: str, local_root: Path) -> Dict[str, Any]:
    raw = os.environ.get(env_name)
    status: Dict[str, Any] = {
        "env_name": env_name,
        "value": raw,
        "path": None,
        "error": None,
    }
    if not raw:
        return status

    root = Path(raw).expanduser()
    resolved = root.resolve(strict=False)
    status["path"] = str(resolved)
    if "training_free" in root.parts or "training_free" in resolved.parts:
        status["error"] = (
            f"Refusing {env_name} inside training_free; SparseVideo runtime kernels "
            f"must live under {local_root}."
        )
        return status
    try:
        resolved.relative_to(local_root.resolve())
    except ValueError:
        status["error"] = (
            f"Refusing {env_name} outside the SparseVideo-owned runtime root "
            f"{local_root}: {resolved}"
        )
    return status


def _native_kernel_dirs() -> List[Path]:
    dirs: List[Path] = []

    env_root = os.environ.get("SPARSEVIDEO_NATIVE_KERNEL_ROOT")
    if env_root:
        dirs.append(_reject_training_free_path(Path(env_root), env_name="SPARSEVIDEO_NATIVE_KERNEL_ROOT"))

    repo_root = _repo_root()
    dirs.append(repo_root / "src" / "sparsevideo" / "kernels" / "native" / "build")
    return dirs


def native_kernel_load_status() -> Dict[str, Any]:
    """Check whether the SparseVideo-owned `_kernels` extension really loads.

    `optional_kernel_status()` deliberately does not import optional/native
    packages. Inference preflight calls this stronger check so native kernel
    status reports loadability, not just file presence.
    """
    try:
        candidate_dirs = _native_kernel_dirs()
    except RuntimeError as exc:
        return {
            "built_extension": False,
            "native_extension": False,
            "native_load_checked": True,
            "native_import_error": str(exc),
            "native_import_error_type": type(exc).__name__,
            "module_file": None,
        }
    native_kernel_files = _glob_existing(candidate_dirs, "_kernels*.so")
    status: Dict[str, Any] = {
        "built_extension": bool(native_kernel_files),
        "native_extension": False,
        "native_load_checked": True,
        "native_import_error": None,
        "native_import_error_type": None,
        "module_file": None,
    }
    if not native_kernel_files:
        return status

    for path in reversed(candidate_dirs):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import torch  # noqa: F401

        module = importlib.import_module("_kernels")
    except Exception as exc:
        status["native_import_error_type"] = type(exc).__name__
        status["native_import_error"] = str(exc)
        return status

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        status["native_import_error_type"] = "ImportError"
        status["native_import_error"] = "Imported _kernels module has no __file__; cannot verify package ownership"
        return status
    module_path = Path(module_file).resolve()
    if not any(module_path.is_relative_to(path.resolve()) for path in candidate_dirs if path.exists()):
        status["native_import_error_type"] = "ImportError"
        status["native_import_error"] = (
            "Imported _kernels from outside SparseVideo native dirs: "
            f"{module_path}. Candidate dirs: {[str(path) for path in candidate_dirs]}"
        )
        return status

    ops = sorted(name for name in dir(module) if not name.startswith("_"))
    missing_ops = sorted(_EXPECTED_FUSED_NATIVE_OPS - set(ops))
    if missing_ops:
        status["native_import_error_type"] = "ImportError"
        status["native_import_error"] = (
            "SparseVideo `_kernels` is missing expected fused ops: "
            f"{missing_ops}"
        )
        status["module_file"] = str(module_path)
        status["ops"] = ops
        status["missing_ops"] = missing_ops
        return status

    status["native_extension"] = True
    status["module_file"] = str(module_path)
    status["ops"] = ops
    status["missing_ops"] = []
    return status


def flash_attn_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "flash_attn_func": False,
        "flash_attn_varlen_func": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "interface_module_file": None,
        "training_free_package_detected": False,
    }
    try:
        package = importlib.import_module("flash_attn")
        interface = importlib.import_module("flash_attn.flash_attn_interface")
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["imported"] = True
    module_file = getattr(package, "__file__", None)
    interface_module_file = getattr(interface, "__file__", None)
    status["module_file"] = module_file
    status["interface_module_file"] = interface_module_file
    module_paths = [Path(path).resolve() for path in (module_file, interface_module_file) if path]
    if _has_training_free_path(module_paths):
        status["training_free_package_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flash_attn resolved from training_free/, which is reference-only. "
            "Use the environment FlashAttention package for SparseVideo runtime parity."
        )
        return status

    status["flash_attn_func"] = callable(getattr(package, "flash_attn_func", None)) or callable(
        getattr(interface, "flash_attn_func", None)
    )
    status["flash_attn_varlen_func"] = callable(
        getattr(package, "flash_attn_varlen_func", None)
    ) or callable(getattr(interface, "flash_attn_varlen_func", None))
    return status


def flashinfer_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "sparse_imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "sparse_module_file": None,
        "training_free_package_detected": False,
        "top_level_block_sparse_attention_wrapper": False,
        "top_level_single_prefill_with_kv_cache": False,
        "top_level_merge_state": False,
        "sparse_block_sparse_attention_wrapper": False,
        "sparse_variable_block_sparse_attention_wrapper": False,
        "sparse_canonicalize_torch_dtype": False,
        "sparse_mask_mode": False,
        "sparse_pos_encoding_mode": False,
        "sparse_determine_attention_backend": False,
        "sparse_get_batch_prefill_module": False,
    }
    try:
        package = importlib.import_module("flashinfer")
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["imported"] = True
    module_file = getattr(package, "__file__", None)
    status["module_file"] = module_file
    try:
        sparse = importlib.import_module("flashinfer.sparse")
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["sparse_imported"] = True
    sparse_module_file = getattr(sparse, "__file__", None)
    status["sparse_module_file"] = sparse_module_file
    module_paths = [Path(path).resolve() for path in (module_file, sparse_module_file) if path]
    if _has_training_free_path(module_paths):
        status["training_free_package_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flashinfer resolved from training_free/, which is reference-only. "
            "Use the environment FlashInfer package for SparseVideo runtime parity."
        )
        return status

    status["top_level_block_sparse_attention_wrapper"] = callable(
        getattr(package, "BlockSparseAttentionWrapper", None)
    )
    status["top_level_single_prefill_with_kv_cache"] = callable(
        getattr(package, "single_prefill_with_kv_cache", None)
    )
    status["top_level_merge_state"] = callable(getattr(package, "merge_state", None))
    status["sparse_block_sparse_attention_wrapper"] = callable(
        getattr(sparse, "BlockSparseAttentionWrapper", None)
    )
    status["sparse_variable_block_sparse_attention_wrapper"] = callable(
        getattr(sparse, "VariableBlockSparseAttentionWrapper", None)
    )
    status["sparse_canonicalize_torch_dtype"] = callable(getattr(sparse, "canonicalize_torch_dtype", None))
    status["sparse_mask_mode"] = hasattr(sparse, "MaskMode")
    status["sparse_pos_encoding_mode"] = hasattr(sparse, "PosEncodingMode")
    status["sparse_determine_attention_backend"] = callable(getattr(sparse, "determine_attention_backend", None))
    status["sparse_get_batch_prefill_module"] = callable(getattr(sparse, "get_batch_prefill_module", None))
    return status


def adacluster_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "import_error_type": None,
        "import_error": None,
        "fast_kmeans_module_file": None,
        "cluster_sparse_attn_module_file": None,
        "cluster_sparse_attn_topk_module_file": None,
        "training_free_runtime_detected": False,
        "owned_runtime": False,
        "flash_kmeans_single": False,
        "triton_cluster_sparse_attn": False,
        "triton_cluster_sparse_attn_topk": False,
        "kmeans_jit_kernels": False,
        "cluster_sparse_attn_jit_kernel": False,
        "cluster_sparse_attn_topk_jit_kernel": False,
    }
    expected_root = (
        _repo_root() / "src" / "sparsevideo" / "kernels" / "native" / "adacluster"
    ).resolve()
    try:
        fast_kmeans = importlib.import_module(
            "sparsevideo.kernels.native.adacluster.fast_kmeans_single"
        )
        cluster_attn = importlib.import_module(
            "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn"
        )
        cluster_attn_topk = importlib.import_module(
            "sparsevideo.kernels.native.adacluster.triton_cluster_sparse_attn_topk"
        )
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    fast_file = getattr(fast_kmeans, "__file__", None)
    cluster_file = getattr(cluster_attn, "__file__", None)
    cluster_topk_file = getattr(cluster_attn_topk, "__file__", None)
    status["fast_kmeans_module_file"] = fast_file
    status["cluster_sparse_attn_module_file"] = cluster_file
    status["cluster_sparse_attn_topk_module_file"] = cluster_topk_file
    module_paths = [Path(path).resolve() for path in (fast_file, cluster_file, cluster_topk_file) if path]
    if _has_training_free_path(module_paths):
        status["training_free_runtime_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "AdaCluster Triton kernels resolved from training_free/, which is reference-only."
        )
        return status
    status["owned_runtime"] = bool(module_paths) and all(
        path.is_relative_to(expected_root)
        for path in module_paths
    )
    if module_paths and not status["owned_runtime"]:
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "AdaCluster Triton kernels resolved from outside the SparseVideo-owned runtime root: "
            f"{[str(path) for path in module_paths]}. Expected root: {expected_root}"
        )
        return status

    status["imported"] = True
    status["flash_kmeans_single"] = callable(getattr(fast_kmeans, "flash_kmeans_single", None))
    status["triton_cluster_sparse_attn"] = callable(
        getattr(cluster_attn, "triton_cluster_sparse_attn", None)
    )
    status["triton_cluster_sparse_attn_topk"] = callable(
        getattr(cluster_attn_topk, "triton_cluster_sparse_attn_topk", None)
    )
    status["kmeans_jit_kernels"] = all(
        callable(getattr(fast_kmeans, name, None))
        for name in (
            "_compute_norm_squal_impl",
            "_compute_cluster_indices_impl",
            "_compute_new_kernel_impl",
        )
    )
    status["cluster_sparse_attn_jit_kernel"] = callable(
        getattr(cluster_attn, "_cluster_sparse_attn", None)
    )
    status["cluster_sparse_attn_topk_jit_kernel"] = callable(
        getattr(cluster_attn_topk, "_cluster_sparse_attn_topk", None)
    )
    return status


def _owned_sparsevideo_module_load_status(
    module_names: Iterable[str],
    *,
    api_checks: Dict[str, tuple[str, str]],
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_files": {},
        "training_free_runtime_detected": False,
        "owned_runtime": False,
    }
    expected_root = (_repo_root() / "src" / "sparsevideo").resolve()
    modules: Dict[str, Any] = {}
    try:
        for name in module_names:
            modules[name] = importlib.import_module(name)
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    module_paths: List[Path] = []
    for name, module in modules.items():
        module_file = getattr(module, "__file__", None)
        status["module_files"][name] = module_file
        if module_file:
            module_paths.append(Path(module_file).resolve())

    if _has_training_free_path(module_paths):
        status["training_free_runtime_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "SparseVideo-owned sparse runtime modules resolved from training_free/, "
            "which is reference-only."
        )
        return status
    status["owned_runtime"] = bool(module_paths) and all(
        path.is_relative_to(expected_root)
        for path in module_paths
    )
    if module_paths and not status["owned_runtime"]:
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "Sparse runtime modules resolved from outside the SparseVideo-owned source root: "
            f"{[str(path) for path in module_paths]}. Expected root: {expected_root}"
        )
        return status

    status["imported"] = True
    for key, (module_name, attr_name) in api_checks.items():
        attr = getattr(modules[module_name], attr_name, None)
        status[key] = callable(attr) or hasattr(attr, "__getitem__")
    return status


def svg1_runtime_load_status() -> Dict[str, Any]:
    return _owned_sparsevideo_module_load_status(
        (
            "sparsevideo.methods.svg1.method",
            "sparsevideo.methods.svg1.placement",
        ),
        api_checks={
            "svg_attention": ("sparsevideo.methods.svg1.method", "_svg_attention"),
            "svg_flex_attention": ("sparsevideo.methods.svg1.method", "_svg_flex_attention"),
            "svg1_dense_attention": ("sparsevideo.methods.svg1.method", "_svg1_dense_attention"),
            "svg1_hunyuan_flash_attn_varlen": (
                "sparsevideo.methods.svg1.method",
                "_svg1_hunyuan_flash_attn_varlen",
            ),
            "profile_masks": ("sparsevideo.methods.svg1.method", "_profile_masks"),
            "svg_profile_mask_rows": ("sparsevideo.methods.svg1.method", "_svg_profile_mask_rows"),
            "build_svg_block_mask": ("sparsevideo.methods.svg1.method", "_build_svg_block_mask"),
            "svg_kv_blocks": ("sparsevideo.methods.svg1.method", "_svg_kv_blocks"),
            "svg_kv_block_partitions": ("sparsevideo.methods.svg1.method", "_svg_kv_block_partitions"),
            "svg_common_mask": ("sparsevideo.methods.svg1.method", "_svg_common_mask"),
            "place_svg_heads": ("sparsevideo.methods.svg1.method", "_place_svg_heads"),
            "restore_svg_heads": ("sparsevideo.methods.svg1.method", "_restore_svg_heads"),
            "round_svg_window_width": ("sparsevideo.methods.svg1.method", "_round_svg_window_width"),
            "svg_window_width": ("sparsevideo.methods.svg1.method", "_svg_window_width"),
            "sparsity_to_width": ("sparsevideo.methods.svg1.method", "_sparsity_to_width"),
            "resolve_prompt_length": ("sparsevideo.methods.svg1.method", "_resolve_prompt_length"),
            "sparse_head_placement": ("sparsevideo.methods.svg1.placement", "sparse_head_placement"),
            "hidden_states_placement": ("sparsevideo.methods.svg1.placement", "hidden_states_placement"),
            "sparse_head_placement_kernel": (
                "sparsevideo.methods.svg1.placement",
                "_sparse_head_placement_kernel",
            ),
            "hidden_states_placement_kernel": (
                "sparsevideo.methods.svg1.placement",
                "_hidden_states_placement_kernel",
            ),
        },
    )


def svg2_runtime_load_status() -> Dict[str, Any]:
    return _owned_sparsevideo_module_load_status(
        (
            "sparsevideo.methods.svg2.kmeans",
            "sparsevideo.kernels.dynamic_map",
            "sparsevideo.kernels.permute",
            "sparsevideo.kernels.flashinfer_block_sparse",
        ),
        api_checks={
            "triton_kmeans": ("sparsevideo.methods.svg2.kmeans", "triton_kmeans"),
            "euclid_assign_triton": ("sparsevideo.methods.svg2.kmeans", "euclid_assign_triton"),
            "centroid_update_triton": (
                "sparsevideo.methods.svg2.kmeans",
                "triton_centroid_update_sorted_euclid",
            ),
            "euclid_assign_kernel": ("sparsevideo.methods.svg2.kmeans", "_euclid_assign_kernel"),
            "centroid_update_kernel": ("sparsevideo.methods.svg2.kmeans", "_centroid_update_chunk_kernel"),
            "identify_dynamic_map": ("sparsevideo.kernels.dynamic_map", "identify_dynamic_map"),
            "identify_dynamic_map_global": ("sparsevideo.kernels.dynamic_map", "identify_dynamic_map_global"),
            "permute_tensor_by_labels_triton": ("sparsevideo.kernels.permute", "permute_tensor_by_labels_triton"),
            "apply_inverse_permutation_triton": ("sparsevideo.kernels.permute", "apply_inverse_permutation_triton"),
            "permute_kernel": ("sparsevideo.kernels.permute", "_permute_kernel"),
            "inverse_permute_kernel": ("sparsevideo.kernels.permute", "_inverse_permute_kernel"),
            "variable_block_sparse_attn": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "variable_block_sparse_attn",
            ),
            "hunyuan_flashinfer_varlen_attn": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "hunyuan_flashinfer_varlen_attn",
            ),
            "bsr_sparse_attn": ("sparsevideo.kernels.flashinfer_block_sparse", "bsr_sparse_attn"),
            "fill_variable_block_kv_indices_kernel": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "_fill_variable_block_kv_indices_kernel",
            ),
        },
    )


def radial_runtime_load_status() -> Dict[str, Any]:
    return _owned_sparsevideo_module_load_status(
        (
            "sparsevideo.methods.radial.method",
            "sparsevideo.kernels.flashinfer_block_sparse",
        ),
        api_checks={
            "radial_bsr_mask": ("sparsevideo.methods.radial.method", "_radial_bsr_mask"),
            "shrink_mask_strict": ("sparsevideo.methods.radial.method", "_shrink_mask_strict"),
            "radial_flashinfer_attention": (
                "sparsevideo.methods.radial.method",
                "_radial_flashinfer_attention",
            ),
            "radial_sage_attention": ("sparsevideo.methods.radial.method", "_radial_sage_attention"),
            "radial_sage_dense_attention": (
                "sparsevideo.methods.radial.method",
                "_radial_sage_dense_attention",
            ),
            "sparge_mask_convert": ("sparsevideo.methods.radial.method", "_sparge_mask_convert"),
            "sparge_sage_qk_block_sizes": (
                "sparsevideo.methods.radial.method",
                "_sparge_sage_qk_block_sizes",
            ),
            "radial_append_tail_blocks": (
                "sparsevideo.methods.radial.method",
                "_radial_append_tail_blocks",
            ),
            "expand_attention_mask": ("sparsevideo.methods.radial.method", "_expand_attention_mask"),
            "radial_window_width": ("sparsevideo.methods.radial.method", "_radial_window_width"),
            "build_bsr_from_mask": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "build_bsr_from_mask",
            ),
            "variable_block_sparse_attn": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "variable_block_sparse_attn",
            ),
            "bsr_sparse_attn": ("sparsevideo.kernels.flashinfer_block_sparse", "bsr_sparse_attn"),
            "ensure_cuda_home_for_flashinfer_jit": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "_ensure_cuda_home_for_flashinfer_jit",
            ),
        },
    )


def svoo_runtime_load_status() -> Dict[str, Any]:
    return _owned_sparsevideo_module_load_status(
        (
            "sparsevideo.kernels.l2norm",
            "sparsevideo.kernels.layernorm",
            "sparsevideo.kernels.modulate",
            "sparsevideo.kernels.co_cluster",
            "sparsevideo.kernels.dynamic_map",
            "sparsevideo.kernels.permute",
            "sparsevideo.kernels.flashinfer_block_sparse",
            "sparsevideo.kernels.sparsity",
            "sparsevideo.methods.svoo.sparsity",
        ),
        api_checks={
            "triton_l2norm_forward": ("sparsevideo.kernels.l2norm", "triton_l2norm_forward"),
            "l2norm_kernel": ("sparsevideo.kernels.l2norm", "_l2_norm_fwd_fused"),
            "triton_layernorm_forward": ("sparsevideo.kernels.layernorm", "triton_layernorm_forward"),
            "layernorm_param_kernel": ("sparsevideo.kernels.layernorm", "_layer_norm_param_fwd_fused"),
            "layernorm_noparam_kernel": ("sparsevideo.kernels.layernorm", "_layer_norm_noparam_fwd_fused"),
            "triton_modulate_shift_forward": (
                "sparsevideo.kernels.modulate",
                "triton_modulate_shift_forward",
            ),
            "triton_modulate_gate_residual_forward": (
                "sparsevideo.kernels.modulate",
                "triton_modulate_gate_residual_forward",
            ),
            "triton_modulate_shift_batched_forward": (
                "sparsevideo.kernels.modulate",
                "triton_modulate_shift_batched_forward",
            ),
            "triton_modulate_gate_residual_batched_forward": (
                "sparsevideo.kernels.modulate",
                "triton_modulate_gate_residual_batched_forward",
            ),
            "co_cluster_tokens": ("sparsevideo.kernels.co_cluster", "co_cluster_tokens"),
            "co_cluster_assign": ("sparsevideo.kernels.co_cluster", "co_cluster_assign"),
            "centroid_update_sorted_euclid": (
                "sparsevideo.kernels.co_cluster",
                "centroid_update_sorted_euclid",
            ),
            "profile_norm": ("sparsevideo.kernels.co_cluster", "profile_norm"),
            "co_cluster_assign_kernel": ("sparsevideo.kernels.co_cluster", "_fused_cocluster_assign_kernel"),
            "identify_dynamic_map": ("sparsevideo.kernels.dynamic_map", "identify_dynamic_map"),
            "identify_dynamic_map_global": ("sparsevideo.kernels.dynamic_map", "identify_dynamic_map_global"),
            "permute_tensor_by_labels_triton": ("sparsevideo.kernels.permute", "permute_tensor_by_labels_triton"),
            "apply_inverse_permutation_triton": ("sparsevideo.kernels.permute", "apply_inverse_permutation_triton"),
            "variable_block_sparse_attn": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "variable_block_sparse_attn",
            ),
            "hunyuan_flashinfer_varlen_attn": (
                "sparsevideo.kernels.flashinfer_block_sparse",
                "hunyuan_flashinfer_varlen_attn",
            ),
            "counts_from_sorted_probabilities_triton": (
                "sparsevideo.kernels.sparsity",
                "counts_from_sorted_probabilities_triton",
            ),
            "compute_exact_attention_sparsity": (
                "sparsevideo.methods.svoo.sparsity",
                "compute_exact_attention_sparsity",
            ),
            "log_attention_sparsity": ("sparsevideo.methods.svoo.sparsity", "log_attention_sparsity"),
        },
    )


def spas_sage_attn_load_status(*, require_autotune: bool = False) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "spas_sage2_attn_meansim_cuda": False,
        "spas_sage2_attn_meansim_topk_cuda": False,
        "block_sparse_sage2_attn_cuda": False,
        "autotune_checked": require_autotune,
        "autotune": False,
    }
    try:
        from sparsevideo.kernels import spas_sage_runtime

        module = spas_sage_runtime.load_spas_sage_attn_module()
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["imported"] = True
    status["module_file"] = getattr(module, "__file__", None)
    status["spas_sage2_attn_meansim_cuda"] = callable(getattr(module, "spas_sage2_attn_meansim_cuda", None))
    status["spas_sage2_attn_meansim_topk_cuda"] = callable(
        getattr(module, "spas_sage2_attn_meansim_topk_cuda", None)
    )
    status["block_sparse_sage2_attn_cuda"] = callable(getattr(module, "block_sparse_sage2_attn_cuda", None))
    if require_autotune:
        try:
            autotune_class = spas_sage_runtime.load_sparse_attention_meansim_class()
        except Exception as exc:
            status["import_error_type"] = type(exc).__name__
            status["import_error"] = str(exc)
        else:
            status["autotune"] = callable(autotune_class)
    return status


def sageattention_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "sageattn": False,
    }
    try:
        from sparsevideo.kernels import sageattention_runtime

        module = sageattention_runtime.load_sageattention_module()
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["imported"] = True
    status["module_file"] = getattr(module, "__file__", None)
    status["sageattn"] = callable(getattr(module, "sageattn", None))
    return status


def flashomni_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "native_extension_imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "attention_module_file": None,
        "native_extension_module_file": None,
        "training_free_package_detected": False,
        "owned_runtime": False,
        "attention_module": False,
        "batch_flashomni_fa_with_ragged_kv_wrapper": False,
        "segment_packbits": False,
        "packbits": False,
        "jit_has_prebuilt_ops": False,
        "torch_ops_flashomni_kernels": False,
        "torch_ops_batch_sparseFA_with_kv_plan": False,
        "torch_ops_batch_sparseFA_with_ragged_kv_run": False,
    }
    try:
        from sparsevideo.methods.flashomni import method as flashomni_method

        package = flashomni_method._flashomni_import()
        owned_root = flashomni_method._local_flashomni_root().resolve()
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["imported"] = True
    module_file = getattr(package, "__file__", None)
    attention = getattr(package, "attention", None)
    attention_module_file = getattr(attention, "__file__", None)
    status["module_file"] = module_file
    status["attention_module_file"] = attention_module_file
    module_paths = [Path(path).resolve() for path in (module_file, attention_module_file) if path]
    if _has_training_free_path(module_paths):
        status["training_free_package_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flashomni resolved from training_free/, which is reference-only. "
            "Use the SparseVideo-owned FlashOmni runtime under src/sparsevideo/kernels/native/flashomni."
        )
        return status
    status["owned_runtime"] = any(
        path.is_relative_to(owned_root)
        for path in module_paths
    )
    if module_paths and not status["owned_runtime"]:
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flashomni resolved from outside the SparseVideo-owned runtime root: "
            f"{[str(path) for path in module_paths]}. Expected root: {owned_root}"
        )
        return status

    try:
        native_module = importlib.import_module("flashomni.flashomni_kernels")
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status
    status["native_extension_imported"] = True
    status["native_extension_module_file"] = getattr(native_module, "__file__", None)

    native_module_file = status["native_extension_module_file"]
    native_paths = [Path(native_module_file).resolve()] if native_module_file else []
    if _has_training_free_path(native_paths):
        status["training_free_package_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flashomni native extension resolved from training_free/, which is reference-only. "
            "Build/use the SparseVideo-owned FlashOmni extension."
        )
        return status
    if native_paths and not any(
        path.is_relative_to(owned_root)
        for path in native_paths
    ):
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "flashomni native extension resolved from outside the SparseVideo-owned runtime root: "
            f"{[str(path) for path in native_paths]}. Expected root: {owned_root}"
        )
        return status

    status["attention_module"] = attention is not None
    status["batch_flashomni_fa_with_ragged_kv_wrapper"] = callable(
        getattr(attention, "BatchFlashOmniFAWithRaggedKVWrapper", None)
    ) or callable(getattr(package, "BatchFlashOmniFAWithRaggedKVWrapper", None))
    status["segment_packbits"] = callable(getattr(package, "segment_packbits", None))
    status["packbits"] = callable(getattr(package, "packbits", None))
    try:
        jit = importlib.import_module("flashomni.jit")
    except Exception:
        jit = None
    status["jit_has_prebuilt_ops"] = bool(getattr(jit, "has_prebuilt_ops", False))

    try:
        import torch

        ops_namespace = getattr(torch.ops, "flashomni_kernels", None)
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status
    if ops_namespace is not None:
        status["torch_ops_batch_sparseFA_with_kv_plan"] = hasattr(
            ops_namespace, "batch_sparseFA_with_kv_plan",
        )
        status["torch_ops_batch_sparseFA_with_ragged_kv_run"] = hasattr(
            ops_namespace, "batch_sparseFA_with_ragged_kv_run",
        )
    status["torch_ops_flashomni_kernels"] = bool(
        status["torch_ops_batch_sparseFA_with_kv_plan"]
        or status["torch_ops_batch_sparseFA_with_ragged_kv_run"]
    )
    return status


def sta_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "triton_load_checked": False,
        "triton_imported": False,
        "triton_sliding_tile_attention_triton": False,
        "triton_module_file": None,
        "triton_import_error_type": None,
        "triton_import_error": None,
        "h100_native_load_checked": True,
        "h100_package_imported": False,
        "h100_native_extension_imported": False,
        "h100_sta_fwd": False,
        "h100_module_file": None,
        "h100_candidate_files": [],
        "h100_import_error_type": None,
        "h100_import_error": None,
        "a100_block_sparse_load_checked": True,
        "a100_block_sparse_ready": False,
        "a100_block_sparse": {},
        "a100_import_error_type": None,
        "a100_import_error": None,
        "training_free_runtime_detected": False,
    }

    try:
        h100_module = importlib.import_module("sparsevideo.kernels.native.sta_h100")
    except Exception as exc:
        status["h100_import_error_type"] = type(exc).__name__
        status["h100_import_error"] = str(exc)
    else:
        status["h100_package_imported"] = True
        h100_module_file = getattr(h100_module, "__file__", None)
        status["h100_module_file"] = h100_module_file
        if h100_module_file is not None:
            h100_root = Path(h100_module_file).resolve().parent
            if "training_free" in h100_root.parts:
                status["training_free_runtime_detected"] = True
                status["h100_import_error_type"] = "ImportError"
                status["h100_import_error"] = (
                    "STA H100 runtime resolved from training_free/, which is reference-only."
                )
                candidate_files = []
            else:
                candidate_files = _glob_any_existing(
                    [h100_root],
                    ("fastvideo_kernel_ops*.so", "_C/fastvideo_kernel_ops*.so", "build/**/fastvideo_kernel_ops*.so"),
                )
                status["h100_candidate_files"] = candidate_files
        else:
            candidate_files = []

        sta_fwd = getattr(h100_module, "sta_fwd", None)
        status["h100_sta_fwd"] = callable(sta_fwd)
        status["h100_native_extension_imported"] = callable(sta_fwd)
        if status["h100_native_extension_imported"]:
            status["h100_import_error_type"] = None
            status["h100_import_error"] = None

        if not candidate_files and not status["h100_native_extension_imported"]:
            status["h100_import_error_type"] = "ImportError"
            status["h100_import_error"] = "No SparseVideo-owned sta_h100 native extension file was found."
        elif candidate_files:
            import_errors = []
            for candidate in candidate_files:
                spec = importlib.util.spec_from_file_location("fastvideo_kernel_ops", candidate)
                if spec is None or spec.loader is None:
                    import_errors.append(f"{candidate}: missing import spec")
                    continue
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as exc:
                    import_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                    continue
                sta_fwd = getattr(module, "sta_fwd", None)
                if callable(sta_fwd):
                    status["h100_native_extension_imported"] = True
                    status["h100_sta_fwd"] = True
                    status["h100_import_error_type"] = None
                    status["h100_import_error"] = None
                    break
                import_errors.append(f"{candidate}: missing sta_fwd")

            if not status["h100_native_extension_imported"]:
                status["h100_import_error_type"] = "ImportError"
                status["h100_import_error"] = "; ".join(import_errors) or "sta_fwd is not available."

    try:
        a100_status = draft_block_sparse_load_status()
    except Exception as exc:
        status["a100_import_error_type"] = type(exc).__name__
        status["a100_import_error"] = str(exc)
        return status

    status["a100_block_sparse"] = a100_status
    status["a100_block_sparse_ready"] = bool(
        a100_status.get("imported")
        and a100_status.get("cuda_extension_imported")
        and a100_status.get("block_sparse_attn_func")
        and a100_status.get("cuda_fwd_block")
    )
    if not status["a100_block_sparse_ready"]:
        status["a100_import_error_type"] = a100_status.get("import_error_type") or "ImportError"
        status["a100_import_error"] = (
            a100_status.get("import_error")
            or "SparseVideo-owned block-sparse CUDA backend is not ready for STA A100."
        )
    return status


def draft_block_sparse_load_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "load_checked": True,
        "imported": False,
        "cuda_extension_imported": False,
        "import_error_type": None,
        "import_error": None,
        "module_file": None,
        "cuda_module_file": None,
        "training_free_package_detected": False,
        "owned_runtime": False,
        "block_sparse_attn_func": False,
        "cuda_fwd_block": False,
        "cuda_bwd_block": False,
    }
    try:
        from sparsevideo.kernels import draft_block_sparse_runtime

        func = draft_block_sparse_runtime.load_block_sparse_attn_func()
        native_root = draft_block_sparse_runtime._NATIVE_ROOT.resolve()
    except Exception as exc:
        status["import_error_type"] = type(exc).__name__
        status["import_error"] = str(exc)
        return status

    status["block_sparse_attn_func"] = callable(func)
    module = sys.modules.get("block_sparse_attn")
    cuda_module = sys.modules.get("block_sparse_attn_cuda")
    module_file = getattr(module, "__file__", None)
    cuda_module_file = getattr(cuda_module, "__file__", None)
    status["module_file"] = module_file
    status["cuda_module_file"] = cuda_module_file
    module_paths = [Path(path).resolve() for path in (module_file, cuda_module_file) if path]
    if _has_training_free_path(module_paths):
        status["training_free_package_detected"] = True
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "Draft block_sparse_attn resolved from training_free/, which is reference-only."
        )
        return status
    status["owned_runtime"] = bool(module_paths) and all(
        path.is_relative_to(native_root)
        for path in module_paths
    )
    if module_paths and not status["owned_runtime"]:
        status["import_error_type"] = "ImportError"
        status["import_error"] = (
            "Draft block_sparse_attn resolved from outside the SparseVideo-owned runtime root: "
            f"{[str(path) for path in module_paths]}. Expected root: {native_root}"
        )
        return status

    status["imported"] = module is not None
    status["cuda_extension_imported"] = cuda_module is not None
    status["cuda_fwd_block"] = callable(getattr(cuda_module, "fwd_block", None))
    status["cuda_bwd_block"] = callable(getattr(cuda_module, "bwd_block", None))
    return status


def _cuda_root_has_toolkit(root: Path) -> bool:
    return (
        (root / "bin" / "nvcc").exists()
        and (
            (root / "include" / "cuda_runtime.h").exists()
            or (root / "targets" / "x86_64-linux" / "include" / "cuda_runtime.h").exists()
        )
    )


def _cuda_toolkit_status() -> Dict[str, Any]:
    candidates: List[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parents[1])

    prefixes = [Path(sys.prefix).resolve()]
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if base_prefix not in prefixes:
        prefixes.append(base_prefix)
    prefixes.extend(Path(sys.executable).resolve().parents)
    prefixes.append(Path("/usr/local/cuda"))

    for root in prefixes:
        if root not in candidates:
            candidates.append(root)

    for root in candidates:
        if _cuda_root_has_toolkit(root):
            return {
                "available": True,
                "cuda_home_env": os.environ.get("CUDA_HOME"),
                "cuda_path_env": os.environ.get("CUDA_PATH"),
                "nvcc_path": str(root / "bin" / "nvcc"),
                "root": str(root),
            }

    return {
        "available": False,
        "cuda_home_env": os.environ.get("CUDA_HOME"),
        "cuda_path_env": os.environ.get("CUDA_PATH"),
        "nvcc_path": str(Path(nvcc).resolve()) if nvcc else None,
        "root": None,
    }


def _torch_flex_attention_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "module": False,
        "flex_attention": False,
        "block_mask": False,
        "torch_compile": False,
        "error_type": None,
        "error": None,
        "methods": ["svg1"],
    }
    try:
        import torch

        status["torch_compile"] = hasattr(torch, "compile")
        module = importlib.import_module("torch.nn.attention.flex_attention")
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        return status

    status["module"] = True
    status["flex_attention"] = hasattr(module, "flex_attention")
    status["block_mask"] = hasattr(module, "BlockMask")
    return status


def _flash_attn_status(locations: Iterable[Path]) -> Dict[str, Any]:
    location_list = list(locations)
    source_files = []
    for path in location_list:
        source_files.extend(
            str(candidate)
            for candidate in (
                path / "__init__.py",
                path / "flash_attn_interface.py",
            )
            if candidate.exists()
        )
    return {
        "package": bool(location_list),
        "flash_attn_func": _files_contain(source_files, b"flash_attn_func"),
        "flash_attn_varlen_func": _files_contain(source_files, b"flash_attn_varlen_func"),
        "package_locations": [str(path) for path in location_list],
        "methods": ["svg1", "draft", "adacluster"],
    }


def optional_kernel_status() -> Dict[str, Any]:
    """Report optional sparse kernel availability without importing them.

    This is intentionally conservative. It answers whether an optional package
    or compiled artifact appears to be installed; actual CUDA execution still
    must be validated by a real inference smoke on a GPU node.
    """
    spas_locations = _package_locations("spas_sage_attn")
    sageattention_locations = _package_locations("sageattention")
    fastvideo_locations = _package_locations("fastvideo_kernel")
    flashomni_locations = _package_locations("flashomni")
    flashinfer_locations = _package_locations("flashinfer")
    flash_attn_locations = _package_locations("flash_attn")
    native_kernel_dirs_error = None
    try:
        native_kernel_dirs = _native_kernel_dirs()
    except RuntimeError as exc:
        native_kernel_dirs = []
        native_kernel_dirs_error = str(exc)
    native_kernel_files = _glob_existing(native_kernel_dirs, "_kernels*.so")
    repo_root = _repo_root()
    native_source_root = repo_root / "src" / "sparsevideo" / "kernels" / "native"
    spargeattn_source_root = native_source_root / "spargeattn"
    spargeattn_package_root = spargeattn_source_root / "spas_sage_attn"
    sageattention_source_root = native_source_root / "sageattention"
    sageattention_package_root = sageattention_source_root / "sageattention"
    flashomni_source_root = native_source_root / "flashomni"
    flashomni_package_root = flashomni_source_root / "flashomni"
    sta_h100_source_root = native_source_root / "sta_h100"
    spargeattn_local_qattn = bool(_glob_existing([spargeattn_package_root], "_qattn*.so"))
    spargeattn_local_fused = bool(_glob_existing([spargeattn_package_root], "_fused*.so"))
    spargeattn_local_block_sparse = _files_contain(
        [
            str(spargeattn_package_root / "__init__.py"),
            str(spargeattn_package_root / "core.py"),
        ],
        b"block_sparse_sage2_attn_cuda",
    )
    spargeattn_local_ready = (
        (spargeattn_package_root / "__init__.py").exists()
        and spargeattn_local_qattn
        and spargeattn_local_fused
    )
    spargeattn_env_ready = (
        bool(spas_locations)
        and bool(_glob_existing(spas_locations, "_qattn*.so"))
        and bool(_glob_existing(spas_locations, "_fused*.so"))
    )
    spargeattn_selected_runtime = "missing"
    if spargeattn_local_ready:
        spargeattn_selected_runtime = "sparsevideo"
    sageattention_local_qattn = bool(_glob_existing([sageattention_package_root], "_qattn_sm*.so"))
    sageattention_local_fused = bool(_glob_existing([sageattention_package_root], "_fused*.so"))
    sageattention_local_ready = (
        (sageattention_package_root / "__init__.py").exists()
        and sageattention_local_qattn
        and sageattention_local_fused
    )
    sageattention_env_ready = (
        bool(sageattention_locations)
        and bool(_glob_existing(sageattention_locations, "_qattn_sm*.so"))
        and bool(_glob_existing(sageattention_locations, "_fused*.so"))
    )
    sageattention_selected_runtime = "missing"
    if sageattention_local_ready:
        sageattention_selected_runtime = "sparsevideo"
    flashomni_local_aot = bool(
        (flashomni_package_root / "jit" / "aot_config.py").exists()
        or (flashomni_package_root / "aot_config.py").exists()
    )
    flashomni_local_native = bool(_glob_existing([flashomni_package_root], "flashomni_kernels*.so"))
    flashomni_local_ready = (
        (flashomni_package_root / "__init__.py").exists()
        and flashomni_local_aot
        and flashomni_local_native
    )
    flashomni_env_aot = bool(
        _glob_existing(flashomni_locations, "jit/aot_config.py")
        or _glob_existing(flashomni_locations, "aot_config.py")
    )
    flashomni_env_ready = (
        bool(flashomni_locations)
        and flashomni_env_aot
        and bool(_glob_existing(flashomni_locations, "flashomni_kernels*.so"))
    )
    flashomni_selected_runtime = "missing"
    if flashomni_local_ready:
        flashomni_selected_runtime = "sparsevideo"
    fastvideo_extension_files = _glob_any_existing(
        fastvideo_locations,
        ("_C/*.so", "**/_C*.so", "**/*fastvideo_kernel_ops*.so"),
    )
    sta_h100_candidate_dirs = [sta_h100_source_root]
    if (sta_h100_source_root / "build").exists():
        sta_h100_candidate_dirs.extend(path for path in (sta_h100_source_root / "build").glob("**") if path.is_dir())
    sparsevideo_sta_h100_files = _glob_any_existing(
        sta_h100_candidate_dirs,
        ("fastvideo_kernel_ops*.so", "sta_h100*.so"),
    )
    draft_mit_root = repo_root / "src" / "sparsevideo" / "kernels" / "native" / "draft_block_sparse"
    draft_mit_source = _source_dir_status(
        draft_mit_root,
        (
            "block_sparse_attn/**/*.py",
            "csrc/block_sparse_attn/**/*",
            "setup.py",
            "LICENSE",
        ),
    )
    draft_mit_extension_files = _glob_any_existing(
        [draft_mit_root],
        ("block_sparse_attn_cuda*.so", "**/block_sparse_attn_cuda*.so"),
    )

    return {
        "adacluster_kernels": {
            "triton_package": bool(_package_locations("triton")),
            "fast_kmeans_single": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels" / "native" / "adacluster",
                ("fast_kmeans_single.py",),
            ),
            "triton_cluster_sparse_attn": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels" / "native" / "adacluster",
                ("triton_cluster_sparse_attn.py",),
            ),
            "triton_cluster_sparse_attn_topk": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels" / "native" / "adacluster",
                ("triton_cluster_sparse_attn_topk.py",),
            ),
            "methods": ["adacluster"],
        },
        "draft_kernels": {
            "triton_package": bool(_package_locations("triton")),
            "upstream_backend": "mit-han-lab/Block-Sparse-Attention",
            "mit_block_sparse_attn": {
                **draft_mit_source,
                "cuda_extension": bool(draft_mit_extension_files),
                "extension_files": draft_mit_extension_files[:8],
                "selected_runtime": "sparsevideo" if draft_mit_extension_files else "missing",
            },
            "triton_block_sparse_attn": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("block_sparse_attn.py",),
            ),
            "methods": ["draft"],
        },
        "svg1_kernels": {
            "triton_package": bool(_package_locations("triton")),
            "method_source": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "methods" / "svg1",
                ("method.py",),
            ),
            "triton_placement": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "methods" / "svg1",
                ("placement.py",),
            ),
            "methods": ["svg1"],
        },
        "svg2_kernels": {
            "triton_package": bool(_package_locations("triton")),
            "triton_kmeans": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "methods" / "svg2",
                ("kmeans.py",),
            ),
            "dynamic_map": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("dynamic_map.py",),
            ),
            "triton_permute": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("permute.py",),
            ),
            "flashinfer_block_sparse": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("flashinfer_block_sparse.py",),
            ),
            "methods": ["svg2"],
        },
        "radial_kernels": {
            "method_source": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "methods" / "radial",
                ("method.py",),
            ),
            "flashinfer_bsr_wrapper": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("flashinfer_block_sparse.py",),
            ),
            "methods": ["radial"],
        },
        "flex_attention": _torch_flex_attention_status(),
        "svoo_kernels": {
            "triton_package": bool(_package_locations("triton")),
            "triton_l2norm": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("l2norm.py",),
            ),
            "triton_layernorm": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("layernorm.py",),
            ),
            "triton_modulate": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("modulate.py",),
            ),
            "co_cluster": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("co_cluster.py",),
            ),
            "dynamic_map": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("dynamic_map.py",),
            ),
            "triton_permute": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("permute.py",),
            ),
            "flashinfer_block_sparse": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("flashinfer_block_sparse.py",),
            ),
            "sparsity_counts": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "kernels",
                ("sparsity.py",),
            ),
            "sparsity_profiler": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "methods" / "svoo",
                ("sparsity.py",),
            ),
            "wan_fast_block_patch": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "processors",
                ("wan_fast_block.py",),
            ),
            "hunyuan_sparse_forward_patch": _source_dir_status(
                repo_root / "src" / "sparsevideo" / "processors",
                ("hunyuan_sparse_forward.py",),
            ),
            "methods": ["svoo"],
        },
        "spas_sage_attn": {
            "package": (spargeattn_package_root / "__init__.py").exists(),
            "qattn_extension": spargeattn_local_qattn,
            "fused_extension": spargeattn_local_fused,
            "package_locations": [str(spargeattn_package_root)] if (spargeattn_package_root / "__init__.py").exists() else [],
            "training_free_package_detected": _has_training_free_path(spas_locations),
            "training_free_runtime": spargeattn_selected_runtime == "training_free",
            "environment_runtime_detected": spargeattn_env_ready and not _has_training_free_path(spas_locations),
            "selected_runtime": spargeattn_selected_runtime,
            "external_package": {
                "package": bool(spas_locations),
                "qattn_extension": bool(_glob_existing(spas_locations, "_qattn*.so")),
                "fused_extension": bool(_glob_existing(spas_locations, "_fused*.so")),
                "package_locations": [str(path) for path in spas_locations],
                "training_free_package_detected": _has_training_free_path(spas_locations),
                "environment_runtime_detected": spargeattn_env_ready and not _has_training_free_path(spas_locations),
            },
            "sparsevideo_owned_source": _source_dir_status(
                spargeattn_source_root,
                ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.py", "**/setup.py"),
            ),
            "sparsevideo_runtime": {
                "path": str(spargeattn_source_root),
                "package": (spargeattn_package_root / "__init__.py").exists(),
                "qattn_extension": spargeattn_local_qattn,
                "fused_extension": spargeattn_local_fused,
                "block_sparse_sage2_attn_cuda": spargeattn_local_block_sparse,
                "autotune": (spargeattn_package_root / "autotune.py").exists(),
                "gpu_process_pool": (spargeattn_source_root / "tools" / "gpu_process.py").exists(),
                "hunyuan_forward_patch": _source_dir_status(
                    repo_root / "src" / "sparsevideo" / "methods" / "spargeattn",
                    ("hunyuan_forward.py",),
                ),
                "ready": spargeattn_local_ready,
            },
            "env_root": _owned_runtime_env_status("SPARSEVIDEO_SPARGEATTN_ROOT", spargeattn_source_root),
            "methods": ["spargeattn", "radial"],
        },
        "sageattention": {
            "package": (sageattention_package_root / "__init__.py").exists(),
            "qattn_extension": sageattention_local_qattn,
            "fused_extension": sageattention_local_fused,
            "package_locations": [str(sageattention_package_root)] if (sageattention_package_root / "__init__.py").exists() else [],
            "training_free_package_detected": _has_training_free_path(sageattention_locations),
            "training_free_runtime": sageattention_selected_runtime == "training_free",
            "environment_runtime_detected": sageattention_env_ready and not _has_training_free_path(sageattention_locations),
            "selected_runtime": sageattention_selected_runtime,
            "external_package": {
                "package": bool(sageattention_locations),
                "qattn_extension": bool(_glob_existing(sageattention_locations, "_qattn_sm*.so")),
                "fused_extension": bool(_glob_existing(sageattention_locations, "_fused*.so")),
                "package_locations": [str(path) for path in sageattention_locations],
                "training_free_package_detected": _has_training_free_path(sageattention_locations),
                "environment_runtime_detected": sageattention_env_ready and not _has_training_free_path(sageattention_locations),
            },
            "sparsevideo_owned_source": _source_dir_status(
                sageattention_source_root,
                ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.py", "**/setup.py"),
            ),
            "sparsevideo_runtime": {
                "path": str(sageattention_source_root),
                "package": (sageattention_package_root / "__init__.py").exists(),
                "qattn_extension": sageattention_local_qattn,
                "fused_extension": sageattention_local_fused,
                "ready": sageattention_local_ready,
            },
            "env_root": _owned_runtime_env_status("SPARSEVIDEO_SAGEATTENTION_ROOT", sageattention_source_root),
            "methods": ["radial"],
        },
        "sta_kernels": {
            "sparsevideo_fastvideo_triton": _source_dir_status(
                sta_h100_source_root / "python" / "fastvideo_kernel" / "triton_kernels",
                ("st_attn_triton.py",),
            ),
            "sparsevideo_h100": {
                "native_extension": bool(sparsevideo_sta_h100_files),
                "candidate_dirs": [str(path) for path in sta_h100_candidate_dirs],
                "files": sparsevideo_sta_h100_files,
                "source": _source_dir_status(
                    sta_h100_source_root,
                    ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/CMakeLists.txt"),
                ),
            },
            "sparsevideo_a100_block_sparse": {
                "native_extension": bool(draft_mit_extension_files),
                "files": draft_mit_extension_files,
                "source": draft_mit_source,
            },
            "external_fastvideo_kernel": {
                "package": bool(fastvideo_locations),
                "native_extension": bool(fastvideo_extension_files),
                "sta_fwd_op": _files_contain(fastvideo_extension_files, b"sta_fwd"),
                "package_locations": [str(path) for path in fastvideo_locations],
            },
            "methods": ["sta"],
        },
        "flashomni": {
            "package": (flashomni_package_root / "__init__.py").exists(),
            "aot_config": flashomni_local_aot,
            "native_extension": flashomni_local_native,
            "package_locations": [str(flashomni_package_root)] if (flashomni_package_root / "__init__.py").exists() else [],
            "training_free_package_detected": _has_training_free_path(flashomni_locations),
            "training_free_runtime": flashomni_selected_runtime == "training_free",
            "environment_runtime_detected": flashomni_env_ready and not _has_training_free_path(flashomni_locations),
            "selected_runtime": flashomni_selected_runtime,
            "external_package": {
                "package": bool(flashomni_locations),
                "aot_config": flashomni_env_aot,
                "native_extension": bool(_glob_existing(flashomni_locations, "flashomni_kernels*.so")),
                "package_locations": [str(path) for path in flashomni_locations],
                "training_free_package_detected": _has_training_free_path(flashomni_locations),
                "environment_runtime_detected": flashomni_env_ready and not _has_training_free_path(flashomni_locations),
            },
            "sparsevideo_owned_source": _source_dir_status(
                flashomni_source_root,
                ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.py", "**/setup.py"),
            ),
            "sparsevideo_runtime": {
                "path": str(flashomni_source_root),
                "package": (flashomni_package_root / "__init__.py").exists(),
                "aot_config": flashomni_local_aot,
                "native_extension": flashomni_local_native,
                "ready": flashomni_local_ready,
            },
            "env_root": _owned_runtime_env_status("SPARSEVIDEO_FLASHOMNI_ROOT", flashomni_source_root),
            "methods": ["flashomni"],
        },
        "flashinfer": {
            "package": bool(flashinfer_locations),
            "sparse_module": _has_submodule_file(flashinfer_locations, "sparse"),
            "cuda_toolkit": _cuda_toolkit_status(),
            "methods": ["adacluster", "radial", "svg2", "svoo"],
        },
        "flash_attn": _flash_attn_status(flash_attn_locations),
        "svg_svoo_fused_kernels": {
            "built_extension": bool(native_kernel_files),
            "native_extension": bool(native_kernel_files),
            "native_load_checked": False,
            "native_import_error": None,
            "native_import_error_type": None,
            "backend_env": os.environ.get("SPARSEVIDEO_FUSED_KERNEL_BACKEND", "auto"),
            "candidate_dirs": [str(path) for path in native_kernel_dirs],
            "candidate_dirs_error": native_kernel_dirs_error,
            "files": native_kernel_files,
            "source": _source_dir_status(
                native_source_root / "svg_svoo_fused",
                ("**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/CMakeLists.txt"),
            ),
            "methods": ["svg1", "svg2", "svoo"],
        },
    }


def torch_runtime_status() -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "imported": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return {
            "imported": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": False,
            "cuda_error_type": type(exc).__name__,
            "cuda_error": str(exc),
        }

    device_count = 0
    if cuda_available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            device_count = 0
    devices = []
    for device_idx in range(device_count):
        try:
            capability = torch.cuda.get_device_capability(device_idx)
            devices.append(
                {
                    "index": device_idx,
                    "name": torch.cuda.get_device_name(device_idx),
                    "capability": [int(capability[0]), int(capability[1])],
                }
            )
        except Exception:
            continue

    return {
        "imported": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
    }
