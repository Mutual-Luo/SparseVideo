from __future__ import annotations

from typing import Any, Dict, Optional

from .config import (
    _normalize_int_triple,
    _parse_normalized_seq_shape,
    normalize_seq_shape_for_warning,
)
from .models import FLASHOMNI_SPARSE_INFO_KEYS, STA_NATIVE_SEQ_SHAPES


def _flash_attn_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_func: str,
    requirement: str,
) -> Optional[str]:
    flash_attn = kernels.get("flash_attn", {})
    if flash_attn.get("load_checked") and flash_attn.get("import_error"):
        return (
            f"{requirement}. flash_attn failed to import during preflight: "
            f"{flash_attn.get('import_error_type')}: {flash_attn.get('import_error')}."
        )
    missing = []
    if not flash_attn.get("package") and not flash_attn.get("imported"):
        missing.append("flash_attn")
    if not flash_attn.get(required_func):
        missing.append(required_func)
    if not missing:
        return None
    return f"{requirement}. Missing: {missing}."


def _flashinfer_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    flashinfer = kernels.get("flashinfer", {})
    if not flashinfer.get("load_checked"):
        return (
            f"{requirement}. benchmark preflight must import FlashInfer and flashinfer.sparse; "
            "package/source presence alone is not enough to claim parity."
        )
    if flashinfer.get("import_error"):
        return (
            f"{requirement}. flashinfer failed to import during preflight: "
            f"{flashinfer.get('import_error_type')}: {flashinfer.get('import_error')}."
        )
    missing = []
    if not flashinfer.get("imported"):
        missing.append("flashinfer")
    if not flashinfer.get("sparse_imported"):
        missing.append("flashinfer.sparse")
    for attr in required_attrs:
        if not flashinfer.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing FlashInfer API(s): {missing}."


def _spas_sage_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    sparge = kernels.get("spas_sage_attn", {})
    if not sparge.get("load_checked"):
        return None
    if sparge.get("import_error"):
        return (
            f"{requirement}. spas_sage_attn failed to import during preflight: "
            f"{sparge.get('import_error_type')}: {sparge.get('import_error')}."
        )
    missing = []
    if not sparge.get("imported"):
        missing.append("spas_sage_attn")
    for attr in required_attrs:
        if not sparge.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing spas_sage_attn API(s): {missing}."


def _sageattention_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    requirement: str,
) -> Optional[str]:
    sageattention = kernels.get("sageattention", {})
    if not sageattention.get("load_checked"):
        return None
    if sageattention.get("import_error"):
        return (
            f"{requirement}. sageattention failed to import during preflight: "
            f"{sageattention.get('import_error_type')}: {sageattention.get('import_error')}."
        )
    missing = []
    if not sageattention.get("imported"):
        missing.append("sageattention")
    if not sageattention.get("sageattn"):
        missing.append("sageattn")
    if not missing:
        return None
    return f"{requirement}. Missing sageattention API(s): {missing}."


def _flashomni_load_preflight_error(
    kernels: Dict[str, Any],
    *,
    required_attrs: tuple[str, ...],
    requirement: str,
) -> Optional[str]:
    flashomni = kernels.get("flashomni", {})
    if not flashomni.get("load_checked"):
        return None
    if flashomni.get("import_error"):
        return (
            f"{requirement}. flashomni failed to import during preflight: "
            f"{flashomni.get('import_error_type')}: {flashomni.get('import_error')}."
        )
    missing = []
    if not flashomni.get("imported"):
        missing.append("flashomni")
    for attr in required_attrs:
        if not flashomni.get(attr):
            missing.append(attr)
    if not missing:
        return None
    return f"{requirement}. Missing FlashOmni API(s): {missing}."


def preflight_runtime(
    method: str,
    config: Dict[str, Any],
    device: str,
    runtime_status: Dict[str, Any],
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    kernels = runtime_status["optional_kernels"]
    torch_status = runtime_status["torch"]
    errors = []
    warnings = []

    if device.startswith("cuda"):
        if not torch_status.get("cuda_available"):
            errors.append(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )
    elif method != "dense":
        errors.append("Sparse methods require --device cuda for fair inference benchmarking.")

    fused = kernels["svg_svoo_fused_kernels"]
    fused_unavailable_message = (
        "SparseVideo _kernels extension is not detected; RMSNorm/RoPE will use the Triton/PyTorch "
        "path. Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton to benchmark that path explicitly."
    )
    if fused.get("native_load_checked") and fused.get("native_import_error") and not fused.get("native_extension"):
        native_error = str(fused.get("native_import_error")).rstrip(".")
        if fused.get("built_extension"):
            fused_unavailable_message = (
                "SparseVideo _kernels extension is built but failed to load; RMSNorm/RoPE will use the "
                "Triton/PyTorch path. "
                f"Import error: {fused.get('native_import_error_type')}: {native_error}. "
                "Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton only for an explicit non-native benchmark."
            )
        else:
            fused_unavailable_message = (
                "SparseVideo _kernels extension is unavailable; RMSNorm/RoPE will use the Triton/PyTorch path. "
                f"Native root error: {fused.get('native_import_error_type')}: {native_error}. "
                "Set SPARSEVIDEO_FUSED_KERNEL_BACKEND=triton only for an explicit non-native benchmark."
            )
    if method in ("svg1", "svg2", "svoo") and fused.get("backend_env") == "native":
        if not fused.get("native_extension"):
            errors.append(
                "SPARSEVIDEO_FUSED_KERNEL_BACKEND=native requires a loadable SparseVideo _kernels extension. "
                f"{fused_unavailable_message} "
                f"Searched: {fused.get('candidate_dirs')}."
            )
    elif (
        method in ("svg1", "svg2", "svoo")
        and fused.get("backend_env") == "auto"
        and not fused.get("native_extension")
    ):
        errors.append(fused_unavailable_message)

    spargeattn_needs_runtime = (
        method == "spargeattn"
    )
    if spargeattn_needs_runtime:
        sparge = kernels["spas_sage_attn"]
        sparge_env_error = (sparge.get("env_root") or {}).get("error")
        sparsevideo_runtime = sparge.get("sparsevideo_runtime", {})
        if model_type == "hunyuan_video":
            hunyuan_forward_patch = sparsevideo_runtime.get("hunyuan_forward_patch", {})
            if not hunyuan_forward_patch.get("source_files"):
                errors.append(
                    "spargeattn HunyuanVideo sparse/tuned paths require the SparseVideo-owned "
                    "Hunyuan forward patch to trim padded text tokens before clearing attention_mask. "
                    "Without that patch, upstream spas_sage_attn sparse kernels are not "
                    "attention_mask-equivalent."
                )
        sparsevideo_ready = (
            sparsevideo_runtime.get("package")
            and sparsevideo_runtime.get("qattn_extension")
            and sparsevideo_runtime.get("fused_extension")
        )
        if config.get("tune") or config.get("model_out_path"):
            sparsevideo_ready = (
                sparsevideo_ready
                and sparsevideo_runtime.get("autotune")
                and sparsevideo_runtime.get("gpu_process_pool")
            )
        if sparge_env_error:
            errors.append(sparge_env_error)
        elif sparsevideo_ready:
            required_attrs = (
                ("block_sparse_sage2_attn_cuda",)
                if config.get("mode") == "block_sparse"
                else ("spas_sage2_attn_meansim_cuda", "spas_sage2_attn_meansim_topk_cuda")
            )
            if config.get("tune") or config.get("model_out_path"):
                required_attrs = ("autotune",)
            if not sparge.get("load_checked"):
                errors.append(
                    "spargeattn sparse/tuned benchmark preflight must import the SparseVideo-owned "
                    "spas_sage_attn runtime; extension/source presence alone is not enough to claim parity."
                )
            else:
                load_error = _spas_sage_load_preflight_error(
                    kernels,
                    required_attrs=required_attrs,
                    requirement="spargeattn sparse/tuned paths require loadable SparseVideo-owned spas_sage_attn",
                )
                if load_error is not None:
                    errors.append(load_error)
        elif sparge.get("training_free_runtime") or sparge.get("training_free_package_detected"):
            errors.append(
                "spargeattn resolves spas_sage_attn from training_free/, which is reference-only. "
                "SparseVideo-owned source exists under src/sparsevideo/kernels/native/spargeattn, "
                "but its _qattn/_fused extensions are not built."
            )
        elif sparge.get("environment_runtime_detected") or (
            sparge.get("package") and sparge.get("qattn_extension") and sparge.get("fused_extension")
        ):
            errors.append(
                "spargeattn sparse modes require the SparseVideo-owned spas_sage_attn runtime under "
                "src/sparsevideo/kernels/native/spargeattn. Environment spas_sage_attn packages are "
                "not accepted for SparseVideo runtime parity."
            )
        else:
            errors.append(
                "spargeattn sparse modes require spas_sage_attn with _qattn and _fused extensions built. "
                "Build the SparseVideo-owned source at src/sparsevideo/kernels/native/spargeattn."
            )
    if method == "radial" and config.get("use_sage_attention"):
        sparge = kernels["spas_sage_attn"]
        sparge_env_error = (sparge.get("env_root") or {}).get("error")
        sparsevideo_runtime = sparge.get("sparsevideo_runtime", {})
        sparsevideo_ready = (
            sparsevideo_runtime.get("package")
            and sparsevideo_runtime.get("qattn_extension")
            and sparsevideo_runtime.get("fused_extension")
            and sparsevideo_runtime.get("block_sparse_sage2_attn_cuda")
        )
        if sparge_env_error:
            errors.append(sparge_env_error)
        elif not sparsevideo_ready:
            errors.append(
                "radial use_sage_attention requires SparseVideo-owned spas_sage_attn "
                "with _qattn/_fused extensions and block_sparse_sage2_attn_cuda under "
                "src/sparsevideo/kernels/native/spargeattn."
            )
        elif sparge.get("training_free_runtime") and sparge.get("selected_runtime") != "sparsevideo":
            errors.append(
                "radial use_sage_attention resolves spas_sage_attn from training_free/, which is reference-only. "
                "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/spargeattn."
            )
        elif not sparge.get("load_checked"):
            errors.append(
                "radial use_sage_attention benchmark preflight must import the SparseVideo-owned "
                "spas_sage_attn runtime; extension/source presence alone is not enough to claim parity."
            )
        else:
            load_error = _spas_sage_load_preflight_error(
                kernels,
                required_attrs=("block_sparse_sage2_attn_cuda",),
                requirement="radial use_sage_attention requires loadable SparseVideo-owned spas_sage_attn",
            )
            if load_error is not None:
                errors.append(load_error)
        if _has_dense_warmup(config):
            sageattention = kernels.get("sageattention", {})
            sage_env_error = (sageattention.get("env_root") or {}).get("error")
            sage_runtime = sageattention.get("sparsevideo_runtime", {})
            sage_ready = (
                sage_runtime.get("package")
                and sage_runtime.get("qattn_extension")
                and sage_runtime.get("fused_extension")
            )
            if sage_env_error:
                errors.append(sage_env_error)
            elif not sage_ready:
                errors.append(
                    "radial use_sage_attention with dense warmup requires the SparseVideo-owned "
                    "SageAttention dense backend with _qattn_sm* and _fused extensions under "
                    "src/sparsevideo/kernels/native/sageattention."
                )
            elif sageattention.get("training_free_runtime") and sageattention.get("selected_runtime") != "sparsevideo":
                errors.append(
                    "radial use_sage_attention resolves sageattention from training_free/, which is reference-only. "
                    "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/sageattention."
                )
            elif not sageattention.get("load_checked"):
                errors.append(
                    "radial use_sage_attention dense warmup benchmark preflight must import the "
                    "SparseVideo-owned SageAttention runtime; extension/source presence alone is not enough "
                    "to claim parity."
                )
            else:
                load_error = _sageattention_load_preflight_error(
                    kernels,
                    requirement="radial use_sage_attention dense warmup requires loadable SparseVideo-owned SageAttention",
                )
                if load_error is not None:
                    errors.append(load_error)

    if method == "radial":
        radial = kernels.get("radial_kernels", {})
        if not radial.get("method_source", {}).get("source_files"):
            errors.append(
                "radial requires SparseVideo-owned radial method source at "
                "src/sparsevideo/methods/radial/method.py."
            )
        if not radial.get("flashinfer_bsr_wrapper", {}).get("source_files"):
            errors.append(
                "radial requires SparseVideo-owned FlashInfer BSR wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        radial_sources_ready = all(
            radial.get(key, {}).get("source_files")
            for key in ("method_source", "flashinfer_bsr_wrapper")
        )
        if radial_sources_ready:
            radial_runtime = radial.get("owned_runtime", {})
            if not (radial.get("load_checked") or radial_runtime.get("load_checked")):
                errors.append(
                    "radial benchmark preflight must import the SparseVideo-owned radial method and "
                    "FlashInfer BSR wrapper modules; source-file presence alone is not enough to claim parity."
                )
            elif radial_runtime.get("import_error"):
                errors.append(
                    "radial owned method/BSR wrapper modules failed to import during preflight: "
                    f"{radial_runtime.get('import_error_type')}: {radial_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "radial_bsr_mask",
                    "shrink_mask_strict",
                    "radial_flashinfer_attention",
                    "build_bsr_from_mask",
                    "variable_block_sparse_attn",
                    "bsr_sparse_attn",
                    "ensure_cuda_home_for_flashinfer_jit",
                    "expand_attention_mask",
                    "radial_window_width",
                ):
                    if not radial_runtime.get(attr):
                        missing.append(attr)
                if config.get("use_sage_attention"):
                    for attr in (
                        "radial_sage_attention",
                        "radial_sage_dense_attention",
                        "sparge_mask_convert",
                        "sparge_sage_qk_block_sizes",
                        "radial_append_tail_blocks",
                    ):
                        if not radial_runtime.get(attr):
                            missing.append(attr)
                if missing:
                    errors.append(
                        "radial owned method/BSR wrapper runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "flashomni" and config.get("implementation") == "upstream":
        flashomni = kernels["flashomni"]
        flashomni_env_error = (flashomni.get("env_root") or {}).get("error")
        sparsevideo_runtime = flashomni.get("sparsevideo_runtime", {})
        sparsevideo_ready = bool(sparsevideo_runtime.get("ready"))
        if flashomni_env_error:
            errors.append(flashomni_env_error)
        elif sparsevideo_ready:
            if not flashomni.get("load_checked"):
                errors.append(
                    "flashomni implementation=upstream benchmark preflight must import the "
                    "SparseVideo-owned FlashOmni runtime; extension/source presence alone is not "
                    "enough to claim parity."
                )
            else:
                load_error = _flashomni_load_preflight_error(
                    kernels,
                    required_attrs=(
                        "native_extension_imported",
                        "owned_runtime",
                        "batch_flashomni_fa_with_ragged_kv_wrapper",
                        "segment_packbits",
                        "torch_ops_flashomni_kernels",
                        "torch_ops_batch_sparseFA_with_kv_plan",
                        "torch_ops_batch_sparseFA_with_ragged_kv_run",
                    ),
                    requirement=(
                        "flashomni implementation=upstream requires loadable "
                        "SparseVideo-owned FlashOmni CUDA/C++ ops"
                    ),
                )
                if load_error is not None:
                    errors.append(load_error)
        elif not flashomni.get("package"):
            errors.append(
                "flashomni implementation=upstream requires the SparseVideo-owned "
                "flashomni runtime under src/sparsevideo/kernels/native/flashomni."
            )
        elif not flashomni.get("aot_config"):
            errors.append(
                "flashomni implementation=upstream requires SparseVideo-owned FlashOmni "
                "AOT kernels under src/sparsevideo/kernels/native/flashomni."
            )
        elif not flashomni.get("native_extension"):
            errors.append(
                "flashomni implementation=upstream requires SparseVideo-owned "
                "flashomni_kernels*.so from the local AOT build."
            )
        elif flashomni.get("training_free_runtime") or flashomni.get("training_free_package_detected"):
            errors.append(
                "flashomni resolves from training_free/, which is reference-only. "
                "Build/use the SparseVideo-owned runtime under src/sparsevideo/kernels/native/flashomni."
            )
        elif flashomni.get("environment_runtime_detected") or flashomni.get("selected_runtime") != "sparsevideo":
            errors.append(
                "flashomni implementation=upstream requires the SparseVideo-owned runtime under "
                "src/sparsevideo/kernels/native/flashomni. Environment flashomni packages are "
                "not accepted for SparseVideo runtime parity."
            )
        owned_source = flashomni.get("sparsevideo_owned_source", {})
        if not owned_source.get("source_files"):
            errors.append(
                "flashomni has no SparseVideo-owned FlashOmni native source under "
                "src/sparsevideo/kernels/native/flashomni; this is not package-ready kernel parity."
            )
        if config.get("sparse_pattern", "explicit") == "explicit":
            missing = [key for key in FLASHOMNI_SPARSE_INFO_KEYS if config.get(key) is None]
            if missing:
                errors.append(
                    "flashomni sparse_pattern=explicit follows upstream FlashOmni and requires "
                    "precomputed sparse_info, sparse_kv_info, sparse_info_indptr, and "
                    f"sparse_kv_info_indptr tensors. Missing: {missing}. "
                    "The inference CLI cannot synthesize an upstream video sparsity policy."
                )
            else:
                warnings.append(
                    "flashomni sparse_pattern=explicit has caller-provided sparse-info tensors, "
                    "so SparseVideo can verify the FlashOmni kernel adapter dispatch. The current "
                    "training_free/FlashOmni reference publishes the engine/API and synthetic "
                    "benchmark mask helper, but not a reusable Wan/Hunyuan video sparse-info "
                    "policy; this run is not benchmark-ready video-method parity unless those "
                    "tensors are proven to come from an upstream-compatible video policy."
                )
    if method == "flashomni" and config.get("sparse_pattern") == "local_qk_topk":
        errors.append(
            "flashomni sparse_pattern=local_qk_topk uses SparseVideo's block-mean top-k "
            "diagnostic policy, not upstream FlashOmni video-method parity."
        )
    if method == "flashomni" and config.get("sparse_pattern") == "global_random":
        errors.append(
            "flashomni sparse_pattern=global_random matches FlashOmni's upstream synthetic kernel benchmark mask, "
            "not a video diffusion quality-parity sparsity policy. Use it for native-kernel smoke/speed checks only; "
            "use sparse_pattern=explicit with upstream-compatible sparse-info tensors for real method comparisons."
        )
    if (
        method == "flashomni"
        and config.get("sparse_pattern") == "paper_mmdit"
        and model_type == "hunyuan_video"
        and bool(config.get("use_sparse_gemm", False))
    ):
        errors.append(
            "flashomni Hunyuan paper_mmdit sparse-GEMM path is not supported for inference. "
            "Use use_sparse_gemm=false. The retained GEMM code caused measured quality "
            "degradation and performance regression."
        )
    if method == "svoo":
        if not kernels["flashinfer"].get("package"):
            errors.append("svoo sparse path requires the flashinfer package.")
        elif not kernels["flashinfer"].get("sparse_module"):
            errors.append("svoo sparse path requires flashinfer.sparse APIs.")
        elif not kernels["flashinfer"].get("cuda_toolkit", {}).get("available"):
            errors.append(
                "svoo sparse path requires a CUDA toolkit with nvcc for FlashInfer sparse JIT. "
                "Set CUDA_HOME/CUDA_PATH or put nvcc on PATH."
            )
        else:
            error = _flashinfer_load_preflight_error(
                kernels,
                required_attrs=(
                    "sparse_variable_block_sparse_attention_wrapper",
                    "sparse_canonicalize_torch_dtype",
                    "sparse_mask_mode",
                    "sparse_pos_encoding_mode",
                    "sparse_determine_attention_backend",
                    "sparse_get_batch_prefill_module",
                ),
                requirement="svoo sparse path requires loadable FlashInfer sparse APIs",
            )
            if error is not None:
                errors.append(error)

    if method == "svoo":
        svoo = kernels.get("svoo_kernels", {})
        if not svoo.get("triton_package"):
            errors.append("svoo requires the triton package for its upstream co-clustering kernels.")
        if not svoo.get("triton_l2norm", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned Triton L2 normalization source at "
                "src/sparsevideo/kernels/l2norm.py."
            )
        if not svoo.get("triton_layernorm", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Triton layernorm source at "
                "src/sparsevideo/kernels/layernorm.py."
            )
        if not svoo.get("triton_modulate", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Triton modulation source at "
                "src/sparsevideo/kernels/modulate.py."
            )
        if not svoo.get("wan_fast_block_patch", {}).get("source_files"):
            errors.append(
                "svoo Wan upstream inference requires SparseVideo-owned Wan fast-block patch source at "
                "src/sparsevideo/processors/wan_fast_block.py."
            )
        if not svoo.get("hunyuan_sparse_forward_patch", {}).get("source_files"):
            errors.append(
                "svoo Hunyuan upstream inference requires SparseVideo-owned sparse-forward patch source at "
                "src/sparsevideo/processors/hunyuan_sparse_forward.py."
            )
        if not svoo.get("co_cluster", {}).get("source_files"):
            errors.append("svoo requires SparseVideo-owned co-clustering source at src/sparsevideo/kernels/co_cluster.py.")
        if not svoo.get("dynamic_map", {}).get("source_files"):
            errors.append("svoo requires SparseVideo-owned dynamic-map source at src/sparsevideo/kernels/dynamic_map.py.")
        if not svoo.get("triton_permute", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned Triton permutation source at "
                "src/sparsevideo/kernels/permute.py."
            )
        if not svoo.get("flashinfer_block_sparse", {}).get("source_files"):
            errors.append(
                "svoo requires SparseVideo-owned FlashInfer block sparse wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        if config.get("measure_attention_sparsity"):
            if not svoo.get("sparsity_profiler", {}).get("source_files"):
                errors.append(
                    "svoo measure_attention_sparsity requires SparseVideo-owned profiler source at "
                    "src/sparsevideo/methods/svoo/sparsity.py."
                )
            if not svoo.get("sparsity_counts", {}).get("source_files"):
                errors.append(
                    "svoo measure_attention_sparsity requires SparseVideo-owned optional Triton count source at "
                    "src/sparsevideo/kernels/sparsity.py."
                )
        required_source_keys = [
            "triton_l2norm",
            "triton_layernorm",
            "triton_modulate",
            "co_cluster",
            "dynamic_map",
            "triton_permute",
            "flashinfer_block_sparse",
        ]
        if config.get("measure_attention_sparsity"):
            required_source_keys.extend(["sparsity_counts", "sparsity_profiler"])
        sources_ready = all(
            svoo.get(key, {}).get("source_files")
            for key in required_source_keys
        )
        if sources_ready:
            svoo_runtime = svoo.get("owned_triton_runtime", {})
            if not (svoo.get("load_checked") or svoo_runtime.get("load_checked")):
                errors.append(
                    "svoo benchmark preflight must import the SparseVideo-owned Triton/FlashInfer helper modules; "
                    "source-file presence alone is not enough to claim parity."
                )
            elif svoo_runtime.get("import_error"):
                errors.append(
                    "svoo owned Triton/FlashInfer helper modules failed to import during preflight: "
                    f"{svoo_runtime.get('import_error_type')}: {svoo_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "triton_l2norm_forward",
                    "triton_layernorm_forward",
                    "triton_modulate_shift_forward",
                    "triton_modulate_gate_residual_forward",
                    "co_cluster_tokens",
                    "co_cluster_assign",
                    "identify_dynamic_map",
                    "permute_tensor_by_labels_triton",
                    "apply_inverse_permutation_triton",
                    "variable_block_sparse_attn",
                    "hunyuan_flashinfer_varlen_attn",
                ):
                    if not svoo_runtime.get(attr):
                        missing.append(attr)
                if config.get("measure_attention_sparsity"):
                    for attr in ("counts_from_sorted_probabilities_triton", "compute_exact_attention_sparsity"):
                        if not svoo_runtime.get(attr):
                            missing.append(attr)
                if missing:
                    errors.append(
                        "svoo owned Triton/FlashInfer runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "svg2":
        svg2 = kernels.get("svg2_kernels", {})
        if not svg2.get("triton_package"):
            errors.append("svg2 requires the triton package for its upstream-style k-means kernels.")
        if not svg2.get("triton_kmeans", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned Sparse-VideoGen Triton k-means source at "
                "src/sparsevideo/methods/svg2/kmeans.py."
            )
        if not svg2.get("dynamic_map", {}).get("source_files"):
            errors.append("svg2 requires SparseVideo-owned dynamic-map source at src/sparsevideo/kernels/dynamic_map.py.")
        if not svg2.get("triton_permute", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned Triton permutation source at "
                "src/sparsevideo/kernels/permute.py."
            )
        if not svg2.get("flashinfer_block_sparse", {}).get("source_files"):
            errors.append(
                "svg2 requires SparseVideo-owned FlashInfer block sparse wrapper source at "
                "src/sparsevideo/kernels/flashinfer_block_sparse.py."
            )
        svg2_sources_ready = all(
            svg2.get(key, {}).get("source_files")
            for key in (
                "triton_kmeans",
                "dynamic_map",
                "triton_permute",
                "flashinfer_block_sparse",
            )
        )
        if svg2_sources_ready:
            svg2_runtime = svg2.get("owned_triton_runtime", {})
            if not (svg2.get("load_checked") or svg2_runtime.get("load_checked")):
                errors.append(
                    "svg2 benchmark preflight must import the SparseVideo-owned Triton/FlashInfer helper modules; "
                    "source-file presence alone is not enough to claim parity."
                )
            elif svg2_runtime.get("import_error"):
                errors.append(
                    "svg2 owned Triton/FlashInfer helper modules failed to import during preflight: "
                    f"{svg2_runtime.get('import_error_type')}: {svg2_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "triton_kmeans",
                    "euclid_assign_triton",
                    "centroid_update_triton",
                    "identify_dynamic_map",
                    "identify_dynamic_map_global",
                    "permute_tensor_by_labels_triton",
                    "apply_inverse_permutation_triton",
                    "variable_block_sparse_attn",
                    "hunyuan_flashinfer_varlen_attn",
                ):
                    if not svg2_runtime.get(attr):
                        missing.append(attr)
                if missing:
                    errors.append(
                        "svg2 owned Triton/FlashInfer runtime is missing loadable API(s): "
                        f"{missing}."
                    )

    if method == "svg1":
        svg1 = kernels.get("svg1_kernels", {})
        if not svg1.get("triton_package"):
            errors.append("svg1 requires the triton package for its upstream Sparse-VideoGen placement kernels.")
        if not svg1.get("method_source", {}).get("source_files"):
            errors.append(
                "svg1 requires SparseVideo-owned Sparse-VideoGen method source at "
                "src/sparsevideo/methods/svg1/method.py."
            )
        if not svg1.get("triton_placement", {}).get("source_files"):
            errors.append(
                "svg1 requires SparseVideo-owned Sparse-VideoGen Triton placement source at "
                "src/sparsevideo/methods/svg1/placement.py."
            )
        svg1_sources_ready = all(
            svg1.get(key, {}).get("source_files")
            for key in ("method_source", "triton_placement")
        )
        if svg1_sources_ready:
            svg1_runtime = svg1.get("owned_triton_runtime", {})
            if not (svg1.get("load_checked") or svg1_runtime.get("load_checked")):
                errors.append(
                    "svg1 benchmark preflight must import the SparseVideo-owned SVG method and "
                    "Triton placement modules; source-file presence alone is not enough to claim parity."
                )
            elif svg1_runtime.get("import_error"):
                errors.append(
                    "svg1 owned method/Triton placement modules failed to import during preflight: "
                    f"{svg1_runtime.get('import_error_type')}: {svg1_runtime.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "owned_runtime",
                    "svg_attention",
                    "svg_flex_attention",
                    "svg1_dense_attention",
                    "profile_masks",
                    "svg_profile_mask_rows",
                    "build_svg_block_mask",
                    "svg_kv_blocks",
                    "svg_kv_block_partitions",
                    "svg_common_mask",
                    "place_svg_heads",
                    "restore_svg_heads",
                    "round_svg_window_width",
                    "svg_window_width",
                    "sparsity_to_width",
                    "resolve_prompt_length",
                    "sparse_head_placement",
                    "hidden_states_placement",
                    "sparse_head_placement_kernel",
                    "hidden_states_placement_kernel",
                ):
                    if not svg1_runtime.get(attr):
                        missing.append(attr)
                if model_type == "hunyuan_video" and not svg1_runtime.get("svg1_hunyuan_flash_attn_varlen"):
                    missing.append("svg1_hunyuan_flash_attn_varlen")
                if missing:
                    errors.append(
                        "svg1 owned method/Triton placement runtime is missing loadable API(s): "
                        f"{missing}."
                    )
        flex = kernels.get("flex_attention", {})
        missing = []
        if not flex.get("module"):
            missing.append("torch.nn.attention.flex_attention")
        if not flex.get("flex_attention"):
            missing.append("flex_attention")
        if not flex.get("block_mask"):
            missing.append("BlockMask")
        if not flex.get("torch_compile"):
            missing.append("torch.compile")
        if missing:
            detail = ""
            if flex.get("error"):
                detail = f" Import error: {flex.get('error_type')}: {flex.get('error')}."
            errors.append(
                "svg1 requires PyTorch FlexAttention APIs for the upstream Sparse-VideoGen sparse path. "
                f"Missing: {missing}.{detail}"
            )
        if model_type == "hunyuan_video" and _has_dense_warmup(config):
            error = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_varlen_func",
                requirement=(
                    "svg1 Hunyuan dense warmup requires FlashAttention varlen, matching "
                    "Sparse-VideoGen's Hunyuan SVG path"
                ),
            )
            if error is not None:
                errors.append(error)

    if method == "adacluster":
        adacluster = kernels["adacluster_kernels"]
        if not adacluster.get("triton_package"):
            errors.append("adacluster requires the triton package for its upstream fast_kmeans_single/sparse kernels.")
        if not adacluster["fast_kmeans_single"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned upstream fast_kmeans_single source at "
                "src/sparsevideo/kernels/native/adacluster/fast_kmeans_single.py."
            )
        if not adacluster["triton_cluster_sparse_attn"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned upstream triton_cluster_sparse_attn source at "
                "src/sparsevideo/kernels/native/adacluster/triton_cluster_sparse_attn.py."
            )
        if not adacluster["triton_cluster_sparse_attn_topk"].get("source_files"):
            errors.append(
                "adacluster requires SparseVideo-owned optimized triton_cluster_sparse_attn_topk source at "
                "src/sparsevideo/kernels/native/adacluster/triton_cluster_sparse_attn_topk.py."
            )
        adacluster_runtime = adacluster.get("owned_triton_runtime", {})
        if not (adacluster.get("load_checked") or adacluster_runtime.get("load_checked")):
            errors.append(
                "adacluster benchmark preflight must import the SparseVideo-owned upstream Triton kernels; "
                "source-file presence alone is not enough to claim parity."
            )
        elif adacluster_runtime.get("import_error"):
            errors.append(
                "adacluster owned Triton kernels failed to import during preflight: "
                f"{adacluster_runtime.get('import_error_type')}: {adacluster_runtime.get('import_error')}."
            )
        else:
            missing = []
            for attr in (
                "imported",
                "owned_runtime",
                "flash_kmeans_single",
                "triton_cluster_sparse_attn",
                "triton_cluster_sparse_attn_topk",
                "kmeans_jit_kernels",
                "cluster_sparse_attn_jit_kernel",
                "cluster_sparse_attn_topk_jit_kernel",
            ):
                if not adacluster_runtime.get(attr):
                    missing.append(attr)
            if missing:
                errors.append(
                    "adacluster owned Triton runtime is missing loadable API(s): "
                    f"{missing}."
                )
        if model_type == "hunyuan_video" and _has_dense_warmup(config):
            error = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_func",
                requirement=(
                    "adacluster Hunyuan dense warmup requires FlashAttention, matching "
                    "the upstream Hunyuan AdaCluster path"
                ),
            )
            if error is not None:
                errors.append(error)

    if method == "draft":
        message = None
        if model_type in (None, "wan", "hunyuan_video") and _has_dense_warmup(config):
            message = _flash_attn_preflight_error(
                kernels,
                required_func="flash_attn_varlen_func",
                requirement="draft dense warmup requires FlashAttention varlen for upstream parity",
            )
        if message is not None:
            errors.append(message)
        draft = kernels["draft_kernels"]
        mit_backend = draft.get("mit_block_sparse_attn", {})
        mit_ready = mit_backend.get("source_files") and mit_backend.get("cuda_extension")
        if not mit_ready:
            errors.append(
                "draft upstream parity requires SparseVideo-owned MIT Han Lab "
                "Block-Sparse-Attention source and block_sparse_attn_cuda extension under "
                "src/sparsevideo/kernels/native/draft_block_sparse."
            )
        elif draft.get("mit_load_checked") or mit_backend.get("load_checked"):
            mit_load_error = None
            if mit_backend.get("import_error"):
                mit_load_error = (
                    "draft MIT Block-Sparse-Attention backend failed to import during preflight: "
                    f"{mit_backend.get('import_error_type')}: {mit_backend.get('import_error')}."
                )
            else:
                missing = []
                for attr in (
                    "imported",
                    "cuda_extension_imported",
                    "owned_runtime",
                    "block_sparse_attn_func",
                    "cuda_fwd_block",
                    "cuda_bwd_block",
                ):
                    if not mit_backend.get(attr):
                        missing.append(attr)
                if missing:
                    mit_load_error = (
                        "draft MIT Block-Sparse-Attention backend is missing loadable API(s): "
                        f"{missing}."
                    )
            if mit_load_error is not None:
                errors.append(mit_load_error)

    if method == "radial":
        flashinfer = kernels["flashinfer"]
        if not flashinfer.get("package"):
            message = (
                "radial FlashInfer is not importable."
            )
        elif not flashinfer.get("sparse_module"):
            message = (
                "radial requires flashinfer.sparse for the upstream sparse kernel path."
            )
        elif "cuda_toolkit" in flashinfer and not flashinfer.get("cuda_toolkit", {}).get("available"):
            message = (
                "radial FlashInfer sparse kernels require a CUDA toolkit with nvcc for JIT. "
                "Set CUDA_HOME/CUDA_PATH or put nvcc on PATH before benchmarking."
            )
        else:
            message = None
        if message is not None:
            errors.append(message)
        else:
            required_attrs = (
                "top_level_block_sparse_attention_wrapper",
                "top_level_single_prefill_with_kv_cache",
                "top_level_merge_state",
            )
            required_attrs = required_attrs + (
                "sparse_variable_block_sparse_attention_wrapper",
                "sparse_canonicalize_torch_dtype",
                "sparse_mask_mode",
                "sparse_pos_encoding_mode",
                "sparse_determine_attention_backend",
                "sparse_get_batch_prefill_module",
            )
            load_error = _flashinfer_load_preflight_error(
                kernels,
                required_attrs=required_attrs,
                requirement="radial requires loadable FlashInfer sparse APIs for the upstream sparse path",
            )
            if load_error is not None:
                errors.append(load_error)

    if method == "svg2":
        flashinfer = kernels["flashinfer"]
        if not flashinfer.get("package"):
            errors.append("svg2 sparse path requires the flashinfer package.")
        elif not flashinfer.get("sparse_module"):
            errors.append("svg2 sparse path requires flashinfer.sparse APIs.")
        elif "cuda_toolkit" in flashinfer and not flashinfer.get("cuda_toolkit", {}).get("available"):
            errors.append(
                "svg2 FlashInfer sparse kernels require a CUDA toolkit with nvcc for JIT. "
                "Set CUDA_HOME/CUDA_PATH or put nvcc on PATH before benchmarking."
            )
        else:
            load_error = _flashinfer_load_preflight_error(
                kernels,
                required_attrs=(
                    "sparse_variable_block_sparse_attention_wrapper",
                    "sparse_canonicalize_torch_dtype",
                    "sparse_mask_mode",
                    "sparse_pos_encoding_mode",
                    "sparse_determine_attention_backend",
                    "sparse_get_batch_prefill_module",
                ),
                requirement="svg2 requires loadable FlashInfer sparse APIs for the upstream sparse path",
            )
            if load_error is not None:
                errors.append(load_error)

    if method == "sta":
        sta = kernels["sta_kernels"]
        sta_mode = config.get("STA_mode", "STA_inference")
        if sta_mode not in ("STA_inference", "STA_searching"):
            errors.append(
                "sta supports STA_inference in pipelines and STA_searching for mask calibration; "
                "use python -m sparsevideo.methods.sta.search tune for STA_tuning."
            )
        if sta_mode == "STA_searching":
            warnings.append(
                "sta STA_searching returns full-window STA outputs while recording sparse-window losses; "
                "it is a calibration run, not a speed benchmark."
            )
        tile_size = _normalize_int_triple(config.get("tile_size", [6, 8, 8]))
        if tile_size != (6, 8, 8):
            errors.append(
                "sta tile_size differs from FastVideo's fixed upstream tile_size=(6,8,8); "
                "SparseVideo rejects the non-upstream generalized STA path for parity runs."
            )
        has_ampere = _has_ampere_device(torch_status)
        if has_ampere and not sta.get("sparsevideo_a100_block_sparse", {}).get("source", {}).get("source_files"):
            errors.append(
                "sta A100 block-sparse CUDA source is missing under "
                "src/sparsevideo/kernels/native/draft_block_sparse; A100 STA cannot be used for speed claims."
            )
        seq_shape = normalize_seq_shape_for_warning(config.get("seq_shape"))
        if seq_shape is None:
            message = (
                "sta seq_shape is not set. SparseVideo will infer the video layout from token length; "
                "FastVideo STA native shapes are "
                f"{sorted(STA_NATIVE_SEQ_SHAPES)}."
            )
            warnings.append(message)
        elif seq_shape not in STA_NATIVE_SEQ_SHAPES:
            parsed_seq_shape = _parse_normalized_seq_shape(seq_shape)
            if parsed_seq_shape is None:
                errors.append(f"sta seq_shape={seq_shape} is invalid; expected TxHxW.")
            else:
                warnings.append(
                    f"sta seq_shape={seq_shape} uses SparseVideo's generalized STA A100 block-sparse CUDA path "
                    "for this backbone's inferred tile-padded video layout."
                )
        if has_ampere:
            a100_extension = bool(sta.get("sparsevideo_a100_block_sparse", {}).get("native_extension"))
            a100_usable = a100_extension
            if sta.get("a100_block_sparse_load_checked"):
                a100_usable = bool(sta.get("a100_block_sparse_ready"))
                if sta.get("a100_import_error"):
                    errors.append(
                        "sta A100 block-sparse CUDA backend failed to load during preflight: "
                        f"{sta.get('a100_import_error_type')}: {sta.get('a100_import_error')}."
                    )
            if not a100_usable:
                errors.append(
                    "sta A100 block-sparse CUDA backend is not available as SparseVideo-owned native code; "
                    "strict STA speed runs on A100 require this backend."
                )

    return {"errors": errors, "warnings": warnings}



def _has_ampere_device(torch_status: Dict[str, Any]) -> bool:
    for device in torch_status.get("cuda_devices") or []:
        capability = device.get("capability") or []
        if capability and int(capability[0]) == 8:
            return True
    return False


def _has_dense_warmup(config: Dict[str, Any]) -> bool:
    return (
        float(config.get("dense_warmup_step_ratio", 0) or 0) > 0
        or float(config.get("dense_warmup_layer_ratio", 0) or 0) > 0
    )
