#!/usr/bin/env python3
"""SparseVideo Diffusers inference entrypoint.

This file is intentionally the orchestration layer. Model tables, config
resolution, pipeline adapters, preflight checks, and output helpers live in
scripts/_infer so this script stays readable as a package usability test.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SRC_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from _infer_diffusers.args import METHODS, build_parser, parse_method_config, read_prompt
from _infer_diffusers.config import (
    apply_draft_runtime_layout_defaults,
    apply_flashomni_hunyuan_quality_defaults,
    default_num_frames,
    default_svoo_sparsity_csv_path,
    draft_upstream_layout_error,
    materialize_method_config_values,
    model_quality_warnings,
    model_shape_preflight_errors,
    normalize_spargeattn_model_out_path,
    radial_flashinfer_layout_warning,
    sta_layout_preflight_messages,
    validate_method_config,
)
from _infer_diffusers.models import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MODEL_ALIASES,
    MODEL_SPECS,
    STA_STRATEGY_SHAPES,
    STA_UNSUPPORTED_STRATEGY_MODELS,
)
from _infer_diffusers.pipeline import (
    _ltx_single_file_checkpoint,
    _resolve_ltx_text_component_root,
    apply_hunyuan_i2v_prompt_template_compat,
    build_call_kwargs,
    call_pipeline_with_model_compat,
    configure_method_runtime_env,
    infer_hunyuan_prompt_length,
    load_pipeline,
    parse_dtype,
    prepare_pipeline,
    resolve_model_id,
    resolve_scheduler_flow_shift,
    seed_everything,
    should_defer_fused_native_kernel_load,
    should_preload_fused_native_kernels,
)
from _infer_diffusers.preflight import preflight_runtime
from _infer_diffusers.profiles import (
    UPSTREAM_INFERENCE_PROFILES,
    apply_profile_runtime_defaults,
    finalize_runtime_defaults,
    resolve_inference_profile,
)
from _infer_diffusers.utils import (
    append_metrics,
    configure_torch_compile_logging,
    cuda_memory_gb,
    make_output_file,
    maybe_save_spargeattn_tuned_state,
    pipeline_output_summary,
    print_final_run_metrics,
    print_run_summary,
    quiet_runtime_status_call,
    sparse_attention_handle_summary,
    sparse_method_supported,
    sparsevideo_source_fingerprints,
    sync_if_cuda,
    unsupported_sparse_method_message,
    validate_svoo_warmup_status,
)


def run(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[MODEL_ALIASES[args.model]]
    fps = args.fps if args.fps is not None else spec.fps
    strict_kernels = args.strict_kernels or not args.allow_debug_fallbacks
    profile_method = args.profile_for_method or args.method
    if args.num_frames is not None:
        num_frames = args.num_frames
    elif args.duration_seconds is not None:
        num_frames = default_num_frames(args.duration_seconds, fps)
    else:
        num_frames = spec.default_frames
    steps = args.num_inference_steps if args.num_inference_steps is not None else spec.default_steps
    try:
        profile = resolve_inference_profile(args.profile, spec, profile_method)
    except ValueError as exc:
        finalize_runtime_defaults(args)
        height = args.height if args.height is not None else DEFAULT_HEIGHT
        width = args.width if args.width is not None else DEFAULT_WIDTH
        model_id = resolve_model_id(spec, args.model_root, args.model_path)
        output_file = make_output_file(args, spec.key, args.method, num_frames)
        failed_metrics = {
            "model": spec.key,
            "model_arg": args.model,
            "model_id": model_id,
            "method": args.method,
            "method_config": {},
            "profile": args.profile,
            "profile_method": profile_method,
            "profile_overrides": {},
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "requested_duration_seconds": args.duration_seconds,
            "num_inference_steps": steps,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_offload": args.cpu_offload,
            "cpu_offload_mode": args.cpu_offload_mode,
            "vae_dtype": args.vae_dtype,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
            "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
            "strict_kernels": strict_kernels,
            "allow_debug_fallbacks": args.allow_debug_fallbacks,
            "seed": args.seed,
            "output_file": str(output_file),
            "scheduler_flow_shift": None,
            "wan_flow_shift": None,
            "runtime": {"preflight": {"errors": [str(exc)], "warnings": []}},
            "status": "failed",
            "failed_stage": "profile",
            "timings": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.dry_run:
            append_metrics(args.metrics_file, failed_metrics)
        print_final_run_metrics(args, failed_metrics)
        return 1
    height, width, fps, num_frames, steps = apply_profile_runtime_defaults(
        args, profile, fps, num_frames, steps,
    )
    if args.vae_dtype is None and spec.family == "wan":
        args.vae_dtype = "fp32"

    configure_method_runtime_env(args.method)
    import sparsevideo

    user_method_config = parse_method_config(args)
    method_config = sparsevideo.default_method_config(
        args.method, num_inference_steps=steps, model_family=spec.family, model_key=spec.key,
    )
    if profile_method == args.method:
        method_config.update(copy.deepcopy(profile.get("method_config", {})))
    method_config.update(
        sparsevideo.normalize_method_config(args.method, user_method_config)
    )
    if args.method == "flashomni":
        apply_flashomni_hunyuan_quality_defaults(spec, method_config, user_method_config)
    if args.method == "draft":
        apply_draft_runtime_layout_defaults(
            spec, height, width, num_frames, method_config, user_method_config,
        )
    if spec.pipeline_class == "HunyuanVideoImageToVideoPipeline" and args.method in ("svg1", "svg2"):
        if "context_length" not in user_method_config:
            method_config["context_length"] = None
        if "prompt_length" not in user_method_config:
            method_config["prompt_length"] = None
    if args.method == "radial" and not strict_kernels:
        method_config["allow_flex_fallback"] = True
    if args.method == "draft" and not strict_kernels:
        method_config["allow_triton_fallback"] = True
    model_id = resolve_model_id(spec, args.model_root, args.model_path)
    output_file = make_output_file(args, spec.key, args.method, num_frames)
    scheduler_flow_shift = resolve_scheduler_flow_shift(spec, args.height, args.flow_shift)
    wan_flow_shift = scheduler_flow_shift if spec.family == "wan" else None
    unsupported = not sparse_method_supported(spec, args.method)
    try:
        if not unsupported:
            materialize_method_config_values(args.method, method_config)
            if args.method == "spargeattn":
                normalize_spargeattn_model_out_path(method_config, output_file)
            if (
                args.method == "svoo"
                and method_config.get("use_dynamic_min_kc_ratio")
                and (
                    not method_config.get("sparsity_csv_path")
                    or method_config.get("sparsity_csv_path") == "sparsity_profiles/sparsity_results.csv"
                )
            ):
                method_config["sparsity_csv_path"] = default_svoo_sparsity_csv_path(spec)
            validate_method_config(args.method, method_config, model_family=spec.family)
    except (FileNotFoundError, NotImplementedError, TypeError, ValueError) as exc:
        failed_metrics = {
            "model": spec.key,
            "model_arg": args.model,
            "model_id": model_id,
            "method": args.method,
            "method_config": method_config,
            "profile": args.profile,
            "profile_method": profile_method,
            "profile_overrides": profile,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "requested_duration_seconds": args.duration_seconds,
            "num_inference_steps": steps,
            "dtype": args.dtype,
            "device": args.device,
            "cpu_offload": args.cpu_offload,
            "cpu_offload_mode": args.cpu_offload_mode,
            "vae_dtype": args.vae_dtype,
            "vae_tiling": args.vae_tiling,
            "vae_slicing": args.vae_slicing,
            "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
            "strict_kernels": strict_kernels,
            "allow_debug_fallbacks": args.allow_debug_fallbacks,
            "seed": args.seed,
            "output_file": str(output_file),
            "scheduler_flow_shift": scheduler_flow_shift,
            "wan_flow_shift": wan_flow_shift,
            "runtime": {"preflight": {"errors": [str(exc)], "warnings": []}},
            "status": "failed",
            "failed_stage": "validate_method_config",
            "timings": {},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not args.dry_run:
            append_metrics(args.metrics_file, failed_metrics)
        print_final_run_metrics(args, failed_metrics)
        return 1
    from sparsevideo._runtime import (
        adacluster_load_status,
        draft_block_sparse_load_status,
        flash_attn_load_status,
        flashomni_load_status,
        flashinfer_load_status,
        native_kernel_load_status,
        optional_kernel_status,
        radial_runtime_load_status,
        sageattention_load_status,
        spas_sage_attn_load_status,
        sta_load_status,
        svg1_runtime_load_status,
        svg2_runtime_load_status,
        svoo_runtime_load_status,
        torch_runtime_status,
    )

    runtime_status = {
        "optional_kernels": optional_kernel_status(),
        "torch": torch_runtime_status(),
    }
    defer_fused_native_kernel_load = should_defer_fused_native_kernel_load(
        spec, args.method, dry_run=args.dry_run,
    )
    if should_preload_fused_native_kernels(spec, args.method) and not defer_fused_native_kernel_load:
        runtime_status["optional_kernels"]["svg_svoo_fused_kernels"].update(
            native_kernel_load_status()
        )
    needs_flash_attn_load = (
        not unsupported
        and (
            args.method == "draft"
            or (args.method in ("svg1", "adacluster") and spec.family == "hunyuan_video")
        )
    )
    if needs_flash_attn_load:
        runtime_status["optional_kernels"].setdefault("flash_attn", {}).update(
            quiet_runtime_status_call(flash_attn_load_status)
        )
    if not unsupported and args.method == "adacluster":
        runtime_status["optional_kernels"].setdefault("adacluster_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(adacluster_load_status))
        runtime_status["optional_kernels"]["adacluster_kernels"]["load_checked"] = True
    if not unsupported and args.method == "draft":
        runtime_status["optional_kernels"].setdefault("draft_kernels", {}).setdefault(
            "mit_block_sparse_attn", {}
        ).update(quiet_runtime_status_call(draft_block_sparse_load_status))
        runtime_status["optional_kernels"]["draft_kernels"]["mit_load_checked"] = True
    needs_flashinfer_load = (
        not unsupported
        and (
            args.method in ("adacluster", "radial", "svg2")
            or args.method == "svoo"
        )
    )
    if needs_flashinfer_load:
        runtime_status["optional_kernels"].setdefault("flashinfer", {}).update(
            quiet_runtime_status_call(flashinfer_load_status)
        )
    if not unsupported and args.method == "radial":
        runtime_status["optional_kernels"].setdefault("radial_kernels", {}).setdefault(
            "owned_runtime", {}
        ).update(quiet_runtime_status_call(radial_runtime_load_status))
        runtime_status["optional_kernels"]["radial_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svg1":
        runtime_status["optional_kernels"].setdefault("svg1_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svg1_runtime_load_status))
        runtime_status["optional_kernels"]["svg1_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svg2":
        runtime_status["optional_kernels"].setdefault("svg2_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svg2_runtime_load_status))
        runtime_status["optional_kernels"]["svg2_kernels"]["load_checked"] = True
    if not unsupported and args.method == "svoo":
        runtime_status["optional_kernels"].setdefault("svoo_kernels", {}).setdefault(
            "owned_triton_runtime", {}
        ).update(quiet_runtime_status_call(svoo_runtime_load_status))
        runtime_status["optional_kernels"]["svoo_kernels"]["load_checked"] = True
    needs_flashomni_load = (
        not unsupported
        and args.method == "flashomni"
        and method_config.get("implementation") == "upstream"
    )
    if needs_flashomni_load:
        runtime_status["optional_kernels"].setdefault("flashomni", {}).update(
            quiet_runtime_status_call(flashomni_load_status)
        )
    if not unsupported and args.method == "sta":
        runtime_status["optional_kernels"].setdefault("sta_kernels", {}).update(
            quiet_runtime_status_call(sta_load_status)
        )
    spargeattn_needs_runtime = (
        not unsupported
        and args.method == "spargeattn"
    )
    radial_needs_sparge = (
        not unsupported and args.method == "radial" and method_config.get("use_sage_attention")
    )
    if spargeattn_needs_runtime or radial_needs_sparge:
        runtime_status["optional_kernels"].setdefault("spas_sage_attn", {}).update(
            quiet_runtime_status_call(
                spas_sage_attn_load_status,
                require_autotune=bool(
                    args.method == "spargeattn"
                    and (method_config.get("tune") or method_config.get("model_out_path"))
                ),
            )
        )
    radial_needs_sageattention = (
        radial_needs_sparge
        and (
            float(method_config.get("dense_warmup_step_ratio", 0) or 0) > 0
            or float(method_config.get("dense_warmup_layer_ratio", 0) or 0) > 0
        )
    )
    if radial_needs_sageattention:
        runtime_status["optional_kernels"].setdefault("sageattention", {}).update(
            quiet_runtime_status_call(sageattention_load_status)
        )
    if unsupported:
        runtime_status["preflight"] = {"errors": [], "warnings": []}
    else:
        runtime_status["preflight"] = preflight_runtime(
            args.method,
            method_config,
            args.device,
            runtime_status,
            strict_kernels=strict_kernels,
            model_family=spec.family,
        )
    if not unsupported and args.method == "draft":
        draft_error = draft_upstream_layout_error(
            spec, height, width, num_frames, method_config,
        )
        if draft_error is not None:
            runtime_status["preflight"]["errors"].append(draft_error)
    if not unsupported and args.method == "radial":
        radial_warning = radial_flashinfer_layout_warning(
            spec, height, width, num_frames, method_config,
        )
        if radial_warning is not None:
            if method_config.get("use_sage_attention") or strict_kernels:
                runtime_status["preflight"]["errors"].append(radial_warning)
            else:
                runtime_status["preflight"]["warnings"].append(radial_warning)
    if not unsupported and args.method == "sta":
        sta_messages = sta_layout_preflight_messages(
            spec, height, width, num_frames, method_config,
            strict_kernels=strict_kernels,
        )
        runtime_status["preflight"]["errors"].extend(sta_messages["errors"])
        runtime_status["preflight"]["warnings"].extend(sta_messages["warnings"])
    runtime_status["preflight"]["errors"].extend(
        model_shape_preflight_errors(spec, height, width)
    )
    runtime_status["preflight"]["warnings"].extend(
        model_quality_warnings(spec, height, width)
    )
    base_metrics: Dict[str, Any] = {
        "model": spec.key,
        "model_arg": args.model,
        "model_id": model_id,
        "method": args.method,
        "method_config": method_config,
        "profile": args.profile,
        "profile_method": profile_method,
        "profile_overrides": profile,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
        "requested_duration_seconds": args.duration_seconds,
        "num_inference_steps": steps,
        "dtype": args.dtype,
        "device": args.device,
        "cpu_offload": args.cpu_offload,
        "cpu_offload_mode": args.cpu_offload_mode,
        "vae_dtype": args.vae_dtype,
        "vae_tiling": args.vae_tiling,
        "vae_slicing": args.vae_slicing,
        "vae_decoder_chunk_size": args.vae_decoder_chunk_size,
        "skip_decode": args.skip_decode,
        "output_type": "latent" if args.skip_decode else spec.output_type,
        "compatibility_label": spec.compatibility_label,
        "unsupported_reason": spec.unsupported_reason,
        "strict_kernels": strict_kernels,
        "allow_debug_fallbacks": args.allow_debug_fallbacks,
        "seed": args.seed,
        "negative_prompt": args.negative_prompt,
        "output_file": None if args.skip_decode else str(output_file),
        "scheduler_flow_shift": scheduler_flow_shift,
        "wan_flow_shift": wan_flow_shift,
        "runtime": runtime_status,
        "source_fingerprints": sparsevideo_source_fingerprints(args.method),
    }

    if args.dry_run:
        if unsupported:
            base_metrics.update(status="unsupported_dry_run")
            base_metrics["error"] = unsupported_sparse_method_message(spec, args.method)
            print_final_run_metrics(args, base_metrics)
            return 0
        if runtime_status["preflight"]["errors"]:
            base_metrics.update(
                status="failed",
                failed_stage="preflight",
                timings={},
                error_type="RuntimeError",
                error="; ".join(runtime_status["preflight"]["errors"]),
            )
            print_final_run_metrics(args, base_metrics)
            return 1
        base_metrics.update(status="dry_run")
        print_final_run_metrics(args, base_metrics)
        return 0

    if unsupported:
        base_metrics.update(
            status="unsupported",
            error=unsupported_sparse_method_message(spec, args.method),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 2

    if runtime_status["preflight"]["errors"]:
        base_metrics.update(
            status="failed",
            failed_stage="preflight",
            timings={},
            error_type="RuntimeError",
            error="; ".join(runtime_status["preflight"]["errors"]),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 1

    stage = "start"
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    handle = None

    try:
        stage = "import"
        with redirect_stdout(sys.stderr):
            import torch
            from diffusers.utils import export_to_video
        configure_torch_compile_logging(verbose_compile_logs=args.verbose_compile_logs)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Check CUDA_VISIBLE_DEVICES, driver access, and whether this process is running on a GPU node."
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        seed_everything(torch, args.seed)
        try:
            torch.backends.cuda.preferred_linalg_library(backend="magma")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt = read_prompt(args)
        torch_dtype = parse_dtype(torch, args.dtype)
        vae_torch_dtype = parse_dtype(torch, args.vae_dtype) if args.vae_dtype is not None else None

        stage = "load_pipeline"
        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            pipe = load_pipeline(
                spec,
                model_id,
                torch_dtype,
                vae_torch_dtype,
                args.local_files_only,
                height=args.height,
                flow_shift=scheduler_flow_shift,
            )
            prepare_pipeline(
                pipe,
                args.device,
                args.cpu_offload,
                args.vae_tiling,
                args.vae_slicing,
                cpu_offload_mode=args.cpu_offload_mode,
                vae_decoder_chunk_size=args.vae_decoder_chunk_size,
            )
            sync_if_cuda(torch, args.device)
        timings["load_pipeline_sec"] = time.perf_counter() - t0

        if defer_fused_native_kernel_load:
            stage = "deferred_runtime_preflight"
            with redirect_stdout(sys.stderr):
                runtime_status["optional_kernels"]["svg_svoo_fused_kernels"].update(
                    native_kernel_load_status()
                )
            deferred_preflight = preflight_runtime(
                args.method,
                method_config,
                args.device,
                runtime_status,
                strict_kernels=strict_kernels,
                model_family=spec.family,
            )
            runtime_status["preflight"]["warnings"].extend(deferred_preflight["warnings"])
            if deferred_preflight["errors"]:
                runtime_status["preflight"]["errors"].extend(deferred_preflight["errors"])
                raise RuntimeError("; ".join(deferred_preflight["errors"]))

        stage = "apply_sparse_attention"
        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            if args.method in ("svg1", "svg2") and spec.family == "hunyuan_video":
                hunyuan_i2v = spec.pipeline_class == "HunyuanVideoImageToVideoPipeline"
                if method_config.get("context_length") is None and not hunyuan_i2v:
                    method_config["context_length"] = 256
                if method_config.get("prompt_length") is None and not hunyuan_i2v:
                    method_config["prompt_length"] = infer_hunyuan_prompt_length(
                        pipe, prompt, int(method_config["context_length"]),
                    )
            handle = sparsevideo.apply_sparse_attention(pipe, method=args.method, config=method_config)
            base_metrics["sparse_attention_handle"] = sparse_attention_handle_summary(handle)
            sync_if_cuda(torch, args.device)
        timings["apply_sparse_attention_sec"] = time.perf_counter() - t0

        if args.method == "svoo":
            stage = "svoo_kernel_warmup"
            t0 = time.perf_counter()
            with redirect_stdout(sys.stderr):
                from sparsevideo.methods.svoo.warmup import warmup_svoo_kernels_from_pipeline

                warmup_status = warmup_svoo_kernels_from_pipeline(
                    pipe,
                    model_type=spec.family,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    config=method_config,
                    dtype=torch_dtype,
                    device=args.device,
                )
                sync_if_cuda(torch, args.device)
            timings["svoo_kernel_warmup_sec"] = time.perf_counter() - t0
            base_metrics["svoo_kernel_warmup"] = warmup_status
            warmup_warning = validate_svoo_warmup_status(
                warmup_status, strict_kernels=strict_kernels,
            )
            if warmup_warning is not None:
                runtime_status["preflight"]["warnings"].append(warmup_warning)

        stage = "generate"
        if not args.skip_decode:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        if not args.skip_decode and args.skip_existing and output_file.exists():
            base_metrics.update(status="skipped_existing", timings=timings)
            with redirect_stdout(sys.stderr):
                handle.restore()
            base_metrics["sparse_attention_handle_after_restore"] = sparse_attention_handle_summary(handle)
            handle = None
            append_metrics(args.metrics_file, base_metrics)
            print_final_run_metrics(args, base_metrics)
            return 0

        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        call_kwargs = build_call_kwargs(
            args=args,
            spec=spec,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            num_frames=num_frames,
            fps=fps,
        )
        if spec.pipeline_class == "HunyuanVideoImageToVideoPipeline":
            base_metrics["hunyuan_i2v_prompt_template_compat"] = apply_hunyuan_i2v_prompt_template_compat(
                pipe, call_kwargs,
            )

        t0 = time.perf_counter()
        with redirect_stdout(sys.stderr):
            result = call_pipeline_with_model_compat(pipe, call_kwargs, torch, spec, args.device)
            sync_if_cuda(torch, args.device)
        base_metrics["sparse_attention_handle"] = sparse_attention_handle_summary(handle)
        timings["generate_sec"] = time.perf_counter() - t0

        stage = "spargeattn_save_state"
        t0 = time.perf_counter()
        tuned_state_path = None
        if args.method == "spargeattn":
            with redirect_stdout(sys.stderr):
                tuned_state_path = maybe_save_spargeattn_tuned_state(handle, method_config)
        timings["spargeattn_save_state_sec"] = time.perf_counter() - t0
        if tuned_state_path is not None:
            base_metrics["spargeattn_tuned_state_path"] = tuned_state_path

        if args.skip_decode:
            stage = "summarize_latent_output"
            base_metrics["latent_output"] = pipeline_output_summary(torch, getattr(result, "frames", None))
            timings["export_video_sec"] = 0.0
        else:
            stage = "export_video"
            t0 = time.perf_counter()
            with redirect_stdout(sys.stderr):
                export_to_video(result.frames[0], str(output_file), fps=fps)
            timings["export_video_sec"] = time.perf_counter() - t0
        with redirect_stdout(sys.stderr):
            handle.restore()
        base_metrics["sparse_attention_handle_after_restore"] = sparse_attention_handle_summary(handle)
        handle = None

        timings["total_sec"] = time.perf_counter() - t_total
        base_metrics.update(
            status="ok",
            timings=timings,
            seconds_per_frame=timings["generate_sec"] / max(num_frames, 1),
            **cuda_memory_gb(torch),
        )
        append_metrics(args.metrics_file, base_metrics)
        print_final_run_metrics(args, base_metrics)
        return 0
    finally:
        if handle is not None:
            with redirect_stdout(sys.stderr):
                handle.restore()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
