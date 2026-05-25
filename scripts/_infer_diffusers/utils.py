from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_torch_tensor(value: Any) -> bool:
    try:
        import torch
        return torch.is_tensor(value)
    except Exception:
        return False


def json_ready(value: Any) -> Any:
    if is_torch_tensor(value):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def cuda_memory_gb(torch_module) -> Dict[str, float]:
    if not torch_module.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_gb": torch_module.cuda.max_memory_allocated() / (1024**3),
        "cuda_peak_reserved_gb": torch_module.cuda.max_memory_reserved() / (1024**3),
    }


def sync_if_cuda(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def quiet_runtime_status_call(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def configure_torch_compile_logging(*, verbose_compile_logs: bool) -> None:
    if verbose_compile_logs:
        return
    import torch._inductor.config as inductor_config
    import torch._inductor.select_algorithm as select_algorithm

    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)
    logging.getLogger("torch._inductor.runtime.triton_heuristics").setLevel(logging.CRITICAL)
    inductor_config.autotune_num_choices_displayed = 0
    select_algorithm.PRINT_AUTOTUNE = False


def pipeline_output_summary(torch, value: Any) -> Dict[str, Any]:
    if torch.is_tensor(value):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        summary: Dict[str, Any] = {"type": type(value).__name__, "length": len(value)}
        if value:
            summary["first"] = pipeline_output_summary(torch, value[0])
        return summary
    return {"type": type(value).__name__}


def sparse_attention_handle_summary(handle) -> Dict[str, Any]:
    summary = getattr(handle, "summary", None)
    if callable(summary):
        return summary()
    return {"type": type(handle).__name__, "summary_available": False}


def append_metrics(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def print_metrics_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def terminal_one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def append_terminal_message(messages: List[str], value: Any) -> None:
    text = terminal_one_line(value)
    if text and text not in messages:
        messages.append(text)


def collect_runtime_messages(payload: Dict[str, Any], key: str) -> List[str]:
    runtime = payload.get("runtime") or {}
    messages: List[str] = []
    for section_name in ("preflight", "generation_checks"):
        section = runtime.get(section_name) or {}
        for message in section.get(key) or []:
            append_terminal_message(messages, message)
    if key == "errors" and not messages:
        append_terminal_message(messages, payload.get("error"))
    return messages


def print_run_summary(args, payload: Dict[str, Any]) -> None:
    timings = payload.get("timings") or {}
    output_file = payload.get("output_file")
    if (
        payload.get("status") in {"ok", "skipped_existing"}
        and output_file
        and not getattr(args, "dry_run", False)
    ):
        print(output_file)
        return
    lines = [
        f"status={payload.get('status')}",
        f"model={payload.get('model')}",
        f"method={payload.get('method')}",
    ]
    if payload.get("failed_stage"):
        lines.append(f"failed_stage={payload['failed_stage']}")
    if payload.get("error_type"):
        lines.append(f"error_type={payload['error_type']}")
    lines.append(f"output_file={output_file if output_file else '<skip-decode>'}")
    if getattr(args, "dry_run", False):
        lines.append("metrics_file=<not written in dry-run>")
    elif args.metrics_file is None:
        lines.append("metrics_file=<disabled>")
    else:
        lines.append(f"metrics_file={args.metrics_file}")
    if "generate_sec" in timings:
        lines.append(f"generate_sec={timings['generate_sec']:.3f}")
    if "total_sec" in timings:
        lines.append(f"total_sec={timings['total_sec']:.3f}")
    if "seconds_per_frame" in payload:
        lines.append(f"seconds_per_frame={payload['seconds_per_frame']:.3f}")
    if "cuda_peak_allocated_gb" in payload:
        lines.append(f"cuda_peak_allocated_gb={payload['cuda_peak_allocated_gb']:.3f}")
    if "cuda_peak_reserved_gb" in payload:
        lines.append(f"cuda_peak_reserved_gb={payload['cuda_peak_reserved_gb']:.3f}")
    errors = collect_runtime_messages(payload, "errors")
    warnings = collect_runtime_messages(payload, "warnings")
    for index, error in enumerate(errors[:4], start=1):
        lines.append(f"error[{index}]={error}")
    if len(errors) > 4:
        lines.append(f"error_more={len(errors) - 4}")
    for index, warning in enumerate(warnings[:3], start=1):
        lines.append(f"warning[{index}]={warning}")
    if len(warnings) > 3:
        lines.append(f"warning_more={len(warnings) - 3}")
    lines.append("details=use --print-json for the full metrics/config/runtime payload")
    print("\n".join(lines))


def print_final_run_metrics(args, payload: Dict[str, Any]) -> None:
    if getattr(args, "print_json", False):
        print_metrics_json(payload)
        return
    print_run_summary(args, payload)


def make_output_file(args, model: str, method: str, num_frames: int) -> Path:
    if args.output_file is not None:
        return args.output_file
    filename = f"seed{args.seed}_{args.height}x{args.width}_{num_frames}f.mp4"
    return args.output_dir / model / method / filename


def sparsevideo_source_fingerprints(method: str) -> Dict[str, Any]:
    if method != "flashomni":
        return {}
    import hashlib
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    paths = {
        "flashomni_policy_sha256": repo_root / "src" / "sparsevideo" / "methods" / "flashomni" / "policy.py",
        "flashomni_method_sha256": repo_root / "src" / "sparsevideo" / "methods" / "flashomni" / "method.py",
    }
    fingerprints: Dict[str, Any] = {}
    for key, path in paths.items():
        try:
            fingerprints[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            fingerprints[key] = f"unavailable:{type(exc).__name__}"
    return fingerprints


def validate_svoo_warmup_status(status: Dict[str, Any], *, strict_kernels: bool) -> Optional[str]:
    message = None
    if not status.get("enabled"):
        message = "SVOO kernel warmup is disabled; strict benchmark runs must precompile the owned kernel path."
    elif status.get("error"):
        message = f"SVOO kernel warmup failed: {status['error']}"
    elif not status.get("ran"):
        reason = status.get("reason") or "unknown"
        message = f"SVOO kernel warmup did not run: {reason}"
    if message is None:
        return None
    if strict_kernels:
        raise RuntimeError(message)
    return message


def maybe_save_spargeattn_tuned_state(handle, method_config: Dict[str, Any]) -> Optional[str]:
    if not method_config.get("tune"):
        return None
    model_out_path = method_config.get("model_out_path")
    if not model_out_path:
        return None
    method_instance = getattr(handle, "_method_instance", None)
    export_state_dict = getattr(method_instance, "export_state_dict", None)
    if not callable(export_state_dict):
        raise RuntimeError("spargeattn tune=true did not install an exportable tuned-state method")

    import torch
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    state = export_state_dict()
    if not state:
        raise RuntimeError(
            "spargeattn tune=true produced no tuned state. Check that the sparse path ran "
            "with CUDA, sequence length >=128, supported head_dim, and no attention_mask."
        )
    path = _Path(str(model_out_path)).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path)


def sparse_method_supported(spec, method: str) -> bool:
    if method == "dense":
        return True
    if not spec.sparse_supported:
        return False
    if spec.sparse_methods is None:
        return True
    return method in spec.sparse_methods


def unsupported_sparse_method_message(spec, method: str) -> str:
    from sparsevideo._support import unvalidated_method_reason

    method_reason = ""
    if spec.sparse_supported and spec.sparse_methods is not None and method not in spec.sparse_methods:
        method_reason = unvalidated_method_reason(method, smoke_methods=spec.sparse_methods)
    reasons = [reason for reason in (spec.unsupported_reason, method_reason) if reason]
    reason = f" {' '.join(reasons)}" if reasons else ""
    return (
        f"{method} is not implemented for {spec.family}; "
        f"compatibility_label={spec.compatibility_label}; "
        f"supported sparse methods: {list(spec.sparse_methods or ()) or 'none'}.{reason}"
    )
