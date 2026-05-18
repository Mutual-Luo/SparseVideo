from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config
from .policy import flashomni_hunyuan_sparse_blocks, flashomni_paper_sparse_blocks


_FLASHOMNI_MODULE_CACHE: ModuleType | None = None
_FLASHOMNI_Q_BLOCK_CACHE_CPU_THRESHOLD_BYTES = 128 * 1024 * 1024
_FLASHOMNI_Q_BLOCK_CACHE_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass
class _FlashOmniPaperMMDiTResult:
    output: torch.Tensor
    dispatch: str
    backend: str


@dataclass
class _FlashOmniPrefixMaskTrim:
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: torch.Tensor | None
    kv_text_len: int


@dataclass
class _FlashOmniCachedQBlocks:
    values: torch.Tensor
    indices: torch.Tensor
    output_shape: tuple[int, int, int, int]
    q_block_size: int
    num_q_blocks: int


@dataclass
class _FlashOmniPaperMMDiTSchedule:
    full: bool
    compute_symbols: bool
    current_iter: int


class _FlashOmniPaperMMDiTState:
    def __init__(self):
        self.sparse_q: torch.Tensor | None = None
        self.sparse_kv: torch.Tensor | None = None
        self.output_history: list[_FlashOmniCachedQBlocks] = []
        self.gemm_o_bias_history: list[torch.Tensor] = []
        self.last_dispatch: str | None = None

    def has_symbols(self) -> bool:
        return self.sparse_q is not None and self.sparse_kv is not None

    def should_update(self, step: int | None, interval: int) -> bool:
        if not self.has_symbols() or not self.output_history:
            return True
        if step is None or int(step) <= 0:
            return False
        return (int(step) - 1) % max(1, int(interval)) == 0

    def update_symbols(self, sparse_q: torch.Tensor, sparse_kv: torch.Tensor) -> None:
        self.sparse_q = sparse_q.detach()
        self.sparse_kv = sparse_kv.detach()

    def append_output(self, output: torch.Tensor, sparse_q: torch.Tensor, q_block_size: int, order: int) -> None:
        self.output_history.append(_flashomni_cache_q_blocks(output, sparse_q, q_block_size))
        keep = max(1, min(3, int(order) + 1))
        if len(self.output_history) > keep:
            self.output_history = self.output_history[-keep:]

    def predicted_output(self, order: int) -> _FlashOmniCachedQBlocks | None:
        if not self.output_history:
            return None
        order = int(order)
        if order <= 0 or len(self.output_history) < 2:
            return self.output_history[-1]
        if order == 1 or len(self.output_history) < 3:
            return _flashomni_predict_cached_q_blocks(self.output_history[-1], self.output_history[-2])
        return _flashomni_predict_cached_q_blocks(
            self.output_history[-1],
            self.output_history[-2],
            self.output_history[-3],
        )

    def append_gemm_o_bias(self, bias: torch.Tensor, order: int) -> None:
        self.gemm_o_bias_history.append(bias.detach().cpu())
        keep = max(1, min(3, int(order) + 1))
        if len(self.gemm_o_bias_history) > keep:
            self.gemm_o_bias_history = self.gemm_o_bias_history[-keep:]

    def predicted_gemm_o_bias(self, order: int) -> torch.Tensor | None:
        if not self.gemm_o_bias_history:
            return None
        order = int(order)
        if order <= 0 or len(self.gemm_o_bias_history) < 2:
            return self.gemm_o_bias_history[-1]
        if order == 1 or len(self.gemm_o_bias_history) < 3:
            return self.gemm_o_bias_history[-1] + (self.gemm_o_bias_history[-1] - self.gemm_o_bias_history[-2])
        return (
            self.gemm_o_bias_history[-1]
            + (self.gemm_o_bias_history[-1] - self.gemm_o_bias_history[-2])
            + 0.5 * (
                self.gemm_o_bias_history[-1]
                - 2 * self.gemm_o_bias_history[-2]
                + self.gemm_o_bias_history[-3]
            )
        )


class FlashOmniMethod(SparseMethod):
    """FlashOmni sparse kernel adapter.

    Upstream FlashOmni is a sparse kernel interface. SparseVideo supports the
    explicit sparse-info tensor path for video parity and keeps the upstream
    benchmark's global-random sparse-info generator for kernel smoke/speed
    checks only. The old block-level q/k top-k generator is available only as
    sparse_pattern="local_qk_topk", and is not upstream video-method parity.

    Adapter for: training_free/FlashOmni/flashomni/attention.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        normalized_input = self.normalize_config(config)
        super().__init__(config, model_info)
        _sync_flashomni_upstream_config_aliases(self.config, normalized_input)
        if self.config["sparse_pattern"] == "global_random":
            self.config["sparse_block_size_for_q"] = int(self.config["sparse_size"])
            self.config["sparse_block_size_for_kv"] = int(self.config["sparse_size"])
        if self.config["implementation"] not in ("upstream", "flex"):
            raise ValueError("flashomni implementation must be 'upstream' or 'flex'")
        if self.config["backend"] not in ("auto", "fa2", "fa3"):
            raise ValueError("flashomni backend must be 'auto', 'fa2', or 'fa3'")
        if self.config["sparse_pattern"] not in ("explicit", "global_random", "paper_mmdit", "local_qk_topk"):
            raise ValueError(
                "flashomni sparse_pattern must be 'explicit', 'global_random', "
                "'paper_mmdit', or 'local_qk_topk'"
            )
        if self.config["pos_encoding_mode"] not in ("NONE", "ROPE_LLAMA", "ALIBI"):
            raise ValueError("flashomni pos_encoding_mode must be 'NONE', 'ROPE_LLAMA', or 'ALIBI'")
        _validate_ratio("flashomni threshold_q", self.config["threshold_q"])
        _validate_ratio("flashomni threshold_kv", self.config["threshold_kv"])
        _validate_ratio("flashomni saving_threshold_q_for_taylor", self.config["saving_threshold_q_for_taylor"])
        if int(self.config["fresh_threshold"]) < 1:
            raise ValueError("flashomni fresh_threshold must be >= 1")
        if int(self.config["max_order"]) not in (0, 1, 2):
            raise ValueError("flashomni max_order currently supports 0, 1, or 2")
        if int(self.config["first_enhance"]) < 0:
            raise ValueError("flashomni first_enhance must be >= 0")
        if int(self.config["num_inference_steps"]) < 1:
            raise ValueError("flashomni num_inference_steps must be >= 1")
        if self.config["taylor_cache_device"] not in ("cuda", "cpu"):
            raise ValueError("flashomni taylor_cache_device must be 'cuda' or 'cpu'")
        if int(self.config["debug_memory_max_events"]) < 1:
            raise ValueError("flashomni debug_memory_max_events must be >= 1")
        if self.config["is_full"] and self.config["implementation"] != "upstream":
            raise NotImplementedError(
                "flashomni is_full follows the upstream FlashOmni full-kernel path "
                "and requires implementation='upstream'."
            )
        if self.config["implementation"] == "flex" and self.config["sparse_pattern"] != "local_qk_topk":
            raise NotImplementedError(
                "flashomni implementation='flex' is only a slow local_qk_topk diagnostic path; "
                "the upstream sparse-info path requires implementation='upstream'."
            )
        if not self.config["is_full"] and self.config["sparse_pattern"] == "explicit":
            _validate_explicit_sparse_info(self.config)
        if self.config["sparse_pattern"] == "local_qk_topk":
            import warnings

            warnings.warn(
                "flashomni sparse_pattern='local_qk_topk' is a SparseVideo diagnostic "
                "pattern, not upstream FlashOmni video-method parity.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.config["sparse_pattern"] == "paper_mmdit":
            import warnings

            warnings.warn(
                "flashomni sparse_pattern='paper_mmdit' uses SparseVideo-owned FlashOmni "
                "attention sparse-info code. Hunyuan runs use the public anonymous "
                "FlashOmni Hunyuan sparse-symbol policy and upstream default names "
                "threshold_q, threshold_kv, fresh_threshold, max_order, first_enhance, "
                "and saving_threshold_q_for_taylor, with the owned Hunyuan "
                "forward/Taylor-cache patch installed through apply_sparse_attention.",
                RuntimeWarning,
                stacklevel=2,
            )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"flashomni not yet supported for {self.model_info.model_type}")

        cfg = self.config
        paper_state = _FlashOmniPaperMMDiTState()
        plan_kwargs = {
            "causal": cfg["causal"],
            "pos_encoding_mode": cfg["pos_encoding_mode"],
            "use_fp16_qk_reduction": cfg["use_fp16_qk_reduction"],
            "logits_soft_cap": cfg["logits_soft_cap"],
            "sm_scale": cfg["sm_scale"],
            "rope_scale": cfg["rope_scale"],
            "rope_theta": cfg["rope_theta"],
        }

        def query_projection_fn(linear, hidden_states, num_heads):
            if not _flashomni_should_use_sparse_gemm(cfg, paper_state, step_tracker):
                return linear(hidden_states)
            out = _flashomni_sparse_q_gemm(
                linear,
                hidden_states,
                sparse_q=paper_state.sparse_q,
                num_heads=num_heads,
                sparse_q_size=cfg["sparse_block_size_for_q"],
            )
            self.record_runtime_kernel(
                "flashomni_sparse_q_gemm",
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        def output_projection_fn(linear, hidden_states, num_heads):
            if not _flashomni_can_use_sparse_o_gemm(cfg, paper_state, hidden_states):
                return linear(hidden_states)
            if paper_state.last_dispatch == "dense":
                cache_bias = _flashomni_sparse_o_gemm_cache_bias(
                    linear,
                    hidden_states,
                    sparse_q=paper_state.sparse_q,
                    num_heads=num_heads,
                    sparse_q_size=cfg["sparse_block_size_for_q"],
                )
                self.record_runtime_kernel(
                    "flashomni_sparse_o_gemm_cache_bias",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                paper_state.append_gemm_o_bias(cache_bias, int(cfg["D"]))
                out = _flashomni_sparse_o_gemm(
                    linear,
                    hidden_states,
                    sparse_q=paper_state.sparse_q,
                    num_heads=num_heads,
                    sparse_q_size=cfg["sparse_block_size_for_q"],
                    cache_bias=cache_bias,
                )
                self.record_runtime_kernel(
                    "flashomni_sparse_o_gemm",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if paper_state.last_dispatch == "sparse":
                cache_bias = paper_state.predicted_gemm_o_bias(int(cfg["D"]))
                if cache_bias is None:
                    return linear(hidden_states)
                out = _flashomni_sparse_o_gemm(
                    linear,
                    hidden_states,
                    sparse_q=paper_state.sparse_q,
                    num_heads=num_heads,
                    sparse_q_size=cfg["sparse_block_size_for_q"],
                    cache_bias=cache_bias,
                )
                self.record_runtime_kernel(
                    "flashomni_sparse_o_gemm",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            return linear(hidden_states)

        def attn_fn(query, key, value, attention_mask, **kwargs):
            if cfg["is_full"]:
                out = _flashomni_explicit_attention(
                    query, key, value,
                    sparse_info=cfg["sparse_info"],
                    sparse_kv_info=cfg["sparse_kv_info"],
                    sparse_info_indptr=cfg["sparse_info_indptr"],
                    sparse_kv_info_indptr=cfg["sparse_kv_info_indptr"],
                    sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                    sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                    implementation=cfg["implementation"],
                    backend=cfg["backend"],
                    workspace_bytes=cfg["workspace_bytes"],
                    is_full=True,
                    attention_mask=attention_mask,
                    text_len=kwargs.get("text_len", 0),
                    **plan_kwargs,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend="flashomni_full_upstream",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if cfg["sparse_pattern"] == "explicit":
                out = _flashomni_explicit_attention(
                    query, key, value,
                    sparse_info=cfg["sparse_info"],
                    sparse_kv_info=cfg["sparse_kv_info"],
                    sparse_info_indptr=cfg["sparse_info_indptr"],
                    sparse_kv_info_indptr=cfg["sparse_kv_info_indptr"],
                    sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                    sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                    implementation=cfg["implementation"],
                    backend=cfg["backend"],
                    workspace_bytes=cfg["workspace_bytes"],
                    is_full=False,
                    attention_mask=attention_mask,
                    text_len=kwargs.get("text_len", 0),
                    **plan_kwargs,
                )
                self.record_runtime_dispatch(
                    "sparse",
                    backend="flashomni_explicit_upstream",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if cfg["sparse_pattern"] == "global_random":
                out = _flashomni_global_random_attention(
                    query, key, value,
                    spq_Q=cfg["spq_Q"],
                    spq_KV=cfg["spq_KV"],
                    sparse_size=cfg["sparse_size"],
                    text_token=cfg["text_token"],
                    sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                    sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                    implementation=cfg["implementation"],
                    backend=cfg["backend"],
                    workspace_bytes=cfg["workspace_bytes"],
                    attention_mask=attention_mask,
                    text_len=kwargs.get("text_len", 0),
                    **plan_kwargs,
                )
                self.record_runtime_dispatch(
                    "sparse",
                    backend="flashomni_global_random_upstream",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if cfg["sparse_pattern"] == "paper_mmdit":
                paper_text_len = kwargs.get("text_len", 0)
                if not paper_text_len and self.model_info.model_type == "hunyuan_video":
                    paper_text_len = int(cfg["text_token"])
                result = _flashomni_paper_mmdit_attention(
                    query, key, value,
                    tau_q=cfg["threshold_q"],
                    tau_kv=cfg["threshold_kv"],
                    N=cfg["fresh_threshold"],
                    D=cfg["max_order"],
                    S_q=cfg["saving_threshold_q_for_taylor"],
                    text_len=paper_text_len,
                    sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                    sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                    implementation=cfg["implementation"],
                    backend=cfg["backend"],
                    workspace_bytes=cfg["workspace_bytes"],
                    attention_mask=attention_mask,
                    state=paper_state,
                    step=getattr(step_tracker, "step", None),
                    cache_dic=kwargs.get("cache_dic"),
                    current=kwargs.get("current"),
                    first_enhance=cfg["first_enhance"],
                    max_sequence_length=cfg["max_sequence_length"],
                    num_inference_steps=cfg["num_inference_steps"],
                    simthreshd1=cfg["simthreshd1"],
                    model_type=self.model_info.model_type,
                    **plan_kwargs,
                )
                self.record_runtime_dispatch(
                    result.dispatch,
                    backend=result.backend,
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return result.output
            if attention_mask is not None:
                raise NotImplementedError(
                    "flashomni sparse_pattern='local_qk_topk' is a SparseVideo diagnostic path "
                    "and does not implement upstream custom_mask handling. Use sparse_pattern='explicit'."
                )
            out = _flashomni_attention(
                query, key, value,
                sparse_kv_budget=cfg["sparse_kv_budget"],
                sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                implementation=cfg["implementation"],
                backend=cfg["backend"],
                workspace_bytes=cfg["workspace_bytes"],
                **plan_kwargs,
            )
            self.record_runtime_dispatch(
                "sparse",
                backend=_flashomni_local_backend_name(cfg["implementation"]),
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
                query_projection_fn=query_projection_fn,
                output_projection_fn=output_projection_fn,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            query_projection_fn=query_projection_fn,
            output_projection_fn=output_projection_fn,
        )

    def install_model_patches(self, model_info):
        if (
            model_info.model_type == "hunyuan_video"
            and self.config.get("sparse_pattern") == "paper_mmdit"
        ):
            from .hunyuan_forward import install_flashomni_hunyuan_forward_patch

            return [install_flashomni_hunyuan_forward_patch(model_info, self.config, self._ensure_runtime_stats())]
        return super().install_model_patches(model_info)

    def record_runtime_kernel(
        self,
        kernel: str,
        *,
        layer_idx: int | None = None,
        step: int | None = None,
    ) -> None:
        stats = self._ensure_runtime_stats()
        kernel_counts = stats.setdefault("kernel_counts", {})
        kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1
        event = {"kernel": kernel, "layer_idx": layer_idx, "step": step}
        stats["last_kernel"] = {key: value for key, value in event.items() if value is not None}


def _flashomni_local_backend_name(implementation):
    if implementation == "upstream":
        return "flashomni_local_qk_topk_upstream"
    return "flex_debug_fallback"


def _flashomni_should_use_sparse_gemm(config, state, step_tracker) -> bool:
    if not bool(config.get("use_sparse_gemm", False)):
        return False
    if config.get("sparse_pattern") != "paper_mmdit" or config.get("implementation") != "upstream":
        return False
    if int(config.get("sparse_block_size_for_q", 128)) != 128:
        return False
    if state.sparse_q is None:
        return False
    return not _flashomni_paper_mmdit_schedule(
        getattr(step_tracker, "step", None),
        fresh_threshold=int(config["fresh_threshold"]),
        first_enhance=int(config["first_enhance"]),
        num_inference_steps=int(config["num_inference_steps"]),
        has_symbols=state.has_symbols(),
    ).full


def _flashomni_can_use_sparse_o_gemm(config, state, hidden_states) -> bool:
    if not bool(config.get("use_sparse_gemm", False)):
        return False
    if config.get("sparse_pattern") != "paper_mmdit" or config.get("implementation") != "upstream":
        return False
    if int(config.get("sparse_block_size_for_q", 128)) != 128:
        return False
    if state.sparse_q is None or state.last_dispatch not in ("dense", "sparse"):
        return False
    return hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16)


def _flashomni_paper_mmdit_schedule(
    step,
    *,
    fresh_threshold: int,
    first_enhance: int,
    num_inference_steps: int,
    has_symbols: bool,
) -> _FlashOmniPaperMMDiTSchedule:
    if step is None:
        return _FlashOmniPaperMMDiTSchedule(
            full=not has_symbols,
            compute_symbols=not has_symbols,
            current_iter=0,
        )

    step0 = max(0, int(step) - 1)
    fresh_threshold = max(1, int(fresh_threshold))
    first_enhance = max(0, int(first_enhance))
    num_inference_steps = max(1, int(num_inference_steps))
    last_step = step0 == num_inference_steps - 1

    if not has_symbols and (first_enhance == 0 or step0 >= first_enhance - 1):
        return _FlashOmniPaperMMDiTSchedule(full=True, compute_symbols=True, current_iter=step0)

    if step0 < first_enhance:
        compute_symbols = step0 == first_enhance - 1
        return _FlashOmniPaperMMDiTSchedule(
            full=True,
            compute_symbols=compute_symbols,
            current_iter=step0,
        )

    cycle_pos = (step0 - first_enhance) % fresh_threshold
    full = last_step or cycle_pos == fresh_threshold - 1
    return _FlashOmniPaperMMDiTSchedule(
        full=full,
        compute_symbols=full,
        current_iter=step0,
    )


def _flashomni_sparse_q_gemm(linear, hidden_states, sparse_q, num_heads, sparse_q_size):
    if sparse_q is None:
        return linear(hidden_states)
    if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return linear(hidden_states)
    if not _flashomni_sparse_gemm_linear_shape_supported(linear, hidden_states):
        return linear(hidden_states)

    flashomni = _flashomni_import()
    padded_hidden_states, original_len = _flashomni_pad_gemm_sequence(hidden_states, int(sparse_q_size))
    packed_sparse_info, sparse_info_indptr = _flashomni_pack_sparse_q_info(
        flashomni,
        sparse_q,
        q_len=padded_hidden_states.shape[1],
        sparse_q_size=int(sparse_q_size),
        device=hidden_states.device,
    )
    out = torch.zeros(
        padded_hidden_states.shape[0],
        padded_hidden_states.shape[1],
        linear.weight.shape[0],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    out = flashomni.flashomni_gemm(
        padded_hidden_states.contiguous(),
        linear.weight.contiguous(),
        int(num_heads),
        packed_sparse_info,
        sparse_info_indptr,
        0,
        linear.bias,
        out=out,
        sparse_q_size=int(sparse_q_size),
        is_full=False,
    )
    return out[:, :original_len]


def _flashomni_sparse_o_gemm_cache_bias(linear, hidden_states, sparse_q, num_heads, sparse_q_size):
    if not _flashomni_sparse_gemm_linear_shape_supported(linear, hidden_states):
        return linear(hidden_states)
    flashomni = _flashomni_import()
    padded_hidden_states, original_len = _flashomni_pad_gemm_sequence(hidden_states, int(sparse_q_size))
    packed_sparse_info, sparse_info_indptr = _flashomni_pack_sparse_q_info(
        flashomni,
        sparse_q,
        q_len=padded_hidden_states.shape[1],
        sparse_q_size=int(sparse_q_size),
        device=hidden_states.device,
    )
    cache_bias = flashomni.flashomni_gemm_reduction(
        padded_hidden_states.contiguous(),
        linear.weight.contiguous(),
        int(num_heads),
        packed_sparse_info,
        sparse_info_indptr,
        0,
        linear.bias,
        is_for_cache=True,
        sparse_q_size=int(sparse_q_size),
    )
    return cache_bias[:, :original_len]


def _flashomni_sparse_o_gemm(linear, hidden_states, sparse_q, num_heads, sparse_q_size, cache_bias):
    if not _flashomni_sparse_gemm_linear_shape_supported(linear, hidden_states):
        return linear(hidden_states)
    flashomni = _flashomni_import()
    padded_hidden_states, original_len = _flashomni_pad_gemm_sequence(hidden_states, int(sparse_q_size))
    padded_cache_bias, _ = _flashomni_pad_gemm_sequence(
        cache_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        int(sparse_q_size),
    )
    if padded_cache_bias.shape[1] != padded_hidden_states.shape[1]:
        pad_len = padded_hidden_states.shape[1] - padded_cache_bias.shape[1]
        if pad_len < 0:
            padded_cache_bias = padded_cache_bias[:, : padded_hidden_states.shape[1]]
        else:
            padded_cache_bias = F.pad(padded_cache_bias, (0, 0, 0, pad_len))
    packed_sparse_info, sparse_info_indptr = _flashomni_pack_sparse_q_info(
        flashomni,
        sparse_q,
        q_len=padded_hidden_states.shape[1],
        sparse_q_size=int(sparse_q_size),
        device=hidden_states.device,
    )
    out = flashomni.flashomni_gemm_reduction(
        padded_hidden_states.contiguous(),
        linear.weight.contiguous(),
        int(num_heads),
        packed_sparse_info,
        sparse_info_indptr,
        0,
        padded_cache_bias.contiguous(),
        is_for_cache=False,
        sparse_q_size=int(sparse_q_size),
    )
    return out[:, :original_len]


def _flashomni_sparse_gemm_linear_shape_supported(linear, hidden_states) -> bool:
    out_features, in_features = linear.weight.shape
    return int(out_features) % 128 == 0 and int(in_features) % 32 == 0 and hidden_states.shape[-1] == in_features


def _flashomni_pad_gemm_sequence(hidden_states, sparse_q_size):
    original_len = hidden_states.shape[1]
    alignment = math.lcm(128, int(sparse_q_size))
    padded_len = math.ceil(original_len / alignment) * alignment
    if padded_len == original_len:
        return hidden_states, original_len
    return F.pad(hidden_states, (0, 0, 0, padded_len - original_len)), original_len


def _flashomni_pack_sparse_q_info(flashomni, sparse_q, q_len, sparse_q_size, device):
    if sparse_q is None or sparse_q.ndim != 3:
        raise RuntimeError("flashomni GEMM sparse_q must have shape [B, H, q_blocks]")
    sparse_q = sparse_q.to(device=device, dtype=torch.uint8).contiguous()
    batch_size, num_heads, q_blocks = sparse_q.shape
    expected_blocks = math.ceil(int(q_len) / int(sparse_q_size))
    if q_blocks < expected_blocks:
        sparse_q = F.pad(sparse_q, (0, expected_blocks - q_blocks), value=1)
        q_blocks = expected_blocks

    sparse_info = sparse_q.transpose(1, 2).contiguous().view(batch_size, q_blocks, num_heads)
    sparse_info_indptr = torch.arange(
        batch_size + 1,
        device=device,
        dtype=torch.int32,
    ) * (q_blocks * num_heads)
    return flashomni.segment_packbits(
        sparse_info.contiguous().view(-1),
        sparse_info_indptr,
        bitorder="little",
    )


def _validate_explicit_sparse_info(config):
    missing = [key for key in method_config.SPARSE_INFO_KEYS if config.get(key) is None]
    if missing:
        raise NotImplementedError(
            "flashomni sparse_pattern='explicit' follows upstream FlashOmni and requires "
            "precomputed sparse_info, sparse_kv_info, sparse_info_indptr, and "
            f"sparse_kv_info_indptr tensors. Missing: {missing}. "
            "Pass tensors through the Python API or pass torch-saved tensor paths "
            "to scripts/infer.py with the same config keys."
        )
    bad = [key for key in method_config.SPARSE_INFO_KEYS if not torch.is_tensor(config.get(key))]
    if bad:
        raise TypeError(f"flashomni explicit sparse-info inputs must be torch.Tensor values: {bad}")


def _validate_ratio(name, value):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _sync_flashomni_upstream_config_aliases(config, normalized_input):
    pairs = (
        ("threshold_q", "tau_q"),
        ("threshold_kv", "tau_kv"),
        ("fresh_threshold", "N"),
        ("max_order", "D"),
        ("saving_threshold_q_for_taylor", "S_q"),
    )
    for primary, legacy in pairs:
        primary_set = primary in normalized_input
        legacy_set = legacy in normalized_input
        if primary_set and legacy_set and normalized_input[primary] != normalized_input[legacy]:
            default = method_config.CONFIG_DEFAULTS[primary]
            if normalized_input[legacy] == method_config.CONFIG_DEFAULTS[legacy]:
                config[legacy] = config[primary]
                continue
            if normalized_input[primary] == default:
                config[primary] = config[legacy]
                continue
            raise ValueError(
                f"flashomni config keys {primary!r} and {legacy!r} refer to the same upstream "
                "setting but have different non-default values"
            )
        if primary_set:
            config[legacy] = config[primary]
        elif legacy_set:
            config[primary] = config[legacy]
        else:
            config[legacy] = config[primary]


def _flashomni_explicit_attention(query, key, value, sparse_info, sparse_kv_info,
                                  sparse_info_indptr, sparse_kv_info_indptr,
                                  sparse_block_size_for_q, sparse_block_size_for_kv,
                                  implementation, backend, workspace_bytes,
                                  is_full=False, attention_mask=None, text_len=0,
                                  causal=False, pos_encoding_mode="NONE",
                                  use_fp16_qk_reduction=False, logits_soft_cap=0.0,
                                  sm_scale=None, rope_scale=None, rope_theta=None):
    """FlashOmni upstream-style path with caller-provided sparse-info tensors."""
    if implementation != "upstream":
        raise NotImplementedError("flashomni explicit sparse-info path requires implementation='upstream'")
    if not query.is_cuda:
        raise RuntimeError("flashomni upstream sparse-info path requires CUDA")

    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
    out = _flashomni_upstream_attention(
        q,
        k,
        v,
        None,
        q_len=query.shape[1],
        q_block_size=int(sparse_block_size_for_q),
        kv_block_size=int(sparse_block_size_for_kv),
        backend=backend,
        workspace_bytes=workspace_bytes,
        sparse_info=sparse_info,
        sparse_kv_info=sparse_kv_info,
        sparse_info_indptr=sparse_info_indptr,
        sparse_kv_info_indptr=sparse_kv_info_indptr,
        is_full=is_full,
        attention_mask=attention_mask,
        text_len=text_len,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    return out.permute(0, 2, 1, 3)


def _flashomni_attention(query, key, value, sparse_kv_budget,
                         sparse_block_size_for_q, sparse_block_size_for_kv,
                         implementation, backend, workspace_bytes,
                         causal=False, pos_encoding_mode="NONE",
                         use_fp16_qk_reduction=False, logits_soft_cap=0.0,
                         sm_scale=None, rope_scale=None, rope_theta=None):
    """Local q/k block-mean top-k sparse attention through FlashOmni/flex.

    This helper is an explicit SparseVideo diagnostic path and is not upstream
    FlashOmni video-method parity. Use sparse_pattern="explicit" with
    caller-provided sparse-info tensors for upstream-style execution.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    q = query.permute(0, 2, 1, 3)  # [B, H, N, D]
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    q_block_size = int(sparse_block_size_for_q)
    kv_block_size = int(sparse_block_size_for_kv)
    num_q_blocks = (N + q_block_size - 1) // q_block_size
    num_k_blocks = (N + kv_block_size - 1) // kv_block_size

    # Pad to block-aligned
    q_pad_n = num_q_blocks * q_block_size - N
    kv_pad_n = num_k_blocks * kv_block_size - N
    if q_pad_n > 0:
        q_padded = F.pad(q, (0, 0, 0, q_pad_n))
    else:
        q_padded = q
    if kv_pad_n > 0:
        k_padded = F.pad(k, (0, 0, 0, kv_pad_n))
        v_padded = F.pad(v, (0, 0, 0, kv_pad_n))
    else:
        k_padded = k
        v_padded = v

    q_len_padded = num_q_blocks * q_block_size
    kv_len_padded = num_k_blocks * kv_block_size

    # Block-level means: [B, H, num_blocks, D]
    q_blocks = q_padded.view(B, H, num_q_blocks, q_block_size, D).mean(dim=3)
    k_blocks = k_padded.view(B, H, num_k_blocks, kv_block_size, D).mean(dim=3)

    block_scores = torch.matmul(q_blocks, k_blocks.transpose(-2, -1)) * scale
    block_attn = F.softmax(block_scores, dim=-1)

    k_keep = min(num_k_blocks, max(1, int(num_k_blocks * sparse_kv_budget)))
    _, topk_idx = torch.topk(block_attn, k=k_keep, dim=-1)
    # block_mask_pattern: [B, H, nqb, nkb] bool
    block_mask_pattern = torch.zeros_like(block_attn, dtype=torch.bool)
    block_mask_pattern.scatter_(dim=-1, index=topk_idx, value=True)

    if implementation == "upstream":
        if not query.is_cuda:
            raise RuntimeError("flashomni upstream sparse path requires CUDA")
        out = _flashomni_upstream_attention(
            q_padded, k_padded, v_padded, block_mask_pattern,
            q_len=N, q_block_size=q_block_size, kv_block_size=kv_block_size,
            backend=backend, workspace_bytes=workspace_bytes,
            causal=causal, pos_encoding_mode=pos_encoding_mode,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            logits_soft_cap=logits_soft_cap, sm_scale=sm_scale,
            rope_scale=rope_scale, rope_theta=rope_theta,
        )
        if q_pad_n > 0:
            out = out[:, :, :N, :]
        return out.permute(0, 2, 1, 3)

    if implementation != "flex":
        raise ValueError("flashomni implementation must be 'upstream' or 'flex'")
    if (
        causal
        or pos_encoding_mode != "NONE"
        or use_fp16_qk_reduction
        or logits_soft_cap not in (None, 0.0)
        or sm_scale is not None
        or rope_scale is not None
        or rope_theta is not None
    ):
        raise NotImplementedError(
            "flashomni implementation='flex' is a local diagnostic fallback and "
            "does not implement upstream FlashOmni plan modifiers."
        )

    # Explicit fallback: use flex_attention with block mask derived from pattern.
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # block_mask_pattern is [B, H, nqb, nkb], use first batch element
    pattern = block_mask_pattern[0]  # [H, nqb, nkb]

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // q_block_size
        kv_block = kv_idx // kv_block_size
        q_block = torch.clamp(q_block, 0, num_q_blocks - 1)
        kv_block = torch.clamp(kv_block, 0, num_k_blocks - 1)
        return pattern[h, q_block, kv_block]

    bm = create_block_mask(
        mask_mod, B=None, H=H, Q_LEN=q_len_padded, KV_LEN=kv_len_padded,
        device=query.device, BLOCK_SIZE=(q_block_size, kv_block_size),
    )

    out = flex_attention(q_padded, k_padded, v_padded, block_mask=bm)
    if q_pad_n > 0:
        out = out[:, :, :N, :]
    return out.permute(0, 2, 1, 3)


def _flashomni_global_random_attention(query, key, value, spq_Q, spq_KV, sparse_size, text_token,
                                       sparse_block_size_for_q, sparse_block_size_for_kv,
                                       implementation, backend, workspace_bytes,
                                       attention_mask=None, text_len=0,
                                       causal=False, pos_encoding_mode="NONE",
                                       use_fp16_qk_reduction=False, logits_soft_cap=0.0,
                                       sm_scale=None, rope_scale=None, rope_theta=None):
    if implementation != "upstream":
        raise NotImplementedError("flashomni sparse_pattern='global_random' requires implementation='upstream'")
    if not query.is_cuda:
        raise RuntimeError("flashomni upstream global_random sparse path requires CUDA")
    if int(sparse_block_size_for_q) != int(sparse_size) or int(sparse_block_size_for_kv) != int(sparse_size):
        raise ValueError(
            "flashomni sparse_pattern='global_random' follows upstream benchmark sparse_size "
            "and requires sparse_block_size_for_q == sparse_block_size_for_kv == sparse_size"
        )

    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
    sparse_q, sparse_kv = _flashomni_global_random_sparse_blocks(
        batch_size=query.shape[0],
        num_heads=query.shape[2],
        q_len=query.shape[1],
        kv_len=key.shape[1],
        spq_Q=spq_Q,
        spq_KV=spq_KV,
        sparse_size=int(sparse_size),
        text_token=text_token,
        device=query.device,
    )
    return _flashomni_upstream_attention(
        q,
        k,
        v,
        sparse_kv,
        q_len=query.shape[1],
        q_block_size=int(sparse_size),
        kv_block_size=int(sparse_size),
        backend=backend,
        workspace_bytes=workspace_bytes,
        sparse_q_block_pattern=sparse_q,
        attention_mask=attention_mask,
        text_len=text_len,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    ).permute(0, 2, 1, 3)


def _flashomni_paper_mmdit_attention(query, key, value, tau_q, tau_kv, N, D, S_q, text_len,
                                     sparse_block_size_for_q, sparse_block_size_for_kv,
                                     implementation, backend, workspace_bytes,
                                     attention_mask=None, state=None, step=None,
                                     cache_dic=None, current=None,
                                     first_enhance=0, max_sequence_length=-1,
                                     num_inference_steps=50, simthreshd1=0.1,
                                     model_type=None,
                                     causal=False, pos_encoding_mode="NONE",
                                     use_fp16_qk_reduction=False, logits_soft_cap=0.0,
                                     sm_scale=None, rope_scale=None, rope_theta=None):
    if implementation != "upstream":
        raise NotImplementedError("flashomni sparse_pattern='paper_mmdit' requires implementation='upstream'")
    if not query.is_cuda:
        raise RuntimeError("flashomni upstream paper_mmdit sparse path requires CUDA")

    if state is None:
        state = _FlashOmniPaperMMDiTState()
    q_block_size = int(sparse_block_size_for_q)
    kv_block_size = int(sparse_block_size_for_kv)
    cache_order = int(D)
    schedule = _flashomni_paper_mmdit_schedule(
        _flashomni_paper_mmdit_effective_step(step, current),
        fresh_threshold=int(N),
        first_enhance=int(first_enhance),
        num_inference_steps=int(num_inference_steps),
        has_symbols=state.has_symbols(),
    )
    trim = _flashomni_trim_prefix_key_value_mask(
        key,
        value,
        attention_mask,
        text_len=int(text_len or 0),
    )

    if schedule.full:
        if schedule.compute_symbols:
            _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.policy.before")
            sparse_q, sparse_kv = _flashomni_build_mmdit_sparse_symbols(
                query,
                trim.key,
                model_type=model_type,
                q_block_size=q_block_size,
                kv_block_size=kv_block_size,
                threshold_q=float(tau_q),
                threshold_kv=float(tau_kv),
                saving_threshold_q_for_taylor=float(S_q),
                text_len=int(text_len or 0),
                kv_text_len=trim.kv_text_len,
                current_iter=schedule.current_iter,
                max_sequence_length=int(max_sequence_length),
                num_inference_steps=int(num_inference_steps),
                simthreshd1=float(simthreshd1),
                sm_scale=sm_scale,
            )
            _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.policy.after")
            state.update_symbols(sparse_q, sparse_kv)
            _flashomni_save_hunyuan_sparse_ratios(cache_dic, current, sparse_q, sparse_kv)
        _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.full.before")
        out = _flashomni_upstream_attention(
            query,
            trim.key,
            trim.value,
            None,
            q_len=query.shape[1],
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            backend=backend,
            workspace_bytes=workspace_bytes,
            is_full=True,
            attention_mask=trim.attention_mask,
            text_len=trim.kv_text_len,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            input_layout="NHD",
            return_layout="NHD",
        )
        _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.full.after", feature=out)
        if schedule.compute_symbols and state.sparse_q is not None:
            state.append_output(out, state.sparse_q, q_block_size, cache_order)
            _flashomni_save_hunyuan_attn_taylor(cache_dic, current, out)
            _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.full.after_cache", feature=out)
        state.last_dispatch = "dense"
        return _FlashOmniPaperMMDiTResult(out, "dense", "flashomni_full_upstream")

    if state.sparse_q is None or state.sparse_kv is None:
        raise RuntimeError("flashomni paper_mmdit dispatch has no cached sparse symbols")
    sparse_q = state.sparse_q.to(device=query.device, dtype=torch.uint8)
    sparse_kv = state.sparse_kv.to(device=query.device, dtype=torch.uint8)
    taylor_out = _flashomni_hunyuan_attn_taylor_out(cache_dic, current, device=query.device)
    _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.sparse.before")
    out = _flashomni_upstream_attention(
        query,
        trim.key,
        trim.value,
        sparse_kv,
        q_len=query.shape[1],
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        backend=backend,
        workspace_bytes=workspace_bytes,
        sparse_q_block_pattern=sparse_q,
        out=taylor_out,
        attention_mask=trim.attention_mask,
        text_len=trim.kv_text_len,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        input_layout="NHD",
        return_layout="NHD",
    )
    _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.sparse.after", feature=out)
    cached_output = None if taylor_out is not None else state.predicted_output(cache_order)
    if cached_output is not None:
        out = _flashomni_apply_cached_q_blocks(out, cached_output, sparse_q, q_block_size)
        _flashomni_trace_hunyuan_memory(cache_dic, current, "attention.sparse.after_cached_q", feature=out)
    state.last_dispatch = "sparse"
    return _FlashOmniPaperMMDiTResult(out, "sparse", "flashomni_explicit_upstream")


def _flashomni_trace_hunyuan_memory(cache_dic, current, event: str, *, feature=None) -> None:
    if not isinstance(cache_dic, dict) or not isinstance(current, dict):
        return
    trace = cache_dic.get("debug_memory_trace")
    if not isinstance(trace, list):
        return
    from .hunyuan_forward import _flashomni_trace_memory

    _flashomni_trace_memory(cache_dic, current, event, feature=feature)


def _flashomni_paper_mmdit_effective_step(step, current) -> int | None:
    if isinstance(current, dict) and current.get("step") is not None:
        try:
            return int(current["step"]) + 1
        except (TypeError, ValueError):
            pass
    return step


def _flashomni_build_mmdit_sparse_symbols(
    query,
    key,
    *,
    model_type,
    q_block_size,
    kv_block_size,
    threshold_q,
    threshold_kv,
    saving_threshold_q_for_taylor,
    text_len,
    kv_text_len,
    current_iter,
    max_sequence_length,
    num_inference_steps,
    simthreshd1,
    sm_scale,
):
    policy_text_len = int(max_sequence_length)
    if policy_text_len < 0:
        policy_text_len = int(text_len or 0)

    if (
        model_type == "hunyuan_video"
        and query.shape == key.shape
        and policy_text_len > 0
    ):
        return flashomni_hunyuan_sparse_blocks(
            query,
            key,
            sparse_block_size_for_q=q_block_size,
            sparse_block_size_for_kv=kv_block_size,
            threshold_q=threshold_q,
            threshold_kv=threshold_kv,
            current_iter=current_iter,
            max_sequence_length=policy_text_len,
            num_inference_steps=num_inference_steps,
            simthreshd1=simthreshd1,
            sm_scale=sm_scale,
        )

    return flashomni_paper_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=q_block_size,
        sparse_block_size_for_kv=kv_block_size,
        tau_q=threshold_q,
        tau_kv=threshold_kv,
        S_q=saving_threshold_q_for_taylor,
        text_len=int(text_len or 0),
        kv_text_len=kv_text_len,
        sm_scale=sm_scale,
    )


def _flashomni_save_hunyuan_sparse_ratios(cache_dic, current, sparse_q, sparse_kv) -> None:
    if not isinstance(cache_dic, dict) or not isinstance(current, dict):
        return
    try:
        layer_cache = cache_dic["cache"][-1][current["stream"]][current["layer"]]
    except (KeyError, TypeError):
        return
    layer_cache["sparse_ratio"] = [
        torch.count_nonzero(sparse_q).float() / sparse_q.numel(),
        torch.count_nonzero(sparse_kv).float() / sparse_kv.numel(),
    ]


def _flashomni_save_hunyuan_attn_taylor(cache_dic, current, output) -> None:
    if not isinstance(cache_dic, dict) or not isinstance(current, dict):
        return
    try:
        from .hunyuan_forward import derivative_approximation

        layer_cache = cache_dic["cache"][-1][current["stream"]][current["layer"]]
        taylor_started = cache_dic["cache_index"]["taylor_start"][current["stream"]][current["layer"]]
        if not (current.get("flashomni") and not taylor_started and current.get("module") == "attn"):
            return
        if "sparse_ratio" not in layer_cache:
            return
        derivative_approximation(
            cache_dic,
            current,
            output.contiguous().view(-1, output.shape[2], output.shape[3]),
            is_attn=True,
        )
    except (KeyError, TypeError, IndexError):
        return


def _flashomni_hunyuan_attn_taylor_out(cache_dic, current, *, device=None):
    if not isinstance(cache_dic, dict) or not isinstance(current, dict):
        return None
    try:
        from .hunyuan_forward import taylor_formula

        if current.get("type") != "Sparse" or current.get("sparse_type") != "flashomni":
            return None
        if current.get("module") != "attn":
            return None
        return taylor_formula(cache_dic, current, device=device).contiguous()
    except (KeyError, TypeError, IndexError):
        return None


def _flashomni_trim_prefix_key_value_mask(key, value, attention_mask, *, text_len: int) -> _FlashOmniPrefixMaskTrim:
    """Avoid FlashOmni custom-mask mode for Hunyuan prefix-valid padding masks."""
    if attention_mask is None or int(text_len) <= 0:
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, int(text_len or 0))

    batch_size = key.shape[0]
    kv_len = key.shape[1]
    mask = attention_mask.to(device=key.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    elif mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], -1)

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, int(text_len or 0))

    text_len = min(int(text_len), int(kv_len))
    video_len = max(0, int(kv_len) - text_len)
    if mask.shape[-1] == kv_len:
        key_allowed = mask
    elif mask.shape[-1] == text_len:
        key_allowed = torch.ones(batch_size, kv_len, device=key.device, dtype=torch.bool)
        key_allowed[:, video_len:] = mask
    else:
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, text_len)

    valid_lengths = key_allowed.to(torch.int64).sum(dim=-1)
    if not bool(torch.equal(valid_lengths, valid_lengths[:1].expand_as(valid_lengths))):
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, text_len)
    valid_len = int(valid_lengths[0].item())
    if valid_len < video_len:
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, text_len)

    prefix = torch.arange(kv_len, device=key.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
    if not bool(torch.equal(key_allowed, prefix)):
        return _FlashOmniPrefixMaskTrim(key, value, attention_mask, text_len)

    if valid_len >= kv_len:
        return _FlashOmniPrefixMaskTrim(key, value, None, text_len)

    kv_text_len = max(0, valid_len - video_len)
    return _FlashOmniPrefixMaskTrim(
        key[:, :valid_len].contiguous(),
        value[:, :valid_len].contiguous(),
        None,
        kv_text_len,
    )


def _flashomni_cache_q_blocks(output, sparse_q, q_block_size) -> _FlashOmniCachedQBlocks:
    if output.ndim != 4:
        raise RuntimeError("flashomni paper_mmdit output cache expects [B, S, H, D] output")
    if sparse_q.ndim != 3:
        raise RuntimeError("flashomni paper_mmdit sparse_q must have shape [B, H, q_blocks]")

    batch_size, seq_len, num_heads, head_dim = output.shape
    if sparse_q.shape[0] != batch_size or sparse_q.shape[1] != num_heads:
        raise RuntimeError("flashomni paper_mmdit sparse_q batch/head dimensions do not match output")

    q_block_size = int(q_block_size)
    num_q_blocks = int(sparse_q.shape[-1])
    padded_len = num_q_blocks * q_block_size
    if padded_len < seq_len:
        raise RuntimeError(
            "flashomni paper_mmdit sparse_q has fewer blocks than the output sequence requires: "
            f"{num_q_blocks} blocks of {q_block_size} for seq_len={seq_len}"
        )

    cached = ~sparse_q.to(device=output.device, dtype=torch.bool)
    indices = cached.nonzero(as_tuple=False)
    if indices.numel() == 0:
        values = output.new_empty((0, q_block_size, head_dim))
    else:
        cache_bytes = int(indices.shape[0]) * q_block_size * head_dim * output.element_size()
        values = _flashomni_gather_cached_q_blocks(
            output,
            indices,
            q_block_size=q_block_size,
            cache_to_cpu=cache_bytes >= _FLASHOMNI_Q_BLOCK_CACHE_CPU_THRESHOLD_BYTES,
        )
        if values.device.type == "cpu":
            indices = indices.cpu()
    return _FlashOmniCachedQBlocks(
        values=values,
        indices=indices.detach(),
        output_shape=(int(batch_size), int(seq_len), int(num_heads), int(head_dim)),
        q_block_size=q_block_size,
        num_q_blocks=num_q_blocks,
    )


def _flashomni_predict_cached_q_blocks(
    last: _FlashOmniCachedQBlocks,
    prev: _FlashOmniCachedQBlocks,
    prev_prev: _FlashOmniCachedQBlocks | None = None,
) -> _FlashOmniCachedQBlocks:
    if not _flashomni_cached_q_blocks_same_layout(last, prev):
        return last
    if prev_prev is None or not _flashomni_cached_q_blocks_same_layout(last, prev_prev):
        values = last.values + (last.values - prev.values.to(device=last.values.device, dtype=last.values.dtype))
    else:
        prev_values = prev.values.to(device=last.values.device, dtype=last.values.dtype)
        prev_prev_values = prev_prev.values.to(device=last.values.device, dtype=last.values.dtype)
        values = last.values + (last.values - prev_values) + 0.5 * (last.values - 2 * prev_values + prev_prev_values)
    return _FlashOmniCachedQBlocks(
        values=values,
        indices=last.indices,
        output_shape=last.output_shape,
        q_block_size=last.q_block_size,
        num_q_blocks=last.num_q_blocks,
    )


def _flashomni_cached_q_blocks_same_layout(a: _FlashOmniCachedQBlocks, b: _FlashOmniCachedQBlocks) -> bool:
    return (
        a.output_shape == b.output_shape
        and a.q_block_size == b.q_block_size
        and a.num_q_blocks == b.num_q_blocks
        and torch.equal(a.indices, b.indices.to(device=a.indices.device))
    )


def _flashomni_q_block_chunk_size(q_block_size: int, head_dim: int, element_size: int) -> int:
    bytes_per_block = max(1, int(q_block_size) * int(head_dim) * int(element_size))
    return max(1, _FLASHOMNI_Q_BLOCK_CACHE_CHUNK_BYTES // bytes_per_block)


def _flashomni_gather_cached_q_blocks(output, indices, *, q_block_size: int, cache_to_cpu: bool):
    seq_len = int(output.shape[1])
    head_dim = int(output.shape[3])
    values_shape = (int(indices.shape[0]), int(q_block_size), head_dim)
    if cache_to_cpu:
        values = torch.empty(values_shape, device="cpu", dtype=output.dtype)
    else:
        values = output.new_empty(values_shape)

    chunk_size = _flashomni_q_block_chunk_size(q_block_size, head_dim, output.element_size())
    token_range = torch.arange(int(q_block_size), device=output.device)
    for start in range(0, int(indices.shape[0]), chunk_size):
        end = min(start + chunk_size, int(indices.shape[0]))
        chunk_indices = indices[start:end].to(device=output.device)
        batch_idx = chunk_indices[:, 0]
        head_idx = chunk_indices[:, 1]
        block_idx = chunk_indices[:, 2]
        token_offsets = block_idx[:, None] * int(q_block_size) + token_range[None, :]
        valid = token_offsets < seq_len
        token_offsets = token_offsets.clamp(max=max(0, seq_len - 1))
        gathered = output[batch_idx[:, None], token_offsets, head_idx[:, None], :]
        if not bool(valid.all()):
            gathered = gathered.clone()
            gathered[~valid] = 0
        values[start:end].copy_(gathered, non_blocking=False)
    return values.detach()


def _flashomni_apply_cached_q_blocks(out, cached_output, sparse_q, q_block_size):
    if isinstance(cached_output, _FlashOmniCachedQBlocks):
        return _flashomni_apply_compact_cached_q_blocks(out, cached_output, sparse_q, q_block_size)

    if out.shape != cached_output.shape:
        raise RuntimeError(
            "flashomni paper_mmdit cached output shape does not match current attention output: "
            f"{tuple(cached_output.shape)} vs {tuple(out.shape)}"
        )
    if sparse_q.ndim != 3:
        raise RuntimeError("flashomni paper_mmdit sparse_q must have shape [B, H, q_blocks]")
    batch_size, seq_len, num_heads, _ = out.shape
    if sparse_q.shape[0] != batch_size or sparse_q.shape[1] != num_heads:
        raise RuntimeError("flashomni paper_mmdit sparse_q batch/head dimensions do not match output")

    block_idx = torch.arange(seq_len, device=out.device) // int(q_block_size)
    block_idx = block_idx.clamp(max=sparse_q.shape[-1] - 1)
    gather_idx = block_idx.view(1, 1, seq_len).expand(batch_size, num_heads, seq_len)
    active = sparse_q.to(device=out.device, dtype=torch.bool).gather(dim=2, index=gather_idx)
    cached = ~active.permute(0, 2, 1).unsqueeze(-1)
    return torch.where(cached, cached_output.to(device=out.device, dtype=out.dtype), out)


def _flashomni_apply_compact_cached_q_blocks(out, cached_output, sparse_q, q_block_size):
    if tuple(out.shape) != cached_output.output_shape:
        raise RuntimeError(
            "flashomni paper_mmdit cached output shape does not match current attention output: "
            f"{cached_output.output_shape} vs {tuple(out.shape)}"
        )
    if sparse_q.ndim != 3:
        raise RuntimeError("flashomni paper_mmdit sparse_q must have shape [B, H, q_blocks]")
    batch_size, seq_len, num_heads, head_dim = out.shape
    if sparse_q.shape[0] != batch_size or sparse_q.shape[1] != num_heads:
        raise RuntimeError("flashomni paper_mmdit sparse_q batch/head dimensions do not match output")
    if int(q_block_size) != cached_output.q_block_size:
        raise RuntimeError("flashomni paper_mmdit cached q_block_size does not match current sparse_q")
    if int(sparse_q.shape[-1]) != cached_output.num_q_blocks:
        raise RuntimeError("flashomni paper_mmdit cached q block count does not match current sparse_q")

    cached = ~sparse_q.to(device=out.device, dtype=torch.bool)
    expected_indices = cached.nonzero(as_tuple=False)
    cache_indices = cached_output.indices.to(device=out.device)
    if not torch.equal(expected_indices, cache_indices):
        raise RuntimeError("flashomni paper_mmdit cached q blocks do not match current sparse_q")
    if expected_indices.numel() == 0:
        return out

    _flashomni_scatter_cached_q_blocks(
        out,
        cached_output.values,
        cached_output.indices,
        q_block_size=cached_output.q_block_size,
    )
    return out


def _flashomni_scatter_cached_q_blocks(out, values, indices, *, q_block_size: int) -> None:
    seq_len = int(out.shape[1])
    head_dim = int(out.shape[3])
    chunk_size = _flashomni_q_block_chunk_size(q_block_size, head_dim, out.element_size())
    token_range = torch.arange(int(q_block_size), device=out.device)
    for start in range(0, int(indices.shape[0]), chunk_size):
        end = min(start + chunk_size, int(indices.shape[0]))
        chunk_indices = indices[start:end].to(device=out.device)
        chunk_values = values[start:end].to(device=out.device, dtype=out.dtype)
        batch_idx = chunk_indices[:, 0]
        head_idx = chunk_indices[:, 1]
        block_idx = chunk_indices[:, 2]
        full_rows = (block_idx + 1) * int(q_block_size) <= seq_len
        if bool(full_rows.any()):
            full_batch = batch_idx[full_rows]
            full_head = head_idx[full_rows]
            full_tokens = block_idx[full_rows, None] * int(q_block_size) + token_range[None, :]
            out[full_batch[:, None], full_tokens, full_head[:, None], :] = chunk_values[full_rows]
        if not bool(full_rows.all()):
            tail_rows = (~full_rows).nonzero(as_tuple=False).flatten()
            for row in tail_rows.tolist():
                token_start = int(block_idx[row].item()) * int(q_block_size)
                valid_len = max(0, min(int(q_block_size), seq_len - token_start))
                if valid_len:
                    out[
                        int(batch_idx[row].item()),
                        token_start: token_start + valid_len,
                        int(head_idx[row].item()),
                        :,
                    ] = chunk_values[row, :valid_len]


def _flashomni_global_random_sparse_blocks(batch_size, num_heads, q_len, kv_len,
                                           spq_Q, spq_KV, sparse_size, device,
                                           text_token=512):
    """Port of FlashOmni benchmark get_qkvo_global_sparse mask generation.

    text_token is kept for upstream API parity; the upstream benchmark accepts
    it but does not use it in the random global mask math.
    """
    num_q_blocks = math.ceil(int(q_len) / int(sparse_size))
    num_kv_blocks = math.ceil(int(kv_len) / int(sparse_size))
    if num_q_blocks <= 0:
        raise ValueError("flashomni global_random requires at least one query block")

    sparse_q = torch.ones(
        (batch_size, num_heads, num_q_blocks),
        device=device,
        dtype=torch.uint8,
    )
    sparse_kv = torch.ones(
        (batch_size, num_heads, num_q_blocks, num_kv_blocks),
        device=device,
        dtype=torch.uint8,
    )

    num_to_zero_q = int(batch_size * num_heads * num_q_blocks * float(spq_Q))
    num_to_zero_kv = int(batch_size * num_heads * num_q_blocks * num_kv_blocks * float(spq_KV))

    if num_to_zero_q > 0:
        total_q = batch_size * num_heads * num_q_blocks
        flat = torch.randperm(total_q, device=device)[:num_to_zero_q]
        stride_h = num_q_blocks
        stride_b = num_heads * stride_h
        batch_idx = flat // stride_b
        head_idx = (flat % stride_b) // stride_h
        q_idx = flat % stride_h
        sparse_q[batch_idx, head_idx, q_idx] = 0

    if num_to_zero_kv > 0:
        total_kv = batch_size * num_heads * num_q_blocks * num_kv_blocks
        flat = torch.randperm(total_kv, device=device)[:num_to_zero_kv]
        stride_n = num_kv_blocks
        stride_h = num_q_blocks * stride_n
        stride_b = num_heads * stride_h
        batch_idx = flat // stride_b
        head_idx = (flat % stride_b) // stride_h
        q_idx = (flat % stride_h) // stride_n
        kv_idx = flat % stride_n
        sparse_kv[batch_idx, head_idx, q_idx, kv_idx] = 0

    return sparse_q, sparse_kv


def _flashomni_import():
    global _FLASHOMNI_MODULE_CACHE
    if _FLASHOMNI_MODULE_CACHE is not None:
        return _FLASHOMNI_MODULE_CACHE
    try:
        flashomni = None
        for root in _candidate_flashomni_roots():
            if _has_flashomni_extension(root):
                flashomni = _import_flashomni_from_root(root)
                break
        if flashomni is None:
            raise ImportError(
                "SparseVideo-owned FlashOmni runtime is not built under "
                f"{_local_flashomni_root()}"
            )
        if (
            "FLASHOMNI_WORKSPACE_BASE" not in os.environ
            and not os.access(Path.home(), os.W_OK)
        ):
            os.environ["FLASHOMNI_WORKSPACE_BASE"] = tempfile.gettempdir()
        if _is_training_free_runtime(flashomni):
            raise ImportError(
                "flashomni resolved from training_free/, which is reference-only for SparseVideo runtime"
            )
        _FLASHOMNI_MODULE_CACHE = flashomni
    except Exception as exc:
        raise ImportError(
            "flashomni implementation='upstream' requires the SparseVideo-owned "
            "FlashOmni package with its CUDA/C++ ops built under "
            "src/sparsevideo/kernels/native/flashomni. Do not rely on training_free/ "
            "or environment flashomni packages for SparseVideo runtime parity."
        ) from exc
    return flashomni


def _local_flashomni_root() -> Path:
    return Path(__file__).resolve().parents[2] / "kernels" / "native" / "flashomni"


def _candidate_flashomni_roots() -> list[Path]:
    local_root = _local_flashomni_root()
    roots = []
    env_root = os.environ.get("SPARSEVIDEO_FLASHOMNI_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        resolved = root.resolve(strict=False)
        if "training_free" in root.parts or "training_free" in resolved.parts:
            raise ImportError(
                "Refusing SPARSEVIDEO_FLASHOMNI_ROOT inside training_free; "
                "SparseVideo runtime kernels must live under "
                f"{local_root}."
            )
        try:
            resolved.relative_to(local_root.resolve())
        except ValueError:
            raise ImportError(
                "Refusing SPARSEVIDEO_FLASHOMNI_ROOT outside the SparseVideo-owned "
                f"runtime root {local_root}: {resolved}"
            )
        roots.append(resolved)
    roots.append(local_root)
    return roots


def _has_flashomni_extension(root: Path) -> bool:
    package = root / "flashomni"
    return (
        (package / "__init__.py").exists()
        and (
            (package / "jit" / "aot_config.py").exists()
            or (package / "aot_config.py").exists()
        )
        and bool(list(package.glob("flashomni_kernels*.so")))
    )


def _clear_flashomni_modules() -> None:
    global _FLASHOMNI_MODULE_CACHE
    _FLASHOMNI_MODULE_CACHE = None
    for name in list(sys.modules):
        if name == "flashomni" or name.startswith("flashomni."):
            del sys.modules[name]


def _import_flashomni_from_root(root: Path):
    _clear_flashomni_modules()
    root_str = str(root)
    added = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added = True
    try:
        return importlib.import_module("flashomni")
    finally:
        if added and root_str in sys.path:
            sys.path.remove(root_str)


def _is_training_free_runtime(module: ModuleType) -> bool:
    location = getattr(module, "__file__", None)
    if not location:
        return False
    return "training_free" in Path(location).resolve().parts


def _flashomni_upstream_attention(q, k, v, block_mask_pattern,
                                  q_len, q_block_size, kv_block_size,
                                  backend, workspace_bytes,
                                  sparse_info=None, sparse_kv_info=None,
                                  sparse_info_indptr=None, sparse_kv_info_indptr=None,
                                  is_full=False, attention_mask=None, text_len=0,
                                  sparse_q_block_pattern=None, out=None, causal=False,
                                  pos_encoding_mode="NONE",
                                  use_fp16_qk_reduction=False, logits_soft_cap=0.0,
                                  sm_scale=None, rope_scale=None, rope_theta=None,
                                  input_layout="HND", return_layout="HND"):
    """Execute FlashOmni's BatchFlashOmniFAWithRaggedKVWrapper.

    q/k/v are [B, H, S, D]. FlashOmni uses NHD ragged tensors, so each video in
    the batch is represented by one indptr segment.
    """
    flashomni = _flashomni_import()
    if input_layout not in ("HND", "NHD"):
        raise ValueError("flashomni input_layout must be 'HND' or 'NHD'")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("flashomni q/k/v tensors must have shape [B, H, S, D] or [B, S, H, D]")
    if input_layout == "HND":
        B, num_qo_heads, q_len_padded, head_dim_qk = q.shape
        num_kv_heads = k.shape[1]
        kv_len_padded = k.shape[2]
        value_num_kv_heads = v.shape[1]
        value_kv_len_padded = v.shape[2]
        q_nhd = q.transpose(1, 2).contiguous().view(B * q_len_padded, num_qo_heads, head_dim_qk)
        k_nhd = k.transpose(1, 2).contiguous().view(B * kv_len_padded, num_kv_heads, head_dim_qk)
        v_nhd = v.transpose(1, 2).contiguous().view(B * kv_len_padded, num_kv_heads, v.shape[-1])
    else:
        B, q_len_padded, num_qo_heads, head_dim_qk = q.shape
        num_kv_heads = k.shape[2]
        kv_len_padded = k.shape[1]
        value_num_kv_heads = v.shape[2]
        value_kv_len_padded = v.shape[1]
        q_nhd = q.contiguous().view(B * q_len_padded, num_qo_heads, head_dim_qk)
        k_nhd = k.contiguous().view(B * kv_len_padded, num_kv_heads, head_dim_qk)
        v_nhd = v.contiguous().view(B * kv_len_padded, num_kv_heads, v.shape[-1])
    if q_len > q_len_padded:
        raise RuntimeError("flashomni q_len cannot exceed padded query length")
    if k.shape[0] != B or v.shape[0] != B:
        raise RuntimeError("flashomni q/k/v batch sizes must match")
    if value_num_kv_heads != num_kv_heads:
        raise RuntimeError("flashomni key/value head counts must match")
    if k.shape[-1] != head_dim_qk:
        raise RuntimeError("flashomni query/key head dimensions must match")
    head_dim_vo = v.shape[-1]
    if value_kv_len_padded != kv_len_padded:
        raise RuntimeError("flashomni key/value sequence lengths must match")
    device = q.device
    out_nhd = None
    if out is not None:
        out_nhd = out.to(device=device, dtype=q.dtype).contiguous()
        if out_nhd.shape != (B * q_len_padded, num_qo_heads, head_dim_vo):
            raise RuntimeError(
                "flashomni upstream attention out must have shape "
                f"{(B * q_len_padded, num_qo_heads, head_dim_vo)}, got {tuple(out_nhd.shape)}"
            )

    qo_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * q_len_padded
    kv_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * kv_len_padded
    custom_mask = _flashomni_custom_mask_from_attention_mask(
        attention_mask,
        batch_size=B,
        q_len=q_len_padded,
        kv_len=kv_len_padded,
        text_len=text_len,
        device=device,
    )
    packed_custom_mask = None
    if custom_mask is not None:
        mask_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * (q_len_padded * kv_len_padded)
        packed_custom_mask, _ = flashomni.segment_packbits(
            custom_mask.contiguous().view(-1),
            mask_indptr,
            bitorder="little",
        )
        custom_mask = None
    workspace = torch.empty(int(workspace_bytes), dtype=torch.uint8, device=device)
    wrapper = flashomni.attention.BatchFlashOmniFAWithRaggedKVWrapper(
        workspace, kv_layout="NHD", backend=backend,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        sparse_block_size_for_q=q_block_size,
        sparse_block_size_for_kv=kv_block_size,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
        logits_soft_cap=logits_soft_cap,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        custom_mask=custom_mask,
        packed_custom_mask=packed_custom_mask,
    )

    if block_mask_pattern is None:
        if any(
            item is None
            for item in (sparse_info, sparse_kv_info, sparse_info_indptr, sparse_kv_info_indptr)
        ):
            if not is_full:
                raise RuntimeError("flashomni explicit sparse-info path received incomplete sparse tensors")
            packed_sparse_info, sparse_info_indptr = _flashomni_all_ones_sparse_info(
                flashomni, wrapper._sparse_info_indptr_base,
            )
            packed_sparse_kv_info, sparse_kv_info_indptr = _flashomni_all_ones_sparse_info(
                flashomni, wrapper._sparse_kv_info_indptr_base,
            )
        else:
            packed_sparse_info, sparse_info_indptr = _flashomni_normalize_sparse_bits(
                flashomni,
                sparse_info,
                sparse_info_indptr,
                wrapper._sparse_info_indptr_base,
                "sparse_info",
                device,
            )
            packed_sparse_kv_info, sparse_kv_info_indptr = _flashomni_normalize_sparse_bits(
                flashomni,
                sparse_kv_info,
                sparse_kv_info_indptr,
                wrapper._sparse_kv_info_indptr_base,
                "sparse_kv_info",
                device,
            )
    else:
        num_q_blocks = block_mask_pattern.shape[2]
        num_kv_blocks = block_mask_pattern.shape[3]
        if sparse_q_block_pattern is None:
            sparse_info = torch.ones((B, num_q_blocks, num_qo_heads), device=device, dtype=torch.uint8)
            sparse_info = sparse_info.contiguous().view(-1, num_qo_heads)
        else:
            sparse_info = sparse_q_block_pattern.to(
                device=device, dtype=torch.uint8,
            ).transpose(1, 2).contiguous().view(-1, num_qo_heads)
        sparse_kv_info = block_mask_pattern.to(torch.uint8).transpose(1, 2).contiguous().view(
            -1, num_qo_heads, num_kv_blocks,
        )
        sparse_kv_info = sparse_kv_info * sparse_info.unsqueeze(-1)

        packed_sparse_info, sparse_info_indptr = flashomni.segment_packbits(
            sparse_info.contiguous().view(-1),
            wrapper._sparse_info_indptr_base,
            bitorder="little",
        )
        packed_sparse_kv_info, sparse_kv_info_indptr = flashomni.segment_packbits(
            sparse_kv_info.contiguous().view(-1),
            wrapper._sparse_kv_info_indptr_base,
            bitorder="little",
        )
    run_kwargs = {} if out_nhd is None else {"out": out_nhd}
    out = wrapper.run(
        q_nhd, k_nhd, v_nhd,
        packed_sparse_info,
        packed_sparse_kv_info,
        sparse_info_indptr,
        sparse_kv_info_indptr,
        bool(is_full),
        **run_kwargs,
    )
    out = out.view(B, q_len_padded, num_qo_heads, head_dim_vo)
    if return_layout == "NHD":
        return out
    if return_layout != "HND":
        raise ValueError("flashomni return_layout must be 'HND' or 'NHD'")
    return out.transpose(1, 2).contiguous()


def _flashomni_all_ones_sparse_info(flashomni, indptr):
    indptr = indptr.to(dtype=torch.int32).contiguous()
    values = torch.ones(int(indptr[-1].item()), dtype=torch.uint8, device=indptr.device)
    return flashomni.segment_packbits(values, indptr, bitorder="little")


def _flashomni_normalize_sparse_bits(flashomni, values, indptr, expected_unpacked_indptr, name, device):
    values = values.to(device=device, dtype=torch.uint8).contiguous().view(-1)
    indptr = indptr.to(device=device, dtype=torch.int32).contiguous()
    expected_unpacked_indptr = expected_unpacked_indptr.to(device=device, dtype=torch.int32).contiguous()

    expected_unpacked_len = int(expected_unpacked_indptr[-1].item())
    expected_packed_indptr = _flashomni_packed_indptr(expected_unpacked_indptr)
    expected_packed_len = int(expected_packed_indptr[-1].item())

    if torch.equal(indptr, expected_unpacked_indptr):
        if values.numel() != expected_unpacked_len:
            raise RuntimeError(
                f"flashomni {name} has {values.numel()} values but its unpacked indptr "
                f"requires {expected_unpacked_len}"
            )
        return flashomni.segment_packbits(values, indptr, bitorder="little")

    if torch.equal(indptr, expected_packed_indptr):
        if values.numel() != expected_packed_len:
            raise RuntimeError(
                f"flashomni packed {name} has {values.numel()} values but its packed indptr "
                f"requires {expected_packed_len}"
            )
        return values, indptr

    raise RuntimeError(
        f"flashomni {name}_indptr does not match the current wrapper layout. "
        "Pass either upstream unpacked sparse-info tensors with the wrapper's logical "
        "indptr or already packed sparse-info tensors with the packed indptr returned "
        "by flashomni.segment_packbits."
    )


def _flashomni_packed_indptr(unpacked_indptr):
    seglen = unpacked_indptr[1:] - unpacked_indptr[:-1]
    packed_len = (seglen + 7) // 8
    packed_indptr = torch.zeros_like(unpacked_indptr)
    packed_indptr[1:] = torch.cumsum(packed_len, 0)
    return packed_indptr.to(dtype=torch.int32)


def _flashomni_custom_mask_from_attention_mask(
    attention_mask,
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    text_len: int = 0,
    device=None,
):
    """Convert Diffusers key masks to FlashOmni's flattened custom_mask layout."""
    if attention_mask is None:
        return None

    mask = attention_mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    elif mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], -1)

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        raise RuntimeError(
            f"flashomni attention_mask batch size {mask.shape[0]} does not match batch size {batch_size}"
        )

    if mask.shape[-1] == kv_len:
        key_allowed = mask
    elif text_len and mask.shape[-1] == int(text_len):
        key_allowed = torch.ones(batch_size, kv_len, device=device, dtype=torch.bool)
        key_allowed[:, kv_len - int(text_len):] = mask
    else:
        raise RuntimeError(
            "flashomni attention_mask must cover either the full KV sequence "
            f"({kv_len}) or the Hunyuan text tail ({int(text_len)}); got {mask.shape[-1]}"
        )

    if bool(key_allowed.all().item()):
        return None

    return key_allowed[:, None, :].expand(batch_size, q_len, kv_len).contiguous().view(-1)
