from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from .._schedule import configured_dense_warmup_layer_count, configured_dense_warmup_requires_dense, runtime_num_inference_steps
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from ...kernels.spas_sage_runtime import (
    _clear_spas_sage_modules,
    _has_spas_sage_extensions,
    _is_training_free_runtime,
    load_block_sparse_sage2_attn_function,
    load_spas_sage_attn_functions,
    load_sparse_attention_meansim_class,
)
from . import config as method_config

_TUNED_STATE_NAMES = ("is_sparse", "cdfthreshd", "simthreshd1", "simthreshd2", "pvthreshd")


def _load_spas_sage_attn_functions():
    return load_spas_sage_attn_functions()


class SpargeAttnMethod(SparseMethod):
    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if self.config["mode"] not in ("cdfthreshd", "topk", "block_sparse"):
            raise ValueError("spargeattn mode must be one of: cdfthreshd, topk, block_sparse")
        if self.config["tensor_layout"] != "HND":
            raise ValueError("spargeattn SparseVideo processor uses tensor_layout='HND'")
        if self.config["return_sparsity"]:
            raise NotImplementedError(
                "spargeattn return_sparsity=True returns (output, sparsity) in the upstream kernel "
                "and is not supported inside Diffusers attention processors."
            )
        if self.config["pv_l1"] <= self.config["l1"]:
            raise ValueError("spargeattn pv_l1 must be greater than l1")
        if self.config["sim_rule"] not in ("l1", "cosine", "rmse"):
            raise ValueError("spargeattn sim_rule must be one of: l1, cosine, rmse")
        if not isinstance(self.config["rearrange_kwargs"], dict):
            raise TypeError("spargeattn rearrange_kwargs must be a dict")
        self._sparge_cdf_fn = None
        self._sparge_topk_fn = None
        self._sparge_block_sparse_fn = None
        self._SparseAttentionMeansim = None
        self._loaded_tuned_state = None
        self._tuned_attentions = {}
        self._processor_state_index = 0
        self._use_tuned_path = bool(self.config["tune"] or self.config["model_out_path"])
        if self._use_tuned_path:
            self._SparseAttentionMeansim = load_sparse_attention_meansim_class()
            if not self.config["tune"]:
                self._loaded_tuned_state = _load_tuned_state_dict(self.config["model_out_path"])
        else:
            try:
                self._sparge_cdf_fn, self._sparge_topk_fn = _load_spas_sage_attn_functions()
                if self.config["mode"] == "block_sparse":
                    self._sparge_block_sparse_fn = load_block_sparse_sage2_attn_function()
            except ImportError as exc:
                raise ImportError(
                    "spargeattn sparse modes require the SparseVideo-owned spas_sage_attn "
                    "runtime under src/sparsevideo/kernels/native/spargeattn. "
                    "Do not rely on training_free/ or environment packages."
                ) from exc
            if self.config["mode"] == "block_sparse" and self.config["mask_id"] is None:
                raise ValueError("spargeattn mode='block_sparse' requires upstream mask_id")

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in (
            "wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate",
        ):
            raise NotImplementedError(f"spargeattn not yet supported for {self.model_info.model_type}")

        mode = self.config["mode"]
        sparge_value = self.config["value"]
        sparge_cdf_fn = self._sparge_cdf_fn
        sparge_topk_fn = self._sparge_topk_fn
        sparge_block_sparse_fn = self._sparge_block_sparse_fn
        topk = self.config["topk"] if sparge_value is None else sparge_value
        cdfthreshd = self.config["cdfthreshd"] if sparge_value is None else sparge_value
        common_kwargs = _sparge_kernel_kwargs(self.config, include_is_causal=True)
        block_sparse_kwargs = _sparge_kernel_kwargs(self.config, include_is_causal=False)
        tune = bool(self.config["tune"])
        parallel_tune = bool(self.config["parallel_tune"])
        state_layer_idx = self._processor_state_index
        self._processor_state_index += 1
        tuned_attention = self._create_tuned_attention(state_layer_idx) if self._use_tuned_path else None
        dense_warmup_layer_count = configured_dense_warmup_layer_count(self.config, total_layers)
        def attn_fn(query, key, value, attention_mask, **kwargs):
            use_dense_mode = (
                layer_idx < dense_warmup_layer_count
                or configured_dense_warmup_requires_dense(
                    self.config,
                    runtime_num_inference_steps(step_tracker),
                    getattr(step_tracker, "step", None),
                    notifier=self.warmup_notifier,
                )
            )
            if use_dense_mode:
                out = _sparge_dense_attention(
                    query, key, value, attention_mask,
                    model_type=self.model_info.model_type,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_sparge_dense_backend_name(self.model_info.model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            rejection_reason = _sparge_sparse_rejection_reason(query, attention_mask)
            if rejection_reason is not None:
                raise RuntimeError(
                    "spargeattn sparse path requires the upstream spas_sage_attn CUDA kernels; "
                    f"{rejection_reason}. Use method=dense for the dense baseline."
                )

            # Diffusers layout: [B, N, H, D] → SpargeAttn layout: [B, H, N, D]
            original_head_dim = query.shape[-1]
            kernel_head_dim = _sparge_kernel_head_dim(original_head_dim)
            q_hnd = query.permute(0, 2, 1, 3).contiguous()
            k_hnd = key.permute(0, 2, 1, 3).contiguous()
            v_hnd = value.permute(0, 2, 1, 3).contiguous()
            q_hnd, k_hnd, v_hnd = _pad_sparge_head_dim(q_hnd, k_hnd, v_hnd, kernel_head_dim)

            if tuned_attention is not None:
                _move_tuned_state_to_device(tuned_attention, q_hnd.device)
                with _sparge_tune_env(parallel_tune):
                    o_hnd = tuned_attention(
                        q_hnd,
                        k_hnd,
                        v_hnd,
                        is_causal=False,
                        tensor_layout="HND",
                        tune_mode=tune,
                    )
            elif mode == "cdfthreshd":
                kwargs = _sparge_kernel_kwargs_for_head_dim(common_kwargs, original_head_dim, kernel_head_dim)
                kwargs["cdfthreshd"] = cdfthreshd
                if self.config["simthreshd1"] is not None:
                    kwargs["simthreshd1"] = self.config["simthreshd1"]
                o_hnd = sparge_cdf_fn(q_hnd, k_hnd, v_hnd, **kwargs)
            elif mode == "topk":
                kwargs = _sparge_kernel_kwargs_for_head_dim(common_kwargs, original_head_dim, kernel_head_dim)
                kwargs["topk"] = topk
                if self.config["simthreshd1"] is not None:
                    kwargs["simthreshd1"] = self.config["simthreshd1"]
                o_hnd = sparge_topk_fn(q_hnd, k_hnd, v_hnd, **kwargs)
            else:
                kwargs = _sparge_kernel_kwargs_for_head_dim(block_sparse_kwargs, original_head_dim, kernel_head_dim)
                o_hnd = sparge_block_sparse_fn(
                    q_hnd,
                    k_hnd,
                    v_hnd,
                    mask_id=_move_mask_id_to_device(self.config["mask_id"], q_hnd.device),
                    **kwargs,
                )

            o_hnd = o_hnd[..., :original_head_dim].contiguous()
            out = o_hnd.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
            self.record_runtime_dispatch(
                "sparse",
                backend=_sparge_sparse_backend_name(mode, tuned_attention is not None),
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        fused = self.config.get("use_fused_qk_norm_rope", True)
        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
                use_fused_qk_norm_rope=fused,
            )
        if self.model_info.model_type == "cogvideox":
            return SparseCogVideoXAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "ltx_video":
            return SparseLTXVideoAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "allegro":
            return SparseAllegroAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "mochi":
            return SparseMochiAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "easyanimate":
            return SparseEasyAnimateAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            use_fused_qk_norm_rope=fused,
        )

    def install_model_patches(self, model_info):
        if model_info.model_type == "hunyuan_video":
            from .hunyuan_forward import install_spargeattn_hunyuan_forward_patch

            return [
                install_spargeattn_hunyuan_forward_patch(model_info),
                *super().install_model_patches(model_info),
            ]
        return super().install_model_patches(model_info)

    def _create_tuned_attention(self, layer_idx: int):
        tuned_attention = self._SparseAttentionMeansim(
            sim_rule=self.config["sim_rule"],
            l1=self.config["l1"],
            pv_l1=self.config["pv_l1"],
            cos_sim=self.config["cos_sim"],
            rmse=self.config["rmse"],
            rearrange_kwargs=dict(self.config["rearrange_kwargs"]),
            tune_pv=self.config["tune_pv"],
        )
        if self._loaded_tuned_state is not None:
            _load_tuned_attention_layer_state(
                tuned_attention,
                self._loaded_tuned_state,
                self._state_prefix(layer_idx),
            )
        self._tuned_attentions[layer_idx] = tuned_attention
        return tuned_attention

    def _state_prefix(self, layer_idx: int) -> str:
        path = self.model_info._self_attn_paths[layer_idx][0]
        return f"{path}.inner_attention"

    def export_state_dict(self):
        state = {}
        for layer_idx, tuned_attention in self._tuned_attentions.items():
            prefix = self._state_prefix(layer_idx)
            for name in _TUNED_STATE_NAMES:
                value = getattr(tuned_attention, name, None)
                if value is None:
                    continue
                if isinstance(value, torch.nn.Parameter):
                    value = value.detach()
                if torch.is_tensor(value):
                    state[f"{prefix}.{name}"] = value.detach().cpu()
        return state


def _sparge_dense_backend_name(model_type):
    if model_type == "hunyuan_video":
        return "torch_sdpa"
    return "diffusers_dispatch"


def _sparge_sparse_backend_name(mode, tuned):
    if tuned:
        return "spas_sage_tuned"
    if mode == "cdfthreshd":
        return "spas_sage_cdfthreshd"
    if mode == "topk":
        return "spas_sage_topk"
    if mode == "block_sparse":
        return "spas_sage_block_sparse"
    return "spas_sage"


def _sparge_sparse_rejection_reason(query, attention_mask):
    if not query.is_cuda:
        return "query/key/value are not CUDA tensors"
    if query.shape[1] < 128:
        return f"sequence length {query.shape[1]} is smaller than 128"
    if query.shape[-1] > 128:
        return f"head_dim {query.shape[-1]} is larger than the supported padded head_dim 128"
    if attention_mask is not None:
        return "attention_mask is not supported by the sparse kernel path"
    return None


def _sparge_kernel_head_dim(head_dim: int) -> int:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    raise RuntimeError(f"spargeattn cannot pad head_dim={head_dim} to a supported kernel width")


def _pad_sparge_head_dim(query, key, value, kernel_head_dim: int):
    head_dim = query.shape[-1]
    if head_dim == kernel_head_dim:
        return query, key, value
    pad = (0, kernel_head_dim - head_dim)
    return F.pad(query, pad), F.pad(key, pad), F.pad(value, pad)


def _sparge_kernel_kwargs_for_head_dim(base_kwargs, original_head_dim: int, kernel_head_dim: int):
    kwargs = dict(base_kwargs)
    if original_head_dim != kernel_head_dim and kwargs.get("scale") is None:
        kwargs["scale"] = original_head_dim ** -0.5
    return kwargs


def _sparge_dense_attention(query, key, value, attention_mask, *, model_type: str):
    if model_type == "hunyuan_video":
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3).contiguous(),
            key.permute(0, 2, 1, 3).contiguous(),
            value.permute(0, 2, 1, 3).contiguous(),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return out.permute(0, 2, 1, 3).contiguous()
    return dispatch_attention_fn(
        query, key, value,
        attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
    )


def _sparge_kernel_kwargs(config, *, include_is_causal: bool):
    kwargs = {
        "dropout_p": config["dropout_p"],
        "scale": config["scale"],
        "smooth_k": config["smooth_k"],
        "pvthreshd": config["pvthreshd"],
        "attention_sink": config["attention_sink"],
        "tensor_layout": "HND",
        "output_dtype": _resolve_torch_dtype(config["output_dtype"]),
        "return_sparsity": False,
    }
    if include_is_causal:
        kwargs["is_causal"] = False
    return kwargs


def _move_mask_id_to_device(mask_id, device):
    if torch.is_tensor(mask_id):
        return mask_id.to(device=device)
    return mask_id


def _resolve_torch_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.removeprefix("torch.")
        dtype = getattr(torch, normalized, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    raise ValueError(f"Unsupported spargeattn output_dtype: {value!r}")


def _load_tuned_state_dict(path):
    if path is None:
        raise ValueError("spargeattn model_out_path is required when tune=false and tuned-state inference is requested")
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"spargeattn model_out_path does not exist: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tuned_attention_layer_state(tuned_attention, state_dict, prefix: str) -> None:
    missing = []
    for name in _TUNED_STATE_NAMES:
        key = f"{prefix}.{name}"
        if key not in state_dict:
            missing.append(key)
            continue
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(f"spargeattn tuned state {key} must be a torch.Tensor")
        setattr(tuned_attention, name, torch.nn.Parameter(value, requires_grad=False))
    if missing:
        raise KeyError(
            "spargeattn tuned state is missing layer parameters. "
            f"Expected keys like {prefix}.cdfthreshd; missing {missing[:3]}"
        )


def _move_tuned_state_to_device(tuned_attention, device: torch.device) -> None:
    for name in _TUNED_STATE_NAMES:
        value = getattr(tuned_attention, name, None)
        if isinstance(value, torch.nn.Parameter) and value.device != device:
            setattr(tuned_attention, name, torch.nn.Parameter(value.detach().to(device=device), requires_grad=False))


@contextmanager
def _sparge_tune_env(parallel_tune: bool):
    old_parallel = os.environ.get("PARALLEL_TUNE")
    if parallel_tune:
        os.environ["PARALLEL_TUNE"] = "1"
    else:
        os.environ.pop("PARALLEL_TUNE", None)
    try:
        yield
    finally:
        if old_parallel is None:
            os.environ.pop("PARALLEL_TUNE", None)
        else:
            os.environ["PARALLEL_TUNE"] = old_parallel
