from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .._schedule import configured_dense_warmup_layer_count


_PATCH_REFCOUNT = 0
_ORIGINALS = {}


def install_flashomni_hunyuan_forward_patch(model_info, config, runtime_stats=None):
    """Install FlashOmni's package-owned Hunyuan forward/Taylor-cache patch.

    This ports the public anonymous FlashOmni Hunyuan forward path into
    SparseVideo-owned code. The attention sparse-symbol policy still lives in
    ``methods/flashomni/policy.py``; this patch adds the upstream block-level
    ``cal_type`` and ``taylor_formula`` behavior around the processors.
    """
    from diffusers.models.transformers import transformer_hunyuan_video as hunyuan

    memory_trace = None
    if bool(config.get("debug_memory", False)) and runtime_stats is not None:
        memory_trace = runtime_stats.setdefault("memory_trace", [])

    for transformer in model_info.transformers:
        transformer_config = dict(config)
        if memory_trace is not None:
            transformer_config["_memory_trace"] = memory_trace
        transformer._sparsevideo_flashomni_config = transformer_config
        transformer._sparsevideo_flashomni_cache_dic = None
        transformer._sparsevideo_flashomni_current = None

    patch_map = {
        hunyuan.HunyuanVideoSingleTransformerBlock: _flashomni_hunyuan_single_block_forward,
        hunyuan.HunyuanVideoTransformerBlock: _flashomni_hunyuan_double_block_forward,
        hunyuan.HunyuanVideoTransformer3DModel: flashomni_hunyuan_forward,
    }

    old_forward_patches = []
    for transformer in model_info.transformers:
        for module, forward in _flashomni_hunyuan_accelerate_patch_targets(transformer):
            if hasattr(module, "_old_forward"):
                old_forward_patches.append((module, module._old_forward))
                module._old_forward = forward.__get__(module, type(module))

    global _PATCH_REFCOUNT, _ORIGINALS
    if _PATCH_REFCOUNT == 0:
        _ORIGINALS = {cls: cls.forward for cls in patch_map}
        for cls, forward in patch_map.items():
            cls.forward = forward
    _PATCH_REFCOUNT += 1

    restored = False

    def restore():
        nonlocal restored
        global _PATCH_REFCOUNT, _ORIGINALS
        if restored:
            return
        restored = True
        for module, old_forward in reversed(old_forward_patches):
            module._old_forward = old_forward
        for transformer in model_info.transformers:
            for name in (
                "_sparsevideo_flashomni_config",
                "_sparsevideo_flashomni_cache_dic",
                "_sparsevideo_flashomni_current",
            ):
                if hasattr(transformer, name):
                    delattr(transformer, name)
        _PATCH_REFCOUNT = max(0, _PATCH_REFCOUNT - 1)
        if _PATCH_REFCOUNT == 0:
            for cls, forward in _ORIGINALS.items():
                cls.forward = forward
            _ORIGINALS = {}

    return restore


def _flashomni_hunyuan_accelerate_patch_targets(transformer):
    yield transformer, flashomni_hunyuan_forward
    for block in getattr(transformer, "transformer_blocks", []) or []:
        yield block, _flashomni_hunyuan_double_block_forward
    for block in getattr(transformer, "single_transformer_blocks", []) or []:
        yield block, _flashomni_hunyuan_single_block_forward


def flashomni_hunyuan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    attention_kwargs = {} if attention_kwargs is None else attention_kwargs.copy()
    config = getattr(self, "_sparsevideo_flashomni_config", {})
    cache_dic = attention_kwargs.get("cache_dic") or getattr(self, "_sparsevideo_flashomni_cache_dic", None)
    current = attention_kwargs.get("current") or getattr(self, "_sparsevideo_flashomni_current", None)
    if cache_dic is None or current is None:
        cache_dic, current = _flashomni_hunyuan_cache_init(self, config)
        self._sparsevideo_flashomni_cache_dic = cache_dic
        self._sparsevideo_flashomni_current = current
    attention_kwargs["cache_dic"] = cache_dic
    attention_kwargs["current"] = current
    cal_type(cache_dic, current)

    lora_scale = attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, _num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    effective_condition_sequence_length = _flashomni_effective_condition_length(encoder_attention_mask)
    encoder_hidden_states = encoder_hidden_states[:, :effective_condition_sequence_length]
    attention_kwargs["cache_dic"]["max_sequence_length"] = effective_condition_sequence_length
    attention_mask = None

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )
    else:
        current["stream"] = "double_stream"
        for layer, block in enumerate(self.transformer_blocks):
            current["layer"] = layer
            cal_type_sparse(cache_dic, current)
            _flashomni_trace_memory(cache_dic, current, "double.before_block")
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

        current["stream"] = "single_stream"
        for layer, block in enumerate(self.single_transformer_blocks):
            current["layer"] = layer
            cal_type_sparse(cache_dic, current)
            _flashomni_trace_memory(cache_dic, current, "single.before_block")
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        -1,
        p_t,
        p,
        p,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    current["step"] += 1
    if not return_dict:
        return (hidden_states,)
    return Transformer2DModelOutput(sample=hidden_states)


def _flashomni_hunyuan_double_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs=None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]

    if current["type"] == "full" or current.get("sparse_type") == "flashomni":
        current["module"] = "attn"
        taylor_cache_init(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "double.attn.before")
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
            **joint_attention_kwargs,
        )
        _maybe_start_taylor_cache(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "double.attn.after", feature=attn_output)

        current["module"] = "img_attn"
        taylor_cache_init(cache_dic, current)
        if _taylor_started(cache_dic, current):
            derivative_approximation(cache_dic, current, attn_output)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        current["module"] = "txt_attn"
        taylor_cache_init(cache_dic, current)
        if _taylor_started(cache_dic, current):
            derivative_approximation(cache_dic, current, context_attn_output)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        current["module"] = "img_mlp"
        taylor_cache_init(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "double.img_mlp.before_norm")
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        _flashomni_trace_memory(cache_dic, current, "double.img_mlp.before_ff", feature=norm_hidden_states)
        ff_output = self.ff(norm_hidden_states)
        _flashomni_trace_memory(cache_dic, current, "double.img_mlp.after_ff", feature=ff_output)
        if _taylor_started(cache_dic, current):
            derivative_approximation(cache_dic, current, ff_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        current["module"] = "txt_mlp"
        taylor_cache_init(cache_dic, current)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if _taylor_started(cache_dic, current):
            derivative_approximation(cache_dic, current, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    elif current.get("sparse_type") == "taylor_cache":
        current["module"] = "img_attn"
        _flashomni_trace_memory(cache_dic, current, "double.taylor.img_attn.before")
        attn_output = taylor_formula(cache_dic, current, device=hidden_states.device)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        current["module"] = "txt_attn"
        _flashomni_trace_memory(cache_dic, current, "double.taylor.txt_attn.before")
        context_attn_output = taylor_formula(cache_dic, current, device=encoder_hidden_states.device)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        current["module"] = "img_mlp"
        _flashomni_trace_memory(cache_dic, current, "double.taylor.img_mlp.before")
        ff_output = taylor_formula(cache_dic, current, device=hidden_states.device)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        current["module"] = "txt_mlp"
        _flashomni_trace_memory(cache_dic, current, "double.taylor.txt_mlp.before")
        context_ff_output = taylor_formula(cache_dic, current, device=encoder_hidden_states.device)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    return hidden_states, encoder_hidden_states


def _flashomni_hunyuan_single_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs=None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]

    if current["type"] == "full" or current.get("sparse_type") == "flashomni":
        _flashomni_trace_memory(cache_dic, current, "single.mlp.before")
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        _flashomni_trace_memory(cache_dic, current, "single.mlp.after", feature=mlp_hidden_states)
        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        current["module"] = "attn"
        taylor_cache_init(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "single.attn.before")
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        _maybe_start_taylor_cache(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "single.attn.after", feature=attn_output)

        current["module"] = "total"
        taylor_cache_init(cache_dic, current)
        _flashomni_trace_memory(cache_dic, current, "single.total.before_proj")
        hidden_states = torch.cat([torch.cat([attn_output, context_attn_output], dim=1), mlp_hidden_states], dim=2)
        hidden_states = self.proj_out(hidden_states)
        _flashomni_trace_memory(cache_dic, current, "single.total.after_proj", feature=hidden_states)
        if _taylor_started(cache_dic, current):
            derivative_approximation(cache_dic, current, hidden_states)
    elif current.get("sparse_type") == "taylor_cache":
        current["module"] = "total"
        _flashomni_trace_memory(cache_dic, current, "single.taylor.total.before")
        hidden_states = taylor_formula(cache_dic, current, device=hidden_states.device)

    hidden_states = gate.unsqueeze(1) * hidden_states
    hidden_states = hidden_states + residual
    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )
    return hidden_states, encoder_hidden_states


def _flashomni_hunyuan_cache_init(model, config: Dict[str, Any]):
    num_double_layers = int(model.config.num_layers)
    num_single_layers = int(model.config.num_single_layers)
    total_layers = num_double_layers + num_single_layers
    dense_warmup_layer_ratio = float(config.get("dense_warmup_layer_ratio", 0.03))
    cache: Dict[int, Any] = {-1: {"double_stream": {}, "single_stream": {}}}
    cache_index = {"taylor_start": {"double_stream": {}, "single_stream": {}}}
    for layer in range(num_double_layers):
        cache[-1]["double_stream"][layer] = {}
        cache_index["taylor_start"]["double_stream"][layer] = False
    for layer in range(num_single_layers):
        cache[-1]["single_stream"][layer] = {}
        cache_index["taylor_start"]["single_stream"][layer] = False

    cache_dic = {
        "cache_type": "random",
        "cache_index": cache_index,
        "cache": cache,
        "cache_counter": 0,
        "fresh_ratio": 0.0,
        "fresh_threshold": int(config.get("fresh_threshold", 6)),
        "force_fresh": "global",
        "soft_fresh_weight": 0.0,
        "taylor_cache": True,
        "max_order": int(config.get("max_order", 1)),
        "first_enhance": int(config.get("first_enhance", 8)),
        "threshold_q": float(config.get("threshold_q", 0.5)),
        "threshold_kv": float(config.get("threshold_kv", 0.05)),
        "saving_threshold_q_for_taylor": float(config.get("saving_threshold_q_for_taylor", 0.3)),
        "max_sequence_length": int(config.get("max_sequence_length", -1)),
        "dense_warmup_step_ratio": float(config.get("dense_warmup_step_ratio", 0.1)),
        "dense_warmup_layer_ratio": dense_warmup_layer_ratio,
        "dense_warmup_layer_count": configured_dense_warmup_layer_count(
            {"dense_warmup_layer_ratio": dense_warmup_layer_ratio},
            total_layers,
        ),
        "num_double_layers": num_double_layers,
        "taylor_cache_device": str(config.get("taylor_cache_device", "cuda")),
        "debug_memory_trace": config.get("_memory_trace") if bool(config.get("debug_memory", False)) else None,
        "debug_memory_max_events": int(config.get("debug_memory_max_events", 4000)),
    }
    current = {
        "activated_steps": [0],
        "step": 0,
        "num_steps": int(config.get("num_inference_steps", 50)),
        "flashomni": False,
    }
    return cache_dic, current


def cal_type(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    num_steps = max(1, int(current["num_steps"]))
    dense_ratio = max(0.0, min(1.0, float(cache_dic.get("dense_warmup_step_ratio", 0.0))))
    warmup_steps = num_steps if dense_ratio >= 1.0 else int(math.floor(dense_ratio * num_steps))
    is_warmup = int(current["step"]) < warmup_steps

    current["warmup"] = is_warmup
    current["flashomni"] = is_warmup and int(current["step"]) == warmup_steps - 1

    if is_warmup:
        current["type"] = "full"
        current["base_type"] = "full"
        current["sparse_type"] = None
        cache_dic["cache_counter"] = 0
        current["activated_steps"].append(current["step"])
        force_scheduler(cache_dic, current)
    elif cache_dic["taylor_cache"]:
        cache_dic["cache_counter"] += 1
        current["type"] = "Sparse"
        current["base_type"] = "Sparse"
        current["sparse_type"] = None


def cal_type_sparse(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    current["type"] = current.get("base_type", current["type"])
    current["sparse_type"] = None
    if current["type"] != "Sparse":
        return
    if _flashomni_hunyuan_layer_warmup_requires_full(cache_dic, current):
        current["type"] = "full"
        return
    if _taylor_started(cache_dic, current):
        current["sparse_type"] = "taylor_cache"
    else:
        current["sparse_type"] = "flashomni"


def _flashomni_hunyuan_layer_warmup_requires_full(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> bool:
    layer_count = int(cache_dic.get("dense_warmup_layer_count", 0) or 0)
    if layer_count <= 0:
        return False
    layer = int(current.get("layer", 0) or 0)
    if current.get("stream") == "single_stream":
        layer += int(cache_dic.get("num_double_layers", 0) or 0)
    return layer < layer_count


def force_scheduler(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    linear_step_weight = 0.0
    step_factor = 1 - linear_step_weight + 2 * linear_step_weight * current["step"] / current["num_steps"]
    cache_dic["cal_threshold"] = int(torch.round(torch.tensor(cache_dic["fresh_threshold"] / step_factor)).item())


def taylor_cache_init(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    module = current["module"]
    layer_cache = _layer_cache(cache_dic, current)
    if module not in layer_cache:
        layer_cache[module] = {}


def derivative_approximation(
    cache_dic: Dict[str, Any],
    current: Dict[str, Any],
    feature: torch.Tensor,
    is_attn: bool = False,
) -> None:
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    updated = {0: _flashomni_prepare_taylor_cache_tensor(cache_dic, feature)}
    max_order = int(cache_dic["max_order"]) if is_attn else 0
    previous = _layer_cache(cache_dic, current).get(current["module"], {})
    for order in range(max_order):
        if previous.get(order) is not None and current["step"] > cache_dic["first_enhance"] - 2:
            previous_order = _flashomni_taylor_tensor_to(previous[order], feature.device)
            updated[order + 1] = _flashomni_prepare_taylor_cache_tensor(
                cache_dic,
                (_flashomni_taylor_tensor_to(updated[order], feature.device) - previous_order) / difference_distance,
            )
        else:
            break
    _layer_cache(cache_dic, current)[current["module"]] = updated
    _flashomni_trace_memory(cache_dic, current, "taylor.cache_saved", feature=feature)


def taylor_formula(
    cache_dic: Dict[str, Any],
    current: Dict[str, Any],
    *,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    x = current["step"] - current["activated_steps"][-1]
    output = 0
    values = _layer_cache(cache_dic, current)[current["module"]]
    device = torch.device(device) if device is not None else _flashomni_taylor_output_device(values)
    for order in range(len(values)):
        value = _flashomni_taylor_tensor_to(values[order], device)
        output = output + (1 / math.factorial(order)) * value * (x ** order)
    return output


def _flashomni_prepare_taylor_cache_tensor(cache_dic: Dict[str, Any], tensor: torch.Tensor) -> torch.Tensor:
    cached = tensor.detach()
    if str(cache_dic.get("taylor_cache_device", "cuda")) == "cpu":
        return cached.to(device="cpu", non_blocking=True)
    return cached


def _flashomni_taylor_tensor_to(tensor: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
    if device is None or tensor.device == torch.device(device):
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _flashomni_taylor_output_device(values: Dict[int, torch.Tensor]) -> torch.device | None:
    for value in values.values():
        if torch.is_tensor(value) and value.device.type != "cpu":
            return value.device
    return None


def saving_sparse_info(cache_dic: Dict[str, Any], current: Dict[str, Any], sparse_q_ratio, sparse_kv_ratio) -> None:
    _layer_cache(cache_dic, current)["sparse_ratio"] = [sparse_q_ratio, sparse_kv_ratio]


def get_sparse_info(cache_dic: Dict[str, Any], current: Dict[str, Any]):
    return _layer_cache(cache_dic, current)["sparse_ratio"]


def del_cache(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    _layer_cache(cache_dic, current)[current["module"]] = {}


def _maybe_start_taylor_cache(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> None:
    if not (current.get("flashomni") and current["type"] == "full" and not _taylor_started(cache_dic, current)):
        return
    sparse_ratio = _layer_cache(cache_dic, current).get("sparse_ratio")
    if sparse_ratio is None:
        return
    sparse_q_ratio, _ = sparse_ratio
    if float(sparse_q_ratio) <= float(cache_dic["saving_threshold_q_for_taylor"]):
        cache_dic["cache_index"]["taylor_start"][current["stream"]][current["layer"]] = True
        del_cache(cache_dic, current)


def _taylor_started(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> bool:
    return bool(cache_dic["cache_index"]["taylor_start"][current["stream"]][current["layer"]])


def _layer_cache(cache_dic: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    return cache_dic["cache"][-1][current["stream"]][current["layer"]]


def _flashomni_trace_memory(
    cache_dic: Dict[str, Any],
    current: Dict[str, Any],
    event: str,
    *,
    feature: Optional[torch.Tensor] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    trace = cache_dic.get("debug_memory_trace")
    if not isinstance(trace, list):
        return
    item: Dict[str, Any] = {
        "event": event,
        "step": int(current.get("step", -1)),
        "stream": current.get("stream"),
        "layer": current.get("layer"),
        "module": current.get("module"),
        "type": current.get("type"),
        "sparse_type": current.get("sparse_type"),
    }
    if torch.is_tensor(feature):
        item["feature"] = _flashomni_tensor_summary(feature)
    item.update(_flashomni_cuda_memory_summary())
    item["cache"] = _flashomni_cache_memory_summary(cache_dic)
    if extra:
        item.update(extra)
    trace.append(item)
    max_events = max(1, int(cache_dic.get("debug_memory_max_events", 4000)))
    if len(trace) > max_events:
        del trace[: len(trace) - max_events]


def _flashomni_cuda_memory_summary() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "cuda_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def _flashomni_cache_memory_summary(cache_dic: Dict[str, Any]) -> Dict[str, Any]:
    seen: set[int] = set()
    by_path: Dict[str, int] = {}
    totals = {"total_bytes": 0, "cuda_bytes": 0, "cpu_bytes": 0, "tensor_count": 0}
    _flashomni_collect_tensor_bytes(cache_dic.get("cache"), "cache", seen, by_path, totals)
    top_paths = sorted(by_path.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "tensor_count": totals["tensor_count"],
        "total_gb": totals["total_bytes"] / (1024**3),
        "cuda_gb": totals["cuda_bytes"] / (1024**3),
        "cpu_gb": totals["cpu_bytes"] / (1024**3),
        "top": [
            {"path": path, "gb": bytes_value / (1024**3)}
            for path, bytes_value in top_paths
        ],
    }


def _flashomni_collect_tensor_bytes(
    value: Any,
    path: str,
    seen: set[int],
    by_path: Dict[str, int],
    totals: Dict[str, int],
) -> None:
    if torch.is_tensor(value):
        tensor_id = id(value)
        if tensor_id in seen:
            return
        seen.add(tensor_id)
        nbytes = _flashomni_tensor_nbytes(value)
        by_path[path] = nbytes
        totals["total_bytes"] += nbytes
        totals["tensor_count"] += 1
        if value.device.type == "cuda":
            totals["cuda_bytes"] += nbytes
        elif value.device.type == "cpu":
            totals["cpu_bytes"] += nbytes
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _flashomni_collect_tensor_bytes(item, f"{path}.{key}", seen, by_path, totals)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flashomni_collect_tensor_bytes(item, f"{path}.{index}", seen, by_path, totals)


def _flashomni_tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "gb": _flashomni_tensor_nbytes(tensor) / (1024**3),
    }


def _flashomni_tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _flashomni_effective_condition_length(encoder_attention_mask: torch.Tensor) -> int:
    lengths = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    if lengths.numel() == 0:
        return 0
    if not bool(torch.equal(lengths, lengths[:1].expand_as(lengths))):
        raise RuntimeError("FlashOmni Hunyuan forward patch requires equal prompt lengths within a batch")
    return int(lengths.flatten()[0].item())
