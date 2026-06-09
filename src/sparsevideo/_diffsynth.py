"""DiffSynth backend adapter for sparsevideo.apply().

This module intentionally owns only pipeline discovery, attention patching,
runtime tracking, and restore callbacks. DiffSynth model catalogs, local path
resolution, ModelConfig loading, generation kwargs, and media export belong in
scripts/ so the importable package stays a reusable sparse-attention layer.
"""

from __future__ import annotations

import inspect
import types
from importlib import metadata
from typing import Any, Callable, List, Tuple

import torch

from ._step_tracker import StepTracker


def diffsynth_version() -> str | None:
    try:
        return metadata.version("diffsynth")
    except metadata.PackageNotFoundError:
        return None


def discover_diffsynth_model(pipe: Any, *, infer_model_key: Callable) -> Any | None:
    if not _looks_like_diffsynth_pipeline(pipe):
        return None

    cls_name = type(pipe).__name__
    if "LTX2" in cls_name:
        return _discover_diffsynth_ltx2_model(pipe, infer_model_key=infer_model_key)

    transformers = []
    attn_paths = []
    loaded_modules = []
    for attr_name in ("dit", "dit2", "video_dit", "video_dit2"):
        module = getattr(pipe, attr_name, None)
        if module is None:
            continue
        if _is_diffsynth_longcat_module(module):
            paths = _enumerate_diffsynth_longcat_module(attr_name, module)
        else:
            paths = _enumerate_diffsynth_wan_module(attr_name, module)
        if paths:
            transformers.append(module)
            loaded_modules.append((attr_name, module))
            attn_paths.extend(paths)

    for attr_name in ("vace", "vace2"):
        module = getattr(pipe, attr_name, None)
        if module is not None:
            paths = _enumerate_diffsynth_wan_module(attr_name, module)
            if paths:
                transformers.append(module)
                loaded_modules.append((attr_name, module))
                attn_paths.extend(paths)

    if not attn_paths:
        raise ValueError(
            "DiffSynth pipeline is detected but no supported loaded Wan self-attention "
            "modules were found. Load WanVideoPipeline or Mova video_dit/video_dit2 "
            "models before calling sparsevideo.apply()."
        )

    from ._model_info import ModelInfo

    model_key = getattr(pipe, "_sparsevideo_model_key", None) or infer_model_key(pipe, transformers, "wan")
    unpatched_attention_paths, pipeline_notes = _diffsynth_unpatched_aux_attention_status(
        pipe,
        loaded_modules,
    )
    return ModelInfo(
        model_type="wan",
        transformers=transformers,
        model_key=model_key,
        pipeline_backend="diffsynth",
        unpatched_attention_paths=unpatched_attention_paths,
        pipeline_notes=pipeline_notes,
        _self_attn_paths=attn_paths,
    )


def _discover_diffsynth_ltx2_model(pipe: Any, *, infer_model_key: Callable) -> Any:
    module = getattr(pipe, "dit", None)
    if module is None:
        raise ValueError(
            "DiffSynth LTX2AudioVideoPipeline is detected but pipe.dit is not loaded. "
            "Load the LTX2 DiT before calling sparsevideo.apply()."
        )
    attn_paths = _enumerate_diffsynth_ltx2_module("dit", module)
    if not attn_paths:
        raise ValueError(
            "DiffSynth LTX2AudioVideoPipeline is detected but no supported loaded "
            "LTX2 video self-attention modules were found."
        )

    from ._model_info import ModelInfo

    model_key = getattr(pipe, "_sparsevideo_model_key", None) or infer_model_key(pipe, [module], "ltx_video")
    unpatched_attention_paths, pipeline_notes = _diffsynth_unpatched_aux_attention_status(
        pipe,
        [("dit", module)],
    )
    return ModelInfo(
        model_type="ltx_video",
        transformers=[module],
        model_key=model_key,
        pipeline_backend="diffsynth",
        unpatched_attention_paths=unpatched_attention_paths,
        pipeline_notes=pipeline_notes,
        _self_attn_paths=attn_paths,
    )


def install_diffsynth_sparse_attention(
    pipe: Any,
    model_info: Any,
    method_instance: Any,
) -> Tuple[List[Callable[[], None]], List[str]]:
    if getattr(pipe, "use_unified_sequence_parallel", False):
        raise NotImplementedError(
            "DiffSynth unified sequence parallel is already patching self_attn.forward; "
            "SparseVideo's DiffSynth backend currently requires the standard unmodified "
            "DiffSynth attention forward path."
        )

    step_tracker = StepTracker(
        model_type=model_info.model_type,
        num_inference_steps_fn=lambda: _pipeline_num_inference_steps(pipe),
    )
    restore_callbacks: List[Callable[[], None]] = []
    model_fn_restore = install_diffsynth_model_fn_tracker(pipe, step_tracker)
    if model_fn_restore is not None:
        restore_callbacks.append(model_fn_restore)
    try:
        install_model_patches = getattr(method_instance, "install_model_patches", None)
        if callable(install_model_patches):
            restore_callbacks.extend(install_model_patches(model_info))
        patched_attention_paths, attention_restore_callbacks = install_diffsynth_attention_forwards(
            model_info,
            method_instance,
            step_tracker,
        )
        restore_callbacks.extend(attention_restore_callbacks)
    except Exception:
        for restore_callback in reversed(restore_callbacks):
            restore_callback()
        raise
    return restore_callbacks, patched_attention_paths


def apply_diffsynth_sparse_attention(
    pipe: Any,
    model_info: Any,
    method_instance: Any,
    *,
    handle_cls: Callable,
    set_active_handle: Callable,
):
    restore_callbacks, patched_attention_paths = install_diffsynth_sparse_attention(
        pipe,
        model_info,
        method_instance,
    )

    handle = handle_cls(
        model_info=model_info,
        original_processors={},
        step_tracker_hooks=[],
        restore_callbacks=restore_callbacks,
        patched_attention_paths=patched_attention_paths,
        method_instance=method_instance,
        pipe=pipe,
    )
    set_active_handle(pipe, handle)
    return handle


def install_diffsynth_model_fn_tracker(pipe: Any, step_tracker: StepTracker):
    model_fn = getattr(pipe, "model_fn", None)
    if not callable(model_fn):
        return None

    signature = _safe_signature(model_fn)
    had_instance_attr = "model_fn" in getattr(pipe, "__dict__", {})
    original_instance_attr = getattr(pipe, "__dict__", {}).get("model_fn")

    def wrapped_model_fn(*args, **kwargs):
        timestep = _extract_bound_timestep(signature, args, kwargs)
        if timestep is not None:
            step_tracker.observe_timestep(timestep)
        step_tracker.seq_shape = _diffsynth_wan_seq_shape(signature, args, kwargs)
        return model_fn(*args, **kwargs)

    pipe.model_fn = wrapped_model_fn

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        if had_instance_attr:
            pipe.model_fn = original_instance_attr
        else:
            try:
                delattr(pipe, "model_fn")
            except AttributeError:
                pass

    return restore


def _pipeline_num_inference_steps(pipe: Any) -> int | None:
    scheduler = getattr(pipe, "scheduler", None)
    timesteps = getattr(scheduler, "timesteps", None)
    if timesteps is None:
        return None
    try:
        return len(timesteps)
    except TypeError:
        return None


def install_diffsynth_attention_forwards(
    model_info: Any,
    method_instance: Any,
    step_tracker: StepTracker,
) -> Tuple[List[str], List[Callable[[], None]]]:
    patched_paths: List[str] = []
    restore_callbacks: List[Callable[[], None]] = []
    try:
        for layer_idx, (path, attn_module) in enumerate(model_info.iter_self_attn_modules()):
            processor = method_instance.create_processor(
                layer_idx=layer_idx,
                total_layers=model_info.num_self_attn_layers,
                original_processor=None,
                step_tracker=step_tracker,
            )
            attn_fn = getattr(processor, "attn_fn", None)
            if not callable(attn_fn):
                raise TypeError(
                    f"{type(method_instance).__name__} did not expose an attn_fn for DiffSynth path {path}"
                )
            restore_callbacks.append(_patch_diffsynth_attention_forward(attn_module, attn_fn, step_tracker, path))
            patched_paths.append(path)
    except Exception:
        for restore_callback in reversed(restore_callbacks):
            restore_callback()
        raise
    return patched_paths, restore_callbacks


def _looks_like_diffsynth_pipeline(pipe) -> bool:
    cls = type(pipe)
    cls_name = cls.__name__
    module_name = getattr(cls, "__module__", "")
    if module_name in {
        "diffsynth.pipelines.wan_video",
        "diffsynth.pipelines.mova_audio_video",
        "diffsynth.pipelines.ltx2_audio_video",
    }:
        return True
    if cls_name in {"WanVideoPipeline", "MovaAudioVideoPipeline", "LTX2AudioVideoPipeline"}:
        return True
    if hasattr(pipe, "transformer"):
        return False
    return any(hasattr(pipe, name) for name in ("video_dit", "video_dit2"))


def _enumerate_diffsynth_wan_module(prefix: str, module) -> List[Tuple[str, Any]]:
    paths: List[Tuple[str, Any]] = []
    blocks = getattr(module, "blocks", None)
    if blocks is not None:
        for i, block in enumerate(blocks):
            attn = _diffsynth_wan_self_attn_module(block)
            if attn is not None:
                paths.append((f"{prefix}.blocks.{i}.self_attn.attn", attn))

    vace_blocks = getattr(module, "vace_blocks", None)
    if vace_blocks is not None:
        for i, block in enumerate(vace_blocks):
            attn = _diffsynth_wan_self_attn_module(block)
            if attn is not None:
                paths.append((f"{prefix}.vace_blocks.{i}.self_attn.attn", attn))
    return paths


def _enumerate_diffsynth_longcat_module(prefix: str, module) -> List[Tuple[str, Any]]:
    paths: List[Tuple[str, Any]] = []
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        return paths
    for i, block in enumerate(blocks):
        attn = _diffsynth_longcat_self_attn_module(block)
        if attn is not None:
            paths.append((f"{prefix}.blocks.{i}.attn._process_attn", attn))
    return paths


def _enumerate_diffsynth_ltx2_module(prefix: str, module) -> List[Tuple[str, Any]]:
    paths: List[Tuple[str, Any]] = []
    blocks = getattr(module, "transformer_blocks", None)
    if blocks is None:
        return paths
    for i, block in enumerate(blocks):
        attn = _diffsynth_ltx2_video_self_attn_module(block)
        if attn is not None:
            paths.append((f"{prefix}.transformer_blocks.{i}.attn1", attn))
    return paths


def _diffsynth_wan_self_attn_module(block):
    self_attn = getattr(block, "self_attn", None)
    attn = getattr(self_attn, "attn", None)
    if attn is None:
        return None
    if not callable(getattr(attn, "forward", None)):
        return None
    if not isinstance(getattr(attn, "num_heads", None), int):
        return None
    return attn


def _diffsynth_longcat_self_attn_module(block):
    attn = getattr(block, "attn", None)
    if attn is None:
        return None
    if not callable(getattr(attn, "_process_attn", None)):
        return None
    if not isinstance(getattr(attn, "num_heads", None), int):
        return None
    return attn


def _diffsynth_ltx2_video_self_attn_module(block):
    attn = getattr(block, "attn1", None)
    if attn is None:
        return None
    if not callable(getattr(attn, "forward", None)):
        return None
    if not isinstance(getattr(attn, "heads", None), int):
        return None
    if not isinstance(getattr(attn, "dim_head", None), int):
        return None
    for name in ("to_q", "to_k", "to_v", "q_norm", "k_norm", "to_out"):
        if not hasattr(attn, name):
            return None
    return attn


def _is_diffsynth_longcat_module(module) -> bool:
    cls = type(module)
    return (
        cls.__name__ == "LongCatVideoTransformer3DModel"
        or getattr(cls, "__module__", "") == "diffsynth.models.longcat_video_dit"
    )


def _is_diffsynth_ltx2_module(module) -> bool:
    cls = type(module)
    return (
        cls.__name__ == "LTXModel"
        or getattr(cls, "__module__", "") == "diffsynth.models.ltx2_dit"
        or hasattr(module, "transformer_blocks")
    )


def _is_diffsynth_s2v_module(module) -> bool:
    cls = type(module)
    return (
        cls.__name__ == "WanS2VModel"
        or getattr(cls, "__module__", "") == "diffsynth.models.wan_video_dit_s2v"
    )


def _diffsynth_unpatched_aux_attention_status(
    pipe: Any,
    loaded_modules: List[Tuple[str, Any]],
) -> Tuple[List[str], List[str]]:
    unpatched_paths: List[str] = []
    notes: List[str] = []

    def add(path: str, note: str) -> None:
        if path not in unpatched_paths:
            unpatched_paths.append(path)
        if note not in notes:
            notes.append(note)

    for attr_name, module in loaded_modules:
        if _is_diffsynth_s2v_module(module) and getattr(module, "audio_injector", None) is not None:
            add(
                f"{attr_name}.audio_injector.injector.*.attn",
                "DiffSynth S2V audio-injector cross-attention is loaded but not patched; "
                "only main Wan self-attention is sparse.",
            )
        if _is_diffsynth_longcat_module(module):
            blocks = getattr(module, "blocks", None) or []
            has_cross_attn = any(
                callable(getattr(getattr(block, "cross_attn", None), "_process_cross_attn", None))
                for block in blocks
            )
            if has_cross_attn:
                add(
                    f"{attr_name}.blocks.*.cross_attn._process_cross_attn",
                    "DiffSynth LongCat cross-attention is loaded but not patched; "
                    "only LongCat video self-attention is sparse.",
                )
        if getattr(module, "music_injector", None) is not None:
            add(
                f"{attr_name}.music_injector.injector.*.attn",
                "DiffSynth WanToDance music-injector cross-attention is loaded but not patched; "
                "only main Wan self-attention is sparse.",
            )
        if getattr(module, "music_encoder", None) is not None:
            add(
                f"{attr_name}.music_encoder.*.self_attn",
                "DiffSynth WanToDance music-encoder self-attention is loaded but not patched; "
                "only main Wan self-attention is sparse.",
            )
        if _is_diffsynth_ltx2_module(module):
            blocks = getattr(module, "transformer_blocks", None) or []
            if any(hasattr(block, "attn2") for block in blocks):
                add(
                    f"{attr_name}.transformer_blocks.*.attn2",
                    "DiffSynth LTX2 text cross-attention is loaded but not patched; "
                    "only video self-attention is sparse.",
                )
            if any(hasattr(block, "audio_attn1") for block in blocks):
                add(
                    f"{attr_name}.transformer_blocks.*.audio_attn1",
                    "DiffSynth LTX2 audio self-attention is loaded but not patched; "
                    "only video self-attention is sparse.",
                )
            if any(hasattr(block, "audio_attn2") for block in blocks):
                add(
                    f"{attr_name}.transformer_blocks.*.audio_attn2",
                    "DiffSynth LTX2 audio text cross-attention is loaded but not patched; "
                    "only video self-attention is sparse.",
                )
            if any(hasattr(block, "audio_to_video_attn") for block in blocks):
                add(
                    f"{attr_name}.transformer_blocks.*.audio_to_video_attn",
                    "DiffSynth LTX2 audio-to-video cross-attention is loaded but not patched; "
                    "only video self-attention is sparse.",
                )
            if any(hasattr(block, "video_to_audio_attn") for block in blocks):
                add(
                    f"{attr_name}.transformer_blocks.*.video_to_audio_attn",
                    "DiffSynth LTX2 video-to-audio cross-attention is loaded but not patched; "
                    "only video self-attention is sparse.",
                )

    if getattr(pipe, "vap", None) is not None:
        add(
            "vap.MotWanAttentionBlock.flash_attention",
            "DiffSynth VAP/MotWanModel custom flash_attention is loaded but not patched; "
            "only main Wan self-attention is sparse.",
        )

    if getattr(pipe, "animate_adapter", None) is not None:
        add(
            "animate_adapter.scaled_dot_product_attention",
            "DiffSynth Animate adapter scaled-dot-product attention is loaded but not patched; "
            "only main Wan self-attention is sparse.",
        )

    if getattr(pipe, "audio_dit", None) is not None:
        add(
            "audio_dit.blocks.*.self_attn.attn",
            "DiffSynth MOVA audio DiT self-attention is loaded but not patched; "
            "SparseVideo currently targets the MOVA video DiT path.",
        )

    if getattr(pipe, "dual_tower_bridge", None) is not None:
        add(
            "dual_tower_bridge.*.attn",
            "DiffSynth MOVA dual-tower bridge attention is loaded but not patched; "
            "SparseVideo currently targets the MOVA video DiT path.",
        )

    return unpatched_paths, notes


def _patch_diffsynth_attention_forward(
    attn_module: Any,
    attn_fn: Callable,
    step_tracker: StepTracker,
    path: str,
):
    if path.endswith("._process_attn"):
        return _patch_diffsynth_longcat_attention_process(attn_module, attn_fn, step_tracker, path)
    if ".transformer_blocks." in path and path.endswith(".attn1"):
        return _patch_diffsynth_ltx2_attention_forward(attn_module, attn_fn, step_tracker, path)
    return _patch_diffsynth_wan_attention_forward(attn_module, attn_fn, step_tracker, path)


def _patch_diffsynth_wan_attention_forward(
    attn_module: Any,
    attn_fn: Callable,
    step_tracker: StepTracker,
    path: str,
):
    had_instance_forward = "forward" in getattr(attn_module, "__dict__", {})
    original_instance_forward = getattr(attn_module, "__dict__", {}).get("forward")

    def sparse_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        heads = int(getattr(self, "num_heads", 0))
        if heads <= 0:
            raise RuntimeError(f"DiffSynth attention module at {path} has invalid num_heads={heads}")
        query = _diffsynth_flat_qkv_to_sparsevideo(q, heads, path, "q")
        key = _diffsynth_flat_qkv_to_sparsevideo(k, heads, path, "k")
        value = _diffsynth_flat_qkv_to_sparsevideo(v, heads, path, "v")
        hidden_states = attn_fn(
            query,
            key,
            value,
            None,
            timestep=getattr(step_tracker, "timestep", None),
            cache_key_suffix=path,
            pipeline_backend="diffsynth",
            seq_shape=_diffsynth_wan_attention_seq_shape(step_tracker, query.shape[1]),
        )
        return hidden_states.flatten(2, 3).type_as(q)

    attn_module.forward = types.MethodType(sparse_forward, attn_module)

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        if had_instance_forward:
            attn_module.forward = original_instance_forward
        else:
            try:
                delattr(attn_module, "forward")
            except AttributeError:
                pass

    return restore


def _diffsynth_wan_attention_seq_shape(step_tracker, q_len):
    """Return the stashed Wan grid only when it matches the attention sequence.

    Guards against grids we did not model (vace, camera control, sequence parallel):
    a mismatched product falls back to None so the methods keep their current
    token-count heuristic instead of raising.
    """
    seq_shape = getattr(step_tracker, "seq_shape", None)
    if seq_shape is None or len(seq_shape) != 3:
        return None
    grid_f, grid_h, grid_w = seq_shape
    if grid_f * grid_h * grid_w != int(q_len):
        return None
    return seq_shape


def _patch_diffsynth_longcat_attention_process(
    attn_module: Any,
    attn_fn: Callable,
    step_tracker: StepTracker,
    path: str,
):
    had_instance_process = "_process_attn" in getattr(attn_module, "__dict__", {})
    original_instance_process = getattr(attn_module, "__dict__", {}).get("_process_attn")

    def sparse_process(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape):
        heads = int(getattr(self, "num_heads", 0))
        if heads <= 0:
            raise RuntimeError(f"DiffSynth LongCat attention module at {path} has invalid num_heads={heads}")
        query = _diffsynth_bhsd_qkv_to_sparsevideo(q, heads, path, "q")
        key = _diffsynth_bhsd_qkv_to_sparsevideo(k, heads, path, "k")
        value = _diffsynth_bhsd_qkv_to_sparsevideo(v, heads, path, "v")
        hidden_states = attn_fn(
            query,
            key,
            value,
            None,
            timestep=getattr(step_tracker, "timestep", None),
            cache_key_suffix=path,
            pipeline_backend="diffsynth",
            seq_shape=shape,
        )
        return hidden_states.transpose(1, 2).contiguous().type_as(q)

    attn_module._process_attn = types.MethodType(sparse_process, attn_module)

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        if had_instance_process:
            attn_module._process_attn = original_instance_process
        else:
            try:
                delattr(attn_module, "_process_attn")
            except AttributeError:
                pass

    return restore


def _patch_diffsynth_ltx2_attention_forward(
    attn_module: Any,
    attn_fn: Callable,
    step_tracker: StepTracker,
    path: str,
):
    had_instance_forward = "forward" in getattr(attn_module, "__dict__", {})
    original_instance_forward = getattr(attn_module, "__dict__", {}).get("forward")
    original_forward = attn_module.forward
    apply_rotary_emb = _function_global(original_forward, "apply_rotary_emb")

    def sparse_forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        if context is not None:
            raise RuntimeError(
                f"DiffSynth LTX2 attention module at {path} received cross-attention context; "
                "SparseVideo only patches LTX2 video self-attention."
            )
        heads = int(getattr(self, "heads", 0))
        dim_head = int(getattr(self, "dim_head", 0))
        if heads <= 0 or dim_head <= 0:
            raise RuntimeError(
                f"DiffSynth LTX2 attention module at {path} has invalid heads={heads}, dim_head={dim_head}"
            )

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            if apply_rotary_emb is None:
                raise RuntimeError(
                    f"DiffSynth LTX2 attention module at {path} needs apply_rotary_emb, "
                    "but it was not found in the original forward globals."
                )
            rope_type = getattr(self, "rope_type", None)
            q = apply_rotary_emb(q, pe, rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, rope_type)

        expected_channels = heads * dim_head
        if int(q.shape[-1]) != expected_channels:
            raise RuntimeError(
                f"DiffSynth LTX2 attention module at {path} expected q channels={expected_channels}, "
                f"got {int(q.shape[-1])}"
            )
        query = q.unflatten(-1, (heads, dim_head))
        key = k.unflatten(-1, (heads, dim_head))
        value = v.unflatten(-1, (heads, dim_head))
        hidden_states = attn_fn(
            query,
            key,
            value,
            mask,
            timestep=getattr(step_tracker, "timestep", None),
            cache_key_suffix=path,
            pipeline_backend="diffsynth",
        )
        out = hidden_states.flatten(2, 3).type_as(q)

        gate_logits_fn = getattr(self, "to_gate_logits", None)
        if gate_logits_fn is not None:
            gate_logits = gate_logits_fn(x)
            batch, tokens, _ = out.shape
            out = out.view(batch, tokens, heads, dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(batch, tokens, heads * dim_head)
        return self.to_out(out)

    attn_module.forward = types.MethodType(sparse_forward, attn_module)

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        if had_instance_forward:
            attn_module.forward = original_instance_forward
        else:
            try:
                delattr(attn_module, "forward")
            except AttributeError:
                pass

    return restore


def _diffsynth_flat_qkv_to_sparsevideo(
    tensor: torch.Tensor,
    heads: int,
    path: str,
    name: str,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise RuntimeError(
            f"DiffSynth attention module at {path} expected {name} with shape [batch, seq, channels], "
            f"got {tuple(tensor.shape)}"
        )
    channels = int(tensor.shape[-1])
    if channels % heads != 0:
        raise RuntimeError(
            f"DiffSynth attention module at {path} cannot split {name} channels={channels} "
            f"over num_heads={heads}"
        )
    return tensor.unflatten(-1, (heads, channels // heads))


def _diffsynth_bhsd_qkv_to_sparsevideo(
    tensor: torch.Tensor,
    heads: int,
    path: str,
    name: str,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise RuntimeError(
            f"DiffSynth LongCat attention module at {path} expected {name} with shape [batch, heads, seq, dim], "
            f"got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[1]) != heads:
        raise RuntimeError(
            f"DiffSynth LongCat attention module at {path} expected {name} heads={heads}, "
            f"got {int(tensor.shape[1])}"
        )
    return tensor.transpose(1, 2).contiguous()


def _safe_signature(fn: Callable):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _function_global(fn: Callable, name: str):
    function = getattr(fn, "__func__", fn)
    globals_dict = getattr(function, "__globals__", {})
    return globals_dict.get(name)


def _extract_bound_timestep(signature, args, kwargs):
    if "timestep" in kwargs:
        return kwargs["timestep"]
    if signature is None:
        return None
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    return bound.arguments.get("timestep")


def _diffsynth_wan_seq_shape(signature, args, kwargs):
    """Recover the (frames, height, width) patch grid seen by Wan self-attention.

    DiffSynth's ``model_fn_wan_video`` patchifies ``latents`` with ``dit.patch_size``
    and, for Fun reference models, prepends one reference frame to the token
    sequence (``f += 1``). The patched attention forward only receives q/k/v, so we
    stash the grid here for the sparse methods, which would otherwise mis-factor the
    token count (e.g. 34320 reference tokens guessed as 33x26x40 instead of 22x30x52).
    """
    bound = _bind_model_fn_arguments(signature, args, kwargs)
    if bound is None:
        return None
    dit = bound.get("dit")
    latents = bound.get("latents")
    patch = getattr(dit, "patch_size", None)
    if not torch.is_tensor(latents) or latents.ndim != 5 or patch is None or len(patch) != 3:
        return None
    frames, height, width = int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4])
    pf, ph, pw = int(patch[0]), int(patch[1]), int(patch[2])
    if pf <= 0 or ph <= 0 or pw <= 0 or frames % pf or height % ph or width % pw:
        return None
    grid_f, grid_h, grid_w = frames // pf, height // ph, width // pw
    if bound.get("reference_latents") is not None:
        grid_f += 1
    if grid_f <= 0 or grid_h <= 0 or grid_w <= 0:
        return None
    return (grid_f, grid_h, grid_w)


def _bind_model_fn_arguments(signature, args, kwargs):
    if signature is None:
        return dict(kwargs)
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    return dict(bound.arguments)
