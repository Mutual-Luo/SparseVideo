from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_frame_shape, infer_video_token_layout
from .._schedule import configured_dense_warmup_layer_count, configured_dense_warmup_requires_dense, runtime_num_inference_steps
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from . import config as method_config
from .ops import STA_SUPPORTED_SEQ_SHAPES, STA_TILE_SIZE
from .search import MaskSearchRecorder, head_losses, parse_windows, window_key


class STAMethod(SparseMethod):
    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def __init__(self, config, model_info):
        super().__init__(config=config, model_info=model_info)
        self._sta_mode = self.config.get("STA_mode", "STA_inference")
        if self._sta_mode not in ("STA_inference", "STA_searching"):
            raise NotImplementedError(
                "SparseVideo STA supports STA_inference in pipelines and STA_searching for mask calibration; "
                "use python -m sparsevideo.methods.sta.search tune for STA_tuning."
            )
        self._mask_strategy = None if self._sta_mode == "STA_searching" else _load_mask_strategy(self.config.get("mask_strategy_file_path"))
        self._mask_candidates = parse_windows(self.config.get("mask_candidates"))
        self._mask_search_recorder = None

    def _recorder(self):
        if self._mask_search_recorder is None:
            self._mask_search_recorder = MaskSearchRecorder(
                self.config.get("mask_search_output_dir", "result/sta_mask_search"),
                prompt_id=self.config.get("mask_search_prompt_id"),
                candidates=self._mask_candidates,
            )
        return self._mask_search_recorder

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in (
            "wan",
            "hunyuan_video",
            "cogvideox",
            "ltx_video",
            "allegro",
            "mochi",
            "easyanimate",
        ):
            raise NotImplementedError(f"sta not yet supported for {self.model_info.model_type}")

        tile_size = tuple(self.config["tile_size"])
        kernel_size = tuple(self.config["window_size"])
        seq_shape = self.config["seq_shape"]
        has_text = self.config["has_text"]
        model_type = self.model_info.model_type
        mask_strategy = self._mask_strategy
        dense_warmup_layer_count = configured_dense_warmup_layer_count(self.config, total_layers)

        def attn_fn(query, key, value, attention_mask, **kwargs):
            step_idx = max(0, getattr(step_tracker, "step", 1) - 1)
            if self._sta_mode == "STA_searching":
                reference = _sta_dense_attention(query, key, value, attention_mask)
                text_len = kwargs.get("text_len", 0)
                prompt_length = kwargs.get("prompt_length")
                l1_loss = {}
                l2_loss = {}
                for candidate in self._mask_candidates:
                    sparse = _sta_attention(
                        query, key, value,
                        tile_size=tile_size,
                        kernel_size=candidate,
                        model_type=model_type,
                        text_len=text_len,
                        prompt_length=prompt_length,
                        seq_shape=seq_shape,
                        has_text=has_text,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        mask_strategy=None,
                    )
                    l1, l2 = head_losses(reference, sparse)
                    key_name = window_key(candidate)
                    l1_loss[key_name] = l1
                    l2_loss[key_name] = l2
                self._recorder().record(
                    step=step_idx,
                    layer=layer_idx,
                    l1_loss=l1_loss,
                    l2_loss=l2_loss,
                )
                self.record_runtime_dispatch(
                    "search",
                    backend="dense_reference_with_sta_candidates",
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return reference
            if (
                layer_idx < dense_warmup_layer_count
                or configured_dense_warmup_requires_dense(
                    self.config,
                    runtime_num_inference_steps(step_tracker),
                    getattr(step_tracker, "step", None),
                )
            ):
                out = _sta_dense_attention(query, key, value, attention_mask)
                self.record_runtime_dispatch(
                    "dense",
                    backend=_sta_dense_backend_name(attention_mask),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            text_len = kwargs.get("text_len", 0)
            prompt_length = kwargs.get("prompt_length")
            out = _sta_attention(query, key, value, tile_size=tile_size,
                                 kernel_size=kernel_size,
                                 model_type=model_type, text_len=text_len,
                                 prompt_length=prompt_length,
                                 seq_shape=seq_shape, has_text=has_text,
                                 layer_idx=layer_idx,
                                 step_idx=step_idx,
                                 mask_strategy=mask_strategy)
            self.record_runtime_dispatch(
                "sparse",
                backend=_sta_backend_name(query),
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "hunyuan_video":
            return SparseHunyuanVideoAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "cogvideox":
            return SparseCogVideoXAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "ltx_video":
            return SparseLTXVideoAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "allegro":
            return SparseAllegroAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "mochi":
            return SparseMochiAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        return SparseEasyAnimateAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)


def _sta_backend_name(query):
    from . import ops

    try:
        capability = torch.cuda.get_device_capability(query.device)
    except Exception:
        capability = (0, 0)
    if query.is_cuda and ops.sta_fwd is not None and capability[0] >= 9:
        return "fastvideo_sta_h100"
    if query.is_cuda and capability[0] == 8:
        return "fastvideo_sta_a100_triton"
    return "fastvideo_sta_triton"


def _sta_dense_backend_name(attention_mask):
    if attention_mask is not None:
        return "diffusers_dispatch"
    return "torch_sdpa"


def _sta_dense_attention(query, key, value, attention_mask):
    if attention_mask is not None:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        return dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        dropout_p=0.0, is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _sta_attention(query, key, value, tile_size, kernel_size, model_type="wan",
                   text_len=0, prompt_length=None, seq_shape=None, has_text=True,
                   layer_idx=0, step_idx=0, mask_strategy=None):
    """Sliding Tile Attention using the SparseVideo-owned FastVideo STA path.

    query/key/value: [B, N, H, D]
    tile_size: (T_tile, H_tile, W_tile) — tile dimensions
    kernel_size: (kt, kh, kw) — neighborhood kernel in tile units
    """
    B, N, H, D = query.shape

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    vid_start = layout.vid_start
    video_len = layout.video_len

    if video_len <= 0:
        raise RuntimeError("sta sparse path could not find video tokens")

    try:
        T, spatial_h, spatial_w = _infer_video_shape(video_len, model_type=model_type, seq_shape=seq_shape)
        if T * spatial_h * spatial_w != video_len:
            raise ValueError("shape mismatch")
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"sta sparse path could not infer video shape from {video_len} tokens") from exc

    ts, hs, ws = tile_size

    # Pad canvas to tile-aligned
    T_pad = ceil(T / ts) * ts
    H_pad = ceil(spatial_h / hs) * hs
    W_pad = ceil(spatial_w / ws) * ws

    if not query.is_cuda:
        raise RuntimeError("sta sparse path requires CUDA; CPU fallback is disabled for fair inference benchmarking")

    if tile_size != STA_TILE_SIZE:
        raise RuntimeError(
            f"sta sparse path requires FastVideo STA tile_size={STA_TILE_SIZE}; got {tile_size}. "
            "SparseVideo does not silently run the non-upstream generalized STA fallback for parity runs."
        )

    return _sta_sparsevideo_fastvideo_path(
        query, key, value, B, N, H, D,
        vid_start, video_len, text_len, context_len,
        T, spatial_h, spatial_w,
        T_pad, H_pad, W_pad,
        tile_size, kernel_size, model_type, seq_shape, has_text,
        layer_idx, step_idx, mask_strategy, prompt_length,
    )


def _sta_sparsevideo_fastvideo_path(query, key, value, B, N, H, D,
                                    vid_start, video_len, text_len, context_len,
                                    T, spatial_h, spatial_w,
                                    T_pad, H_pad, W_pad,
                                    tile_size, kernel_size, model_type, seq_shape_override,
                                    has_text_config, layer_idx, step_idx, mask_strategy,
                                    prompt_length=None):
    """SparseVideo-owned port of FastVideo's sliding_tile_attention API.

    H100 C++ dispatch is used only if a SparseVideo-owned sta_h100 extension is
    built. Otherwise this uses the SparseVideo-owned Triton port of FastVideo's
    fallback kernel.
    Input layout: [B, H, S, D] (BHSD), matching FastVideo STA.
    window_size: list of (t, h, w) per head — use kernel_size for all heads.
    """
    from .ops import sliding_tile_attention as _sv_sta

    vid_end = vid_start + video_len

    seq_shape = f"{T_pad}x{H_pad}x{W_pad}"

    window_size = _sta_window_sizes(mask_strategy, step_idx, layer_idx, H, kernel_size)

    # Extract video tokens, reshape to [B, H, T_pad*H_pad*W_pad, D]
    q_vid = query[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3).contiguous()
    k_vid = key[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3).contiguous()
    v_vid = value[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3).contiguous()

    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        q_vid = _sta_pad_video_canvas(q_vid, (T, spatial_h, spatial_w), (T_pad, H_pad, W_pad))
        k_vid = _sta_pad_video_canvas(k_vid, (T, spatial_h, spatial_w), (T_pad, H_pad, W_pad))
        v_vid = _sta_pad_video_canvas(v_vid, (T, spatial_h, spatial_w), (T_pad, H_pad, W_pad))

    q_vid = _sta_tile_bhsd(q_vid, (T_pad, H_pad, W_pad), tile_size)
    k_vid = _sta_tile_bhsd(k_vid, (T_pad, H_pad, W_pad), tile_size)
    v_vid = _sta_tile_bhsd(v_vid, (T_pad, H_pad, W_pad), tile_size)

    img_seq_len = T_pad * H_pad * W_pad

    # Handle text/context tokens
    has_text = False
    fvk_text_len = 0
    hunyuan_sparse_text_len = 0
    hunyuan_dense_text_start = None
    if model_type == "hunyuan_video" and text_len > 0:
        text_start = N - text_len
        text_capacity = _sta_hunyuan_text_capacity(seq_shape, tile_size) if has_text_config else 0
        hunyuan_sparse_text_len = min(int(text_len), text_capacity)
        if hunyuan_sparse_text_len > 0:
            fvk_text_len = _sta_effective_text_length(prompt_length, hunyuan_sparse_text_len)
            text_stop = text_start + hunyuan_sparse_text_len
            q_text = query[:, text_start:text_stop, :, :].permute(0, 2, 1, 3).contiguous()
            k_text = key[:, text_start:text_stop, :, :].permute(0, 2, 1, 3).contiguous()
            v_text = value[:, text_start:text_stop, :, :].permute(0, 2, 1, 3).contiguous()
            q_in = torch.cat([q_vid, q_text], dim=2)
            k_in = torch.cat([k_vid, k_text], dim=2)
            v_in = torch.cat([v_vid, v_text], dim=2)
            has_text = True
        else:
            q_in, k_in, v_in = q_vid, k_vid, v_vid
        if hunyuan_sparse_text_len < int(text_len):
            hunyuan_dense_text_start = text_start + hunyuan_sparse_text_len
    elif model_type != "hunyuan_video" and context_len > 0:
        q_ctx = query[:, :context_len, :, :].permute(0, 2, 1, 3).contiguous()
        k_ctx = key[:, :context_len, :, :].permute(0, 2, 1, 3).contiguous()
        v_ctx = value[:, :context_len, :, :].permute(0, 2, 1, 3).contiguous()
        q_in = torch.cat([q_vid, q_ctx], dim=2)
        k_in = torch.cat([k_vid, k_ctx], dim=2)
        v_in = torch.cat([v_vid, v_ctx], dim=2)
        has_text = bool(has_text_config)
        fvk_text_len = context_len
    else:
        q_in, k_in, v_in = q_vid, k_vid, v_vid

    out = _sv_sta(q_in, k_in, v_in, window_size, fvk_text_len, has_text, seq_shape)

    # out: [B, H, img_seq_len (+ text), D]
    out_vid_tiled = out[:, :, :img_seq_len, :]
    out_vid_pad = _sta_untile_bhsd(out_vid_tiled, (T_pad, H_pad, W_pad), tile_size)
    out_vid_pad = out_vid_pad.view(B, H, T_pad, H_pad, W_pad, D)
    out_vid = out_vid_pad[:, :, :T, :spatial_h, :spatial_w, :].reshape(B, H, video_len, D)
    out_vid = out_vid.permute(0, 2, 1, 3)  # [B, video_len, H, D]
    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        out_vid = _sta_repair_padded_border_outputs(
            query,
            key,
            value,
            out_vid,
            vid_start,
            (T, spatial_h, spatial_w),
            (T_pad, H_pad, W_pad),
            tile_size,
        )

    if model_type == "hunyuan_video":
        if text_len > 0:
            text_parts = []
            if hunyuan_sparse_text_len > 0:
                out_text = out[:, :, img_seq_len:img_seq_len + hunyuan_sparse_text_len, :].permute(0, 2, 1, 3)
                text_parts.append(out_text)
            if hunyuan_dense_text_start is not None:
                q_text_tail = query[:, hunyuan_dense_text_start:N, :, :].permute(0, 2, 1, 3)
                k_all = key.permute(0, 2, 1, 3)
                v_all = value.permute(0, 2, 1, 3)
                out_text_tail = F.scaled_dot_product_attention(q_text_tail, k_all, v_all, dropout_p=0.0)
                text_parts.append(out_text_tail.permute(0, 2, 1, 3))
            return torch.cat([out_vid, *text_parts], dim=1)
        return out_vid
    else:
        parts = []
        if context_len > 0:
            out_ctx = out[:, :, img_seq_len:, :].permute(0, 2, 1, 3)
            parts.append(out_ctx)
        parts.append(out_vid)
        remaining = N - context_len - video_len
        if remaining > 0:
            q_rem = query[:, vid_start + video_len:, :, :].permute(0, 2, 1, 3)
            k_all = key.permute(0, 2, 1, 3)
            v_all = value.permute(0, 2, 1, 3)
            out_rem = F.scaled_dot_product_attention(q_rem, k_all, v_all, dropout_p=0.0)
            parts.append(out_rem.permute(0, 2, 1, 3))
        return torch.cat(parts, dim=1)


def _sta_effective_text_length(prompt_length, text_len):
    if prompt_length is None:
        return int(text_len)
    return max(0, min(int(prompt_length), int(text_len)))


def _sta_hunyuan_text_capacity(seq_shape, tile_size):
    if str(seq_shape).lower() == "30x48x80":
        return 256
    return max(0, int(tile_size[0]) * int(tile_size[1]) * int(tile_size[2]))


def _infer_video_shape(video_len, model_type="wan", seq_shape=None):
    """Infer (T, H, W) from video token count."""
    return infer_video_frame_shape(video_len, model_type=model_type, seq_shape=seq_shape)


def _sta_pad_video_canvas(x: torch.Tensor, source_shape, target_shape):
    source_t, source_h, source_w = source_shape
    target_t, target_h, target_w = target_shape
    batch, heads, seq_len, dim = x.shape
    if seq_len != source_t * source_h * source_w:
        raise ValueError("STA pad input sequence length does not match source shape")
    if target_t < source_t or target_h < source_h or target_w < source_w:
        raise ValueError("STA target canvas must not be smaller than source shape")
    x = x.view(batch, heads, source_t, source_h, source_w, dim)
    if target_t > source_t:
        x = torch.cat([x, x[:, :, -1:, :, :, :].expand(-1, -1, target_t - source_t, -1, -1, -1)], dim=2)
    if target_h > source_h:
        x = torch.cat([x, x[:, :, :, -1:, :, :].expand(-1, -1, -1, target_h - source_h, -1, -1)], dim=3)
    if target_w > source_w:
        x = torch.cat([x, x[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, target_w - source_w, -1)], dim=4)
    return x.reshape(batch, heads, target_t * target_h * target_w, dim).contiguous()


def _sta_repair_padded_border_outputs(query, key, value, out_vid, vid_start, source_shape, target_shape, tile_size):
    repair_idx = _sta_padded_border_indices(source_shape, target_shape, tile_size, query.device)
    if repair_idx.numel() == 0:
        return out_vid

    repaired = out_vid.clone()
    k_all = key.permute(0, 2, 1, 3).contiguous()
    v_all = value.permute(0, 2, 1, 3).contiguous()
    for chunk in repair_idx.split(1024):
        q_chunk = query[:, vid_start + chunk, :, :].permute(0, 2, 1, 3).contiguous()
        out_chunk = F.scaled_dot_product_attention(q_chunk, k_all, v_all, dropout_p=0.0)
        repaired[:, chunk, :, :] = out_chunk.permute(0, 2, 1, 3)
    return repaired


def _sta_padded_border_indices(source_shape, target_shape, tile_size, device=None):
    source_t, source_h, source_w = source_shape
    target_t, target_h, target_w = target_shape
    tile_t, tile_h, tile_w = tile_size
    mask = torch.zeros(source_t, source_h, source_w, dtype=torch.bool, device=device)
    if target_t > source_t and source_t % tile_t:
        mask[(source_t // tile_t) * tile_t :, :, :] = True
    if target_h > source_h and source_h % tile_h:
        mask[:, (source_h // tile_h) * tile_h :, :] = True
    if target_w > source_w and source_w % tile_w:
        mask[:, :, (source_w // tile_w) * tile_w :] = True
    return mask.flatten().nonzero(as_tuple=False).flatten()


def _is_supported_fastvideo_shape(shape: tuple[int, int, int]) -> bool:
    return shape in set(STA_SUPPORTED_SEQ_SHAPES.values())


def _sta_tile_bhsd(x: torch.Tensor, canvas_shape, tile_size):
    canvas_t, canvas_h, canvas_w = canvas_shape
    tile_t, tile_h, tile_w = tile_size
    batch, heads, seq_len, dim = x.shape
    if seq_len != canvas_t * canvas_h * canvas_w:
        raise ValueError("STA tile input sequence length does not match canvas shape")
    return (
        x.view(
            batch,
            heads,
            canvas_t // tile_t,
            tile_t,
            canvas_h // tile_h,
            tile_h,
            canvas_w // tile_w,
            tile_w,
            dim,
        )
        .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
        .reshape(batch, heads, seq_len, dim)
    )


def _sta_untile_bhsd(x: torch.Tensor, canvas_shape, tile_size):
    canvas_t, canvas_h, canvas_w = canvas_shape
    tile_t, tile_h, tile_w = tile_size
    batch, heads, seq_len, dim = x.shape
    if seq_len != canvas_t * canvas_h * canvas_w:
        raise ValueError("STA untile input sequence length does not match canvas shape")
    return (
        x.view(
            batch,
            heads,
            canvas_t // tile_t,
            canvas_h // tile_h,
            canvas_w // tile_w,
            tile_t,
            tile_h,
            tile_w,
            dim,
        )
        .permute(0, 1, 2, 5, 3, 6, 4, 7, 8)
        .reshape(batch, heads, seq_len, dim)
    )


def _load_mask_strategy(path):
    if path is None:
        return None
    strategy_path = Path(str(path)).expanduser()
    if not strategy_path.is_absolute():
        strategy_path = Path.cwd() / strategy_path
    if "training_free" in strategy_path.resolve().parts:
        raise RuntimeError(f"Refusing to load STA mask strategy from training_free path: {strategy_path}")
    with strategy_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"STA mask strategy must be a dict, got {type(data).__name__}")
    return {str(key): _strategy_triple(value, f"mask_strategy[{key!r}]") for key, value in data.items()}


def _sta_window_sizes(mask_strategy, step_idx, layer_idx, num_heads, default_window):
    default = _strategy_triple(default_window, "window_size")
    if mask_strategy is None:
        return [default] * num_heads
    windows = []
    for head_idx in range(num_heads):
        windows.append(mask_strategy.get(f"{int(step_idx)}_{int(layer_idx)}_{head_idx}", default))
    return windows


def _strategy_triple(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    return tuple(int(part) for part in value)
