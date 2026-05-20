from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_frame_shape, infer_video_token_layout
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from . import config as method_config
from .ops import STA_SUPPORTED_SEQ_SHAPES, STA_TILE_SIZE


class STAMethod(SparseMethod):
    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def __init__(self, config, model_info):
        super().__init__(config=config, model_info=model_info)
        if self.config["STA_mode"] != "STA_inference":
            raise NotImplementedError("SparseVideo STA currently supports upstream STA_inference mode only")
        self._mask_strategy = _load_mask_strategy(self.config.get("mask_strategy_file_path"))

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

        def attn_fn(query, key, value, attention_mask, **kwargs):
            text_len = kwargs.get("text_len", 0)
            prompt_length = kwargs.get("prompt_length")
            out = _sta_attention(query, key, value, tile_size=tile_size,
                                 kernel_size=kernel_size,
                                 model_type=model_type, text_len=text_len,
                                 prompt_length=prompt_length,
                                 seq_shape=seq_shape, has_text=has_text,
                                 layer_idx=layer_idx,
                                 step_idx=max(0, step_tracker.step - 1),
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

    if model_type in ("wan", "hunyuan_video") and not _is_supported_fastvideo_shape((T_pad, H_pad, W_pad)):
        raise RuntimeError(
            "sta sparse path only supports FastVideo STA native padded seq_shape values "
            f"{sorted(STA_SUPPORTED_SEQ_SHAPES)}; got {T_pad}x{H_pad}x{W_pad}. "
            "Use the upstream profile/resolution/frame count or a different method."
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

    # Pad to T_pad, H_pad, W_pad
    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        q_vid = q_vid.view(B, H, T, spatial_h, spatial_w, D)
        q_vid = F.pad(q_vid, (0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        q_vid = q_vid.reshape(B, H, T_pad * H_pad * W_pad, D)
        k_vid = k_vid.view(B, H, T, spatial_h, spatial_w, D)
        k_vid = F.pad(k_vid, (0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        k_vid = k_vid.reshape(B, H, T_pad * H_pad * W_pad, D)
        v_vid = v_vid.view(B, H, T, spatial_h, spatial_w, D)
        v_vid = F.pad(v_vid, (0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        v_vid = v_vid.reshape(B, H, T_pad * H_pad * W_pad, D)

    q_vid = _sta_tile_bhsd(q_vid, (T_pad, H_pad, W_pad), tile_size)
    k_vid = _sta_tile_bhsd(k_vid, (T_pad, H_pad, W_pad), tile_size)
    v_vid = _sta_tile_bhsd(v_vid, (T_pad, H_pad, W_pad), tile_size)

    img_seq_len = T_pad * H_pad * W_pad

    # Handle text/context tokens
    has_text = False
    fvk_text_len = 0
    if model_type == "hunyuan_video" and text_len > 0:
        fvk_text_len = _sta_effective_text_length(prompt_length, text_len)
        q_text = query[:, N - text_len:, :, :].permute(0, 2, 1, 3).contiguous()
        k_text = key[:, N - text_len:, :, :].permute(0, 2, 1, 3).contiguous()
        v_text = value[:, N - text_len:, :, :].permute(0, 2, 1, 3).contiguous()
        q_in = torch.cat([q_vid, q_text], dim=2)
        k_in = torch.cat([k_vid, k_text], dim=2)
        v_in = torch.cat([v_vid, v_text], dim=2)
        has_text = bool(has_text_config)
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

    if model_type == "hunyuan_video":
        if text_len > 0:
            out_text = out[:, :, img_seq_len:, :].permute(0, 2, 1, 3)
            return torch.cat([out_vid, out_text], dim=1)
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


def _infer_video_shape(video_len, model_type="wan", seq_shape=None):
    """Infer (T, H, W) from video token count."""
    return infer_video_frame_shape(video_len, model_type=model_type, seq_shape=seq_shape)


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
