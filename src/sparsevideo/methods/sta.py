from __future__ import annotations

import importlib.util
from math import ceil
import warnings

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ._layout import infer_video_token_layout
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


_TRITON_FALLBACK_WARNED = False
_HAS_FASTVIDEO_KERNEL = None


def _has_fastvideo_kernel():
    global _HAS_FASTVIDEO_KERNEL
    if _HAS_FASTVIDEO_KERNEL is None:
        try:
            _HAS_FASTVIDEO_KERNEL = importlib.util.find_spec("fastvideo_kernel") is not None
        except Exception:
            _HAS_FASTVIDEO_KERNEL = False
    return _HAS_FASTVIDEO_KERNEL


def _dense_attention(query, key, value):
    return dispatch_attention_fn(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
    )


def _warn_triton_fallback(exc):
    global _TRITON_FALLBACK_WARNED
    if _TRITON_FALLBACK_WARNED:
        return
    msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    warnings.warn(
        f"sta Triton sparse path failed ({type(exc).__name__}: {msg}); "
        "falling back to dense attention.",
        RuntimeWarning,
        stacklevel=2,
    )
    _TRITON_FALLBACK_WARNED = True


class STAMethod(SparseMethod):
    CONFIG_DEFAULTS = {
        "tile_size": [4, 4, 4],
        "kernel_size": [3, 3, 3],
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"sta not yet supported for {self.model_info.model_type}")

        tile_size = tuple(self.config["tile_size"])
        kernel_size = tuple(self.config["kernel_size"])
        skip_steps = self.config["skip_first_steps"]
        skip_layers = self.config["skip_first_layers"]
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            use_sparse = (
                layer_idx >= skip_layers
                and step_tracker.step > skip_steps
            )
            if not use_sparse:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            text_len = kwargs.get("text_len", 0)
            return _sta_attention(query, key, value, tile_size=tile_size,
                                  kernel_size=kernel_size,
                                  model_type=model_type, text_len=text_len)

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _sta_attention(query, key, value, tile_size, kernel_size, model_type="wan", text_len=0):
    """Sliding Tile Attention with 3D neighborhood overlap via Triton kernel.

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
        return _dense_attention(query, key, value)

    try:
        T, spatial_h, spatial_w = _infer_video_shape(video_len)
        if T * spatial_h * spatial_w != video_len:
            raise ValueError("shape mismatch")
    except (ValueError, RuntimeError):
        return _dense_attention(query, key, value)

    ts, hs, ws = tile_size

    # Pad canvas to tile-aligned
    T_pad = ceil(T / ts) * ts
    H_pad = ceil(spatial_h / hs) * hs
    W_pad = ceil(spatial_w / ws) * ws

    # fastvideo_kernel path > Triton path (CUDA only)
    if query.is_cuda:
        if _has_fastvideo_kernel():
            try:
                return _sta_fastvideo_path(
                    query, key, value, B, N, H, D,
                    vid_start, video_len, text_len, context_len,
                    T, spatial_h, spatial_w,
                    T_pad, H_pad, W_pad,
                    tile_size, kernel_size, model_type,
                )
            except Exception as exc:
                _warn_triton_fallback(exc)
        try:
            return _sta_triton_path(
                query, key, value, B, N, H, D,
                vid_start, video_len, text_len, context_len,
                T, spatial_h, spatial_w,
                T_pad, H_pad, W_pad,
                tile_size, kernel_size, model_type,
            )
        except Exception as exc:
            _warn_triton_fallback(exc)
            return _dense_attention(query, key, value)

    # CPU fallback: non-overlapping tile SDPA
    return _sta_sdpa_fallback(
        query, key, value, B, N, H, D,
        vid_start, video_len, context_len,
        T, spatial_h, spatial_w,
        T_pad, H_pad, W_pad,
        tile_size, model_type, text_len,
    )


def _sta_triton_path(query, key, value, B, N, H, D,
                     vid_start, video_len, text_len, context_len,
                     T, spatial_h, spatial_w,
                     T_pad, H_pad, W_pad,
                     tile_size, kernel_size, model_type):
    """Triton kernel path: 3D neighborhood sliding tile attention."""
    from ..kernels.sta_triton import triton_sliding_tile_attention

    ts, hs, ws = tile_size

    # Extract video tokens and reshape to padded 3D grid
    vid_end = vid_start + video_len
    q_vid = query[:, vid_start:vid_end, :, :]
    k_vid = key[:, vid_start:vid_end, :, :]
    v_vid = value[:, vid_start:vid_end, :, :]

    # Reshape to [B, T, H_s, W_s, Heads, D] then pad
    q_3d = q_vid.view(B, T, spatial_h, spatial_w, H, D)
    k_3d = k_vid.view(B, T, spatial_h, spatial_w, H, D)
    v_3d = v_vid.view(B, T, spatial_h, spatial_w, H, D)

    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        q_3d = F.pad(q_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        k_3d = F.pad(k_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        v_3d = F.pad(v_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))

    img_seq_len = T_pad * H_pad * W_pad

    # Flatten to [B, img_seq_len, H, D] then fold to [B*H, seq_len, D]
    q_flat = q_3d.reshape(B, img_seq_len, H, D)
    k_flat = k_3d.reshape(B, img_seq_len, H, D)
    v_flat = v_3d.reshape(B, img_seq_len, H, D)

    # Determine text tokens for the Triton kernel
    if model_type == "hunyuan_video" and text_len > 0:
        # Text at end: append text tokens after video
        q_text = query[:, N - text_len:, :, :]
        k_text = key[:, N - text_len:, :, :]
        v_text = value[:, N - text_len:, :, :]
        q_flat = torch.cat([q_flat, q_text], dim=1)
        k_flat = torch.cat([k_flat, k_text], dim=1)
        v_flat = torch.cat([v_flat, v_text], dim=1)
        triton_text_len = text_len
    elif model_type != "hunyuan_video" and context_len > 0:
        # Non-video prefix tokens are appended after video for the STA kernel.
        q_ctx = query[:, :context_len, :, :]
        k_ctx = key[:, :context_len, :, :]
        v_ctx = value[:, :context_len, :, :]
        q_flat = torch.cat([q_flat, q_ctx], dim=1)
        k_flat = torch.cat([k_flat, k_ctx], dim=1)
        v_flat = torch.cat([v_flat, v_ctx], dim=1)
        triton_text_len = context_len
    else:
        triton_text_len = 0

    total_seq = q_flat.shape[1]

    # Fold to [B*H, seq_len, D]
    q_bh = q_flat.permute(0, 2, 1, 3).reshape(B * H, total_seq, D)
    k_bh = k_flat.permute(0, 2, 1, 3).reshape(B * H, total_seq, D)
    v_bh = v_flat.permute(0, 2, 1, 3).reshape(B * H, total_seq, D)

    out_bh = triton_sliding_tile_attention(
        q_bh, k_bh, v_bh,
        canvas_shape=(T_pad, H_pad, W_pad),
        tile_size=tile_size,
        kernel_size=kernel_size,
        text_length=triton_text_len,
    )

    # Unfold: [B*H, seq_len, D] -> [B, H, seq_len, D] -> [B, seq_len, H, D]
    out_full = out_bh.reshape(B, H, total_seq, D).permute(0, 2, 1, 3)

    # Extract video output (remove padding) and text output
    out_vid_padded = out_full[:, :img_seq_len, :, :]
    out_vid_3d = out_vid_padded.view(B, T_pad, H_pad, W_pad, H, D)
    out_vid_3d = out_vid_3d[:, :T, :spatial_h, :spatial_w, :, :]
    out_vid = out_vid_3d.reshape(B, video_len, H, D)

    if triton_text_len > 0:
        out_text = out_full[:, img_seq_len:img_seq_len + triton_text_len, :, :]

    # Reassemble in original token order
    if model_type == "hunyuan_video":
        if text_len > 0:
            return torch.cat([out_vid, out_text], dim=1)
        return out_vid
    else:
        # Non-Hunyuan layout: [optional prefix, video, optional tail]
        parts = []
        if context_len > 0:
            parts.append(out_text)  # context was appended as "text"
        parts.append(out_vid)
        remaining = N - context_len - video_len
        if remaining > 0:
            # Remaining tokens: dense attention
            q_rem = query[:, vid_start + video_len:, :, :].permute(0, 2, 1, 3)
            k_all = key.permute(0, 2, 1, 3)
            v_all = value.permute(0, 2, 1, 3)
            out_rem = F.scaled_dot_product_attention(q_rem, k_all, v_all, dropout_p=0.0)
            parts.append(out_rem.permute(0, 2, 1, 3))
        return torch.cat(parts, dim=1)


def _sta_fastvideo_path(query, key, value, B, N, H, D,
                        vid_start, video_len, text_len, context_len,
                        T, spatial_h, spatial_w,
                        T_pad, H_pad, W_pad,
                        tile_size, kernel_size, model_type):
    """fastvideo_kernel.sliding_tile_attention path.

    On SM90 (H100): uses ThunderKittens CUDA kernel (sta_fwd).
    On SM80 (A100): automatically falls back to Triton (sliding_tile_attention_triton).

    This matches the original FastVideo execution path exactly.
    Input layout: [B, H, S, D] (BHSD) as expected by fastvideo_kernel.
    window_size: list of (t, h, w) per head — use kernel_size for all heads.
    """
    from fastvideo_kernel import sliding_tile_attention as _fvk_sta

    ts, hs, ws = tile_size
    kt, kh, kw = kernel_size
    vid_end = vid_start + video_len

    # Determine 3D sequence shape string (nearest supported: 30x48x80, 36x48x48, 18x48x80)
    seq_shape = f"{T_pad}x{H_pad}x{W_pad}"
    _SUPPORTED_SHAPES = {"30x48x80", "36x48x48", "18x48x80"}

    # Build per-head window_size list using kernel_size
    # Each entry is (tiles_t, tiles_h, tiles_w) for that head
    nt = T_pad // ts
    nh = H_pad // hs
    nw = W_pad // ws
    window_size = [(kt, kh, kw)] * H

    # Extract video tokens, reshape to [B, H, T_pad*H_pad*W_pad, D]
    q_vid = query[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3)
    k_vid = key[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3)
    v_vid = value[:, vid_start:vid_end, :, :].permute(0, 2, 1, 3)

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

    img_seq_len = T_pad * H_pad * W_pad

    # Handle text/context tokens
    has_text = False
    fvk_text_len = 0
    if model_type == "hunyuan_video" and text_len > 0:
        q_text = query[:, N - text_len:, :, :].permute(0, 2, 1, 3)
        k_text = key[:, N - text_len:, :, :].permute(0, 2, 1, 3)
        v_text = value[:, N - text_len:, :, :].permute(0, 2, 1, 3)
        q_in = torch.cat([q_vid, q_text], dim=2)
        k_in = torch.cat([k_vid, k_text], dim=2)
        v_in = torch.cat([v_vid, v_text], dim=2)
        has_text = True
        fvk_text_len = text_len
    elif model_type != "hunyuan_video" and context_len > 0:
        q_ctx = query[:, :context_len, :, :].permute(0, 2, 1, 3)
        k_ctx = key[:, :context_len, :, :].permute(0, 2, 1, 3)
        v_ctx = value[:, :context_len, :, :].permute(0, 2, 1, 3)
        q_in = torch.cat([q_vid, q_ctx], dim=2)
        k_in = torch.cat([k_vid, k_ctx], dim=2)
        v_in = torch.cat([v_vid, v_ctx], dim=2)
        has_text = True
        fvk_text_len = context_len
    else:
        q_in, k_in, v_in = q_vid, k_vid, v_vid

    # fastvideo_kernel expects [B, H, S, D] BHSD layout
    # Only use named seq_shape when it's a supported value
    if seq_shape in _SUPPORTED_SHAPES:
        out = _fvk_sta(q_in, k_in, v_in, window_size, fvk_text_len, has_text, seq_shape)
    else:
        # Unsupported shape — fall back to triton path
        from ..kernels.sta_triton import triton_sliding_tile_attention
        q_bh = q_in.reshape(B * H, -1, D)
        k_bh = k_in.reshape(B * H, -1, D)
        v_bh = v_in.reshape(B * H, -1, D)
        out_bh = triton_sliding_tile_attention(
            q_bh, k_bh, v_bh,
            canvas_shape=(T_pad, H_pad, W_pad),
            tile_size=tile_size,
            kernel_size=kernel_size,
            text_length=fvk_text_len,
        )
        out = out_bh.reshape(B, H, -1, D)

    # out: [B, H, img_seq_len (+ text), D]
    out_vid_pad = out[:, :, :img_seq_len, :].view(B, H, T_pad, H_pad, W_pad, D)
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


def _sta_sdpa_fallback(query, key, value, B, N, H, D,
                       vid_start, video_len, context_len,
                       T, spatial_h, spatial_w,
                       T_pad, H_pad, W_pad,
                       tile_size, model_type, text_len):
    """CPU fallback: non-overlapping tile SDPA (original approach)."""
    ts, hs, ws = tile_size
    tile_tokens = ts * hs * ws

    vid_end = vid_start + T * spatial_h * spatial_w
    q_vid = query[:, vid_start:vid_end, :, :]
    k_vid = key[:, vid_start:vid_end, :, :]
    v_vid = value[:, vid_start:vid_end, :, :]

    q_3d = q_vid.view(B, T, spatial_h, spatial_w, H, D)
    k_3d = k_vid.view(B, T, spatial_h, spatial_w, H, D)
    v_3d = v_vid.view(B, T, spatial_h, spatial_w, H, D)

    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        q_3d = F.pad(q_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        k_3d = F.pad(k_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        v_3d = F.pad(v_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))

    nT, nH, nW = T_pad // ts, H_pad // hs, W_pad // ws
    q_tiles = q_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)
    k_tiles = k_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)
    v_tiles = v_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)

    q_tiles = q_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)
    k_tiles = k_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)
    v_tiles = v_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)

    num_tiles = nT * nH * nW
    q_flat = q_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)
    k_flat = k_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)
    v_flat = v_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)

    out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=0.0)

    out_tiles = out_flat.permute(0, 2, 1, 3).reshape(B, nT, nH, nW, ts, hs, ws, H, D)
    out_3d = out_tiles.permute(0, 1, 4, 2, 5, 3, 6, 7, 8).reshape(B, T_pad, H_pad, W_pad, H, D)
    out_3d = out_3d[:, :T, :spatial_h, :spatial_w, :, :]
    out_vid = out_3d.reshape(B, T * spatial_h * spatial_w, H, D)

    k_all = key.permute(0, 2, 1, 3)
    v_all = value.permute(0, 2, 1, 3)

    if model_type == "hunyuan_video":
        parts = [out_vid]
        if text_len > 0:
            q_ctx = query[:, N - text_len:, :, :].permute(0, 2, 1, 3)
            out_ctx = F.scaled_dot_product_attention(q_ctx, k_all, v_all, dropout_p=0.0)
            parts.append(out_ctx.permute(0, 2, 1, 3))
        return torch.cat(parts, dim=1)
    else:
        if context_len > 0:
            q_ctx = query[:, :vid_start, :, :].permute(0, 2, 1, 3)
            out_ctx = F.scaled_dot_product_attention(q_ctx, k_all, v_all, dropout_p=0.0)
            out = torch.cat([out_ctx.permute(0, 2, 1, 3), out_vid], dim=1)
        else:
            out = out_vid
        remaining = N - vid_start - T * spatial_h * spatial_w
        if remaining > 0:
            q_rem = query[:, vid_end:, :, :].permute(0, 2, 1, 3)
            out_rem = F.scaled_dot_product_attention(q_rem, k_all, v_all, dropout_p=0.0)
            out = torch.cat([out, out_rem.permute(0, 2, 1, 3)], dim=1)
        return out


def _infer_video_shape(video_len):
    """Infer (T, H, W) from video token count."""
    for T in (33, 25, 21, 17, 13, 9, 5):
        if video_len % T == 0:
            spatial = video_len // T
            for h in range(int(spatial**0.5), 0, -1):
                if spatial % h == 0:
                    w = spatial // h
                    if 0.3 <= h / w <= 1.0:
                        return T, h, w
            return T, int(spatial**0.5), ceil(spatial / int(spatial**0.5))
    T = 13
    spatial = video_len // T
    h = int(spatial ** 0.5)
    w = ceil(spatial / h)
    return T, h, w
