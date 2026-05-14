from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class STAMethod(SparseMethod):
    CONFIG_DEFAULTS = {
        "tile_size": [4, 4, 4],
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"sta not yet supported for {self.model_info.model_type}")

        tile_size = tuple(self.config["tile_size"])
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
                                  model_type=model_type, text_len=text_len)

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _sta_attention(query, key, value, tile_size, model_type="wan", text_len=0):
    """Sliding Tile Attention: compute attention independently within 3D tiles.

    query/key/value: [B, N, H, D]
    tile_size: (T_tile, H_tile, W_tile)
    """
    B, N, H, D = query.shape

    # Determine token layout: Wan has text at START, HunyuanVideo has text at END
    if model_type == "hunyuan_video":
        ctx_start = N - text_len if text_len > 0 else N
        vid_start = 0
        video_len = ctx_start
    else:
        context_len = 226
        vid_start = context_len
        ctx_start = 0
        video_len = N - context_len

    if video_len <= 0:
        return dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    try:
        T, spatial_h, spatial_w = _infer_video_shape(video_len)
        if T * spatial_h * spatial_w != video_len:
            raise ValueError("shape mismatch")
    except (ValueError, RuntimeError):
        return dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    ts, hs, ws = tile_size
    tile_tokens = ts * hs * ws

    # Extract video tokens
    vid_end = vid_start + T * spatial_h * spatial_w
    q_vid = query[:, vid_start:vid_end, :, :]
    k_vid = key[:, vid_start:vid_end, :, :]
    v_vid = value[:, vid_start:vid_end, :, :]

    # Reshape video tokens to [B, T, H_s, W_s, H_head, D]
    q_3d = q_vid.view(B, T, spatial_h, spatial_w, H, D)
    k_3d = k_vid.view(B, T, spatial_h, spatial_w, H, D)
    v_3d = v_vid.view(B, T, spatial_h, spatial_w, H, D)

    # Pad to tile-aligned sizes
    T_pad = ceil(T / ts) * ts
    H_pad = ceil(spatial_h / hs) * hs
    W_pad = ceil(spatial_w / ws) * ws

    if T_pad != T or H_pad != spatial_h or W_pad != spatial_w:
        q_3d = F.pad(q_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        k_3d = F.pad(k_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))
        v_3d = F.pad(v_3d, (0, 0, 0, 0, 0, W_pad - spatial_w, 0, H_pad - spatial_h, 0, T_pad - T))

    # Reshape into tiles: [B, T//ts, ts, H//hs, hs, W//ws, ws, Heads, D]
    nT, nH, nW = T_pad // ts, H_pad // hs, W_pad // ws
    q_tiles = q_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)
    k_tiles = k_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)
    v_tiles = v_3d.view(B, nT, ts, nH, hs, nW, ws, H, D)

    # Merge tile spatial dims: [B, num_tiles, ts*hs*ws, H, D]
    q_tiles = q_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)
    k_tiles = k_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)
    v_tiles = v_tiles.permute(0, 1, 3, 5, 2, 4, 6, 7, 8).reshape(B, nT * nH * nW, tile_tokens, H, D)

    # Compute attention per tile: [B*num_tiles, H, tile_tokens, D]
    num_tiles = nT * nH * nW
    q_flat = q_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)
    k_flat = k_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)
    v_flat = v_tiles.reshape(B * num_tiles, tile_tokens, H, D).permute(0, 2, 1, 3)

    out_flat = F.scaled_dot_product_attention(q_flat, k_flat, v_flat, dropout_p=0.0)

    # Reshape back
    out_tiles = out_flat.permute(0, 2, 1, 3).reshape(B, nT, nH, nW, ts, hs, ws, H, D)
    out_3d = out_tiles.permute(0, 1, 4, 2, 5, 3, 6, 7, 8).reshape(B, T_pad, H_pad, W_pad, H, D)

    # Remove padding
    out_3d = out_3d[:, :T, :spatial_h, :spatial_w, :, :]
    out_vid = out_3d.reshape(B, T * spatial_h * spatial_w, H, D)

    # Context tokens: use dense attention
    k_all = key.permute(0, 2, 1, 3)
    v_all = value.permute(0, 2, 1, 3)

    if model_type == "hunyuan_video":
        # Text tokens at end; video output is first, then dense for text
        parts = [out_vid]
        if text_len > 0:
            q_ctx = query[:, ctx_start:, :, :].permute(0, 2, 1, 3)
            out_ctx = F.scaled_dot_product_attention(q_ctx, k_all, v_all, dropout_p=0.0)
            parts.append(out_ctx.permute(0, 2, 1, 3))
        out = torch.cat(parts, dim=1)
    else:
        # Wan: text at start, video after
        q_ctx = query[:, :vid_start, :, :].permute(0, 2, 1, 3)
        out_ctx = F.scaled_dot_product_attention(q_ctx, k_all, v_all, dropout_p=0.0)
        out_ctx = out_ctx.permute(0, 2, 1, 3)
        out = torch.cat([out_ctx, out_vid], dim=1)
        # Handle any remaining tokens after video
        remaining = N - vid_start - T * spatial_h * spatial_w
        if remaining > 0:
            q_rem = query[:, vid_end:, :, :].permute(0, 2, 1, 3)
            out_rem = F.scaled_dot_product_attention(q_rem, k_all, v_all, dropout_p=0.0)
            out = torch.cat([out, out_rem.permute(0, 2, 1, 3)], dim=1)

    return out


def _infer_video_shape(video_len):
    """Infer (T, H, W) from video token count."""
    # Common Wan shapes: T * H * W
    # 480p: 13 * 30 * 45 = 17550, or 21 * 30 * 45, etc.
    # 720p: 13 * 48 * 80 = 49920
    # Try common spatial dimensions
    for T in (33, 25, 21, 17, 13, 9, 5):
        if video_len % T == 0:
            spatial = video_len // T
            # Try to factorize spatial into H * W (roughly 2:3 or 3:4 ratio)
            for h in range(int(spatial**0.5), 0, -1):
                if spatial % h == 0:
                    w = spatial // h
                    if 0.3 <= h / w <= 1.0:
                        return T, h, w
            return T, int(spatial**0.5), ceil(spatial / int(spatial**0.5))
    # Fallback
    T = 13
    spatial = video_len // T
    h = int(spatial ** 0.5)
    w = ceil(spatial / h)
    return T, h, w
