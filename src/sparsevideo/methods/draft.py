from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class DraftMethod(SparseMethod):
    """Draft Attention: 2D pooling guidance for sparse attention.

    Computes low-resolution attention via avg-pooling to build a block-sparse
    mask, then executes full-resolution attention with that mask.

    Port of: training_free/draft-attention/draft_attention.py
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "pool_h": 8,
        "pool_w": 16,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"draft not yet supported for {self.model_info.model_type}")

        cfg = self.config
        skip_steps = cfg["skip_first_steps"]
        skip_layers = cfg["skip_first_layers"]

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
            return _draft_attention(
                query, key, value,
                budget=cfg["budget"],
                pool_h=cfg["pool_h"],
                pool_w=cfg["pool_w"],
                model_type=model_type,
                text_len=kwargs.get("text_len", 0),
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _draft_attention(query, key, value, budget, pool_h, pool_w, model_type="wan", text_len=0):
    """Draft Attention: 2D pooling guidance for sparse video-video attention.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    if model_type == "hunyuan_video":
        context_len = 0
        video_len = N - text_len
    else:
        context_len = 226
        video_len = N - context_len

    if video_len <= 1:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    try:
        T, frame_h, frame_w = _infer_video_shape(video_len)
        if T * frame_h * frame_w != video_len:
            raise ValueError("shape mismatch")
    except (ValueError, RuntimeError):
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )
    frame_size = frame_h * frame_w
    video_end = context_len + T * frame_size

    q_vid = query[:, context_len:video_end, :, :]
    k_vid = key[:, context_len:video_end, :, :]

    # 2D avg pooling on Q and K per frame
    q_2d = q_vid.view(B, T, frame_h, frame_w, H, D).permute(0, 1, 4, 5, 2, 3)
    q_2d = q_2d.reshape(B * T, H * D, frame_h, frame_w)
    k_2d = k_vid.view(B, T, frame_h, frame_w, H, D).permute(0, 1, 4, 5, 2, 3)
    k_2d = k_2d.reshape(B * T, H * D, frame_h, frame_w)

    q_pooled = F.avg_pool2d(q_2d, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)
    k_pooled = F.avg_pool2d(k_2d, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)

    ph, pw = q_pooled.shape[-2], q_pooled.shape[-1]

    # [B, T*ph*pw, H, D]
    q_pooled = q_pooled.reshape(B, T, H, D, ph, pw).permute(0, 1, 4, 5, 2, 3)
    q_pooled = q_pooled.reshape(B, T * ph * pw, H, D)
    k_pooled = k_pooled.reshape(B, T, H, D, ph, pw).permute(0, 1, 4, 5, 2, 3)
    k_pooled = k_pooled.reshape(B, T * ph * pw, H, D)

    # Draft scores on pooled tokens: [B, H, S, S]
    S = q_pooled.shape[1]
    qp = q_pooled.permute(0, 2, 1, 3)
    kp = k_pooled.permute(0, 2, 1, 3)
    draft_scores = torch.matmul(qp, kp.transpose(-2, -1)) * scale
    draft_attn = F.softmax(draft_scores, dim=-1)

    # Top-budget block mask: [B, H, S, S]
    k_keep = max(1, int(S * budget))
    _, topk_idx = torch.topk(draft_attn, k=k_keep, dim=-1)
    draft_mask = torch.zeros(B, H, S, S, dtype=torch.bool, device=query.device)
    draft_mask.scatter_(dim=-1, index=topk_idx, value=True)

    # Map each video token to its pooled block index
    vid_idx = torch.arange(T * frame_size, device=query.device)
    vid_t = vid_idx // frame_size
    vid_spatial = vid_idx % frame_size
    vid_h = vid_spatial // frame_w
    vid_w = vid_spatial % frame_w
    pool_idx = vid_t * (ph * pw) + (vid_h // pool_h) * pw + (vid_w // pool_w)
    pool_idx = pool_idx.clamp(max=S - 1)

    # Expand block mask to full video resolution via gather
    # Row gather: [B, H, S, S] -> [B, H, video_len, S]
    row_idx = pool_idx.view(1, 1, -1, 1).expand(B, H, -1, S)
    mask_rows = torch.gather(draft_mask, 2, row_idx)
    # Col gather: [B, H, video_len, S] -> [B, H, video_len, video_len]
    col_idx = pool_idx.view(1, 1, 1, -1).expand(B, H, T * frame_size, -1)
    full_mask = torch.gather(mask_rows, 3, col_idx)

    # Video-video attention with sparse mask
    q_v = q_vid.permute(0, 2, 1, 3)
    k_v = k_vid.permute(0, 2, 1, 3)
    v_v = value[:, context_len:video_end, :, :].permute(0, 2, 1, 3)

    attn_bias = torch.zeros(B, H, T * frame_size, T * frame_size, device=query.device, dtype=query.dtype)
    attn_bias.masked_fill_(~full_mask, float('-inf'))

    out_vid = F.scaled_dot_product_attention(q_v, k_v, v_v, attn_mask=attn_bias, dropout_p=0.0)
    out_vid = out_vid.permute(0, 2, 1, 3)

    # Context tokens: dense attention to everything
    q_all = query.permute(0, 2, 1, 3)
    k_all = key.permute(0, 2, 1, 3)
    v_all = value.permute(0, 2, 1, 3)

    if context_len > 0:
        out_ctx = F.scaled_dot_product_attention(
            q_all[:, :, :context_len, :], k_all, v_all, dropout_p=0.0,
        )
        out_ctx = out_ctx.permute(0, 2, 1, 3)
        out = torch.cat([out_ctx, out_vid], dim=1)
    else:
        out = out_vid

    # Any remaining tokens after video
    remaining = N - video_end
    if remaining > 0:
        out_rem = F.scaled_dot_product_attention(
            q_all[:, :, video_end:, :], k_all, v_all, dropout_p=0.0,
        )
        out = torch.cat([out, out_rem.permute(0, 2, 1, 3)], dim=1)

    return out


def _infer_video_shape(video_len):
    candidates = (33, 25, 21, 17, 13, 9, 5, 3, 1)
    for T in candidates:
        if video_len % T == 0:
            spatial = video_len // T
            for h in range(int(spatial**0.5), 0, -1):
                if spatial % h == 0:
                    w = spatial // h
                    if 0.3 <= h / w <= 1.0:
                        return T, h, w
            h = int(spatial**0.5)
            if h > 0:
                return T, h, ceil(spatial / h)
    raise ValueError(f"Cannot infer video shape from {video_len} tokens")
