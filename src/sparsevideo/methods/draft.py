from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor


class DraftMethod(SparseMethod):
    """Draft Attention: 2D pooling guidance for sparse attention.

    Computes low-resolution attention via avg-pooling to build a block-sparse
    mask, then executes full-resolution attention via flex_attention.

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
        if self.model_info.model_type == "wan":
            cfg = self.config
            skip_steps = cfg["skip_first_steps"]
            skip_layers = cfg["skip_first_layers"]

            def attn_fn(query, key, value, attention_mask):
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
                )

            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        raise NotImplementedError(f"draft not yet supported for {self.model_info.model_type}")


def _draft_attention(query, key, value, budget, pool_h, pool_w):
    """Draft Attention: 2D pooling guidance → flex_attention.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    context_len = 226
    if N <= context_len:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    video_len = N - context_len
    T, frame_h, frame_w = _infer_video_shape(video_len)
    frame_size = frame_h * frame_w
    video_end = context_len + T * frame_size

    q_vid = query[:, context_len:video_end, :, :]
    k_vid = key[:, context_len:video_end, :, :]

    # 2D avg pooling on Q and K per frame
    # Reshape: [B, T, frame_h, frame_w, H, D] → [B*T, H*D, frame_h, frame_w]
    q_2d = q_vid.view(B, T, frame_h, frame_w, H, D).permute(0, 1, 4, 5, 2, 3)
    q_2d = q_2d.reshape(B * T, H * D, frame_h, frame_w)
    k_2d = k_vid.view(B, T, frame_h, frame_w, H, D).permute(0, 1, 4, 5, 2, 3)
    k_2d = k_2d.reshape(B * T, H * D, frame_h, frame_w)

    q_pooled = F.avg_pool2d(q_2d, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)
    k_pooled = F.avg_pool2d(k_2d, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)

    ph, pw = q_pooled.shape[-2], q_pooled.shape[-1]

    # Reshape back: [B, T*ph*pw, H, D]
    q_pooled = q_pooled.reshape(B, T, H, D, ph, pw).permute(0, 1, 4, 5, 2, 3)
    q_pooled = q_pooled.reshape(B, T * ph * pw, H, D)
    k_pooled = k_pooled.reshape(B, T, H, D, ph, pw).permute(0, 1, 4, 5, 2, 3)
    k_pooled = k_pooled.reshape(B, T * ph * pw, H, D)

    # Compute draft attention scores on pooled tokens
    S = q_pooled.shape[1]
    qp = q_pooled.permute(0, 2, 1, 3)
    kp = k_pooled.permute(0, 2, 1, 3)
    draft_scores = torch.matmul(qp, kp.transpose(-2, -1)) * scale
    draft_attn = F.softmax(draft_scores, dim=-1)

    # Keep top-budget fraction per row → block pattern [B, H, S, S]
    k_keep = max(1, int(S * budget))
    _, topk_idx = torch.topk(draft_attn, k=k_keep, dim=-1)
    draft_mask = torch.zeros(B, H, S, S, dtype=torch.bool, device=query.device)
    draft_mask.scatter_(dim=-1, index=topk_idx, value=True)

    # Build pooled-token-to-full-token mapping
    # Each pooled position (t, ph_i, pw_j) covers video tokens in a spatial block
    # Store the draft_mask as a tensor and use it in mask_mod
    # We need to map full video token indices to pooled indices
    # video_token_idx → (t, h, w) → pooled_idx = t * ph * pw + (h // pool_h) * pw + (w // pool_w)

    def mask_mod(b, h, q_idx, kv_idx):
        q_in_video = (q_idx >= context_len) & (q_idx < video_end)
        kv_in_video = (kv_idx >= context_len) & (kv_idx < video_end)
        both_video = q_in_video & kv_in_video

        # Map video tokens to pooled indices
        q_vid_pos = q_idx - context_len
        q_t = q_vid_pos // frame_size
        q_spatial = q_vid_pos % frame_size
        q_h = q_spatial // frame_w
        q_w = q_spatial % frame_w
        q_pool_idx = q_t * (ph * pw) + (q_h // pool_h) * pw + (q_w // pool_w)

        kv_vid_pos = kv_idx - context_len
        kv_t = kv_vid_pos // frame_size
        kv_spatial = kv_vid_pos % frame_size
        kv_h = kv_spatial // frame_w
        kv_w = kv_spatial % frame_w
        kv_pool_idx = kv_t * (ph * pw) + (kv_h // pool_h) * pw + (kv_w // pool_w)

        # Clamp to valid range
        q_pool_idx = torch.clamp(q_pool_idx, 0, S - 1)
        kv_pool_idx = torch.clamp(kv_pool_idx, 0, S - 1)

        # Check draft mask (use first batch element since mask varies by B)
        is_active = draft_mask[0, h, q_pool_idx, kv_pool_idx]

        return (~both_video) | is_active

    bm = create_block_mask(mask_mod, B=None, H=H, Q_LEN=N, KV_LEN=N, device=query.device)

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    out = flex_attention(q, k, v, block_mask=bm)
    return out.permute(0, 2, 1, 3)


def _infer_video_shape(video_len):
    for T in (33, 25, 21, 17, 13, 9, 5):
        if video_len % T == 0:
            spatial = video_len // T
            for h in range(int(spatial**0.5), 0, -1):
                if spatial % h == 0:
                    w = spatial // h
                    if 0.3 <= h / w <= 1.0:
                        return T, h, w
            h = int(spatial**0.5)
            return T, h, ceil(spatial / h)
    T = 13
    spatial = video_len // T
    h = int(spatial ** 0.5)
    return T, h, ceil(spatial / h)
