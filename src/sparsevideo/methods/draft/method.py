from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_token_layout
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class DraftMethod(SparseMethod):
    """Draft Attention: 2D pooling guidance for sparse attention.

    Computes low-resolution attention via avg-pooling to build a block-sparse
    mask, then executes full-resolution attention using Triton block-sparse kernel.

    Port of: training_free/draft-attention/draft_attention.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if not self.config["block_sparse_attention"]:
            raise NotImplementedError(
                "draft block_sparse_attention=False disables the upstream sparse path "
                "and is not exposed as a SparseVideo sparse method; use method='dense' "
                "for the dense baseline."
            )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"draft not yet supported for {self.model_info.model_type}")

        cfg = self.config

        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            return _draft_attention(
                query, key, value,
                sparsity_ratio=cfg["sparsity_ratio"],
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


def _draft_attention(query, key, value, sparsity_ratio, pool_h, pool_w,
                     model_type="wan", text_len=0):
    """Draft Attention: 2D pooling guidance → Triton block-sparse execution.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    video_len = layout.video_len

    if video_len <= 1:
        raise RuntimeError("draft sparse path could not find video tokens")

    try:
        T, frame_h, frame_w = _infer_video_shape(video_len)
        if T * frame_h * frame_w != video_len:
            raise ValueError("shape mismatch")
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f"draft sparse path could not infer video shape from {video_len} tokens") from exc
    frame_size = frame_h * frame_w
    video_end = context_len + T * frame_size

    # Use Triton block-sparse path if on CUDA
    if query.is_cuda:
        try:
            return _draft_triton_path(
                query, key, value, B, N, H, D, scale,
                context_len, video_end, T, frame_h, frame_w, frame_size,
                sparsity_ratio, pool_h, pool_w, model_type, text_len,
            )
        except Exception as exc:
            raise RuntimeError("draft Triton sparse path failed") from exc

    raise RuntimeError("draft sparse path requires CUDA")


def _draft_triton_path(query, key, value, B, N, H, D, scale,
                       context_len, video_end, T, frame_h, frame_w, frame_size,
                       sparsity_ratio, pool_h, pool_w, model_type, text_len):
    """Pool-guided block-sparse attention.

    Primary backend: flashinfer VariableBlockSparseAttentionWrapper.
    Fallback: Triton block_sparse_attention.
    """
    from ...kernels.block_sparse_attn import block_sparse_attention
    from ...kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    q_vid = query[:, context_len:video_end, :, :]
    k_vid = key[:, context_len:video_end, :, :]

    # 2D avg pooling on Q and K per frame to get pooled tokens
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

    # Upstream sparsity_ratio is the fraction to skip; keep the complement.
    keep_ratio = max(0.0, min(1.0, 1.0 - float(sparsity_ratio)))
    k_keep = max(1, int(S * keep_ratio))
    _, topk_idx = torch.topk(draft_attn, k=k_keep, dim=-1)
    draft_mask = torch.zeros(B, H, S, S, dtype=torch.bool, device=query.device)
    draft_mask.scatter_(dim=-1, index=topk_idx, value=True)

    # Map each video token to its pool-block index
    vid_idx = torch.arange(T * frame_size, device=query.device)
    vid_t = vid_idx // frame_size
    vid_spatial = vid_idx % frame_size
    vid_h = vid_spatial // frame_w
    vid_w = vid_spatial % frame_w
    pool_idx = vid_t * (ph * pw) + (vid_h // pool_h) * pw + (vid_w // pool_w)
    pool_idx = pool_idx.clamp(max=S - 1)

    # Fold B*H together
    video_tokens = T * frame_size
    q_flat = q_vid.permute(0, 2, 1, 3).reshape(B * H, video_tokens, D)
    k_flat = k_vid.permute(0, 2, 1, 3).reshape(B * H, video_tokens, D)
    v_vid = value[:, context_len:video_end, :, :]
    v_flat = v_vid.permute(0, 2, 1, 3).reshape(B * H, video_tokens, D)

    labels = pool_idx.unsqueeze(0).expand(B * H, -1)  # [B*H, video_tokens]

    # Cluster sizes (same for Q and K — same spatial partition)
    q_sizes = torch.zeros(B * H, S, dtype=torch.int32, device=query.device)
    q_sizes.scatter_add_(1, labels.to(torch.int32),
                         torch.ones(B * H, video_tokens, dtype=torch.int32, device=query.device))
    k_sizes = q_sizes.clone()

    # dynamic_map: [B*H, S, S]
    dynamic_map = draft_mask.reshape(B * H, S, S)

    # Sort tokens by pool-block label
    sorted_idx = labels.argsort(dim=-1)
    q_sorted = torch.gather(q_flat, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    k_sorted = torch.gather(k_flat, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, D))
    v_sorted = torch.gather(v_flat, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, D))

    # Block-sparse attention — flashinfer primary, Triton fallback
    if HAS_FLASHINFER:
        out_sorted = variable_block_sparse_attn(
            q_sorted, k_sorted, v_sorted,
            dynamic_map, q_sizes, k_sizes,
        )
    else:
        out_sorted = block_sparse_attention(
            q_sorted, k_sorted, v_sorted,
            q_sizes.to(torch.long), k_sizes.to(torch.long), dynamic_map, scale,
        )

    # Unsort
    inv_idx = sorted_idx.argsort(dim=-1)
    out_flat = torch.gather(out_sorted, 1, inv_idx.unsqueeze(-1).expand(-1, -1, D))
    out_vid = out_flat.reshape(B, H, video_tokens, D).permute(0, 2, 1, 3)

    # Context/text tokens: dense attention to everything
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
