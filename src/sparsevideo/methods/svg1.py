from __future__ import annotations

from math import ceil

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ._layout import infer_video_token_layout
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class SVG1Method(SparseMethod):
    """SVG1: Sparse VideoGen stripe-based attention with online MSE profiling.

    Port of: training_free/Sparse-VideoGen/svg/models/wan/attention.py + utils.py
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "num_sampled_rows": 32,
        "multiplier": 2,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"svg1 not yet supported for {self.model_info.model_type}")

        cfg = self.config
        skip_steps = cfg["skip_first_steps"]
        skip_layers = cfg["skip_first_layers"]

        state = {"block_mask": None, "profiled_step": -1}
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
            try:
                return _svg_attention(
                    query, key, value,
                    multiplier=cfg["multiplier"],
                    num_sampled_rows=cfg["num_sampled_rows"],
                    state=state,
                    step_tracker_step=step_tracker.step,
                    model_type=model_type,
                    text_len=kwargs.get("text_len", 0),
                )
            except Exception:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _svg_attention(query, key, value, multiplier, num_sampled_rows, state,
                   step_tracker_step, model_type="wan", text_len=0):
    """SVG stripe-based sparse attention with per-head profiling.

    query/key/value: [B, N, H, D]
    """
    from torch.nn.attention.flex_attention import flex_attention

    B, N, H, D = query.shape
    scale = D ** -0.5

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    video_len = layout.video_len

    if video_len <= 0:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    frame_size = _estimate_frame_size(video_len)
    num_frames = video_len // frame_size
    video_end = context_len + num_frames * frame_size
    window_width = multiplier * frame_size

    if state["block_mask"] is None:
        head_choices = _profile_masks(
            query, key, value, scale, context_len, video_end,
            frame_size, num_frames, window_width, num_sampled_rows,
        )
        state["block_mask"] = _build_svg_block_mask(
            head_choices, N, H, context_len, video_end,
            frame_size, num_frames, window_width, query.device,
        )
        state["profiled_step"] = step_tracker_step

    bm = state["block_mask"]

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    out = flex_attention(q, k, v, block_mask=bm)
    return out.permute(0, 2, 1, 3)


def _profile_masks(query, key, value, scale, context_len, video_end,
                   frame_size, num_frames, window_width, num_sampled_rows):
    """Profile two mask candidates on sampled rows and select best per head.

    Returns head_choices: [H] tensor (0=sliding_window, 1=temporal_transpose).
    """
    B, N, H, D = query.shape
    device = query.device

    num_sample = min(num_sampled_rows, N)
    sampled_idx = torch.randint(0, N, (num_sample,), device=device)

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    sampled_q = q[:, :, sampled_idx, :]
    scores_sample = torch.matmul(sampled_q, k.transpose(-2, -1)) * scale

    weights_dense = F.softmax(scores_sample, dim=-1)
    out_dense = torch.matmul(weights_dense, v)

    all_idx = torch.arange(N, device=device)

    # Mask A: sliding window in frame-major order (spatial locality)
    mask_a = _sliding_window_mask_rows(sampled_idx, all_idx, context_len, video_end, frame_size, window_width)
    # Mask B: sliding window in spatial-major (temporal locality)
    mask_b = _temporal_transpose_mask_rows(sampled_idx, all_idx, context_len, video_end, frame_size, num_frames, window_width)

    mses = []
    for mask in [mask_a, mask_b]:
        masked_scores = scores_sample.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(masked_scores, dim=-1)
        out_sp = torch.matmul(w, v)
        mse = ((out_sp - out_dense) ** 2).mean(dim=(-2, -1))
        mses.append(mse)

    mses = torch.stack(mses, dim=0)
    head_choices = mses.argmin(dim=0).squeeze(0)
    return head_choices


def _sliding_window_mask_rows(q_idx, all_idx, context_len, video_end, frame_size, window_width):
    """Build sliding window mask for sampled query rows. [num_sample, N]"""
    q = q_idx.unsqueeze(1)
    k = all_idx.unsqueeze(0)
    is_context = (k < context_len) | (q >= video_end) | (k >= video_end) | (q < context_len)
    is_sink = (k >= context_len) & (k < context_len + frame_size)
    in_window = torch.abs(q - k) <= window_width
    return is_context | is_sink | in_window


def _temporal_transpose_mask_rows(q_idx, all_idx, context_len, video_end, frame_size, num_frames, window_width):
    """Build temporal transpose mask for sampled query rows. [num_sample, N]"""
    q = q_idx.unsqueeze(1)
    k = all_idx.unsqueeze(0)

    q_in_video = (q >= context_len) & (q < video_end)
    k_in_video = (k >= context_len) & (k < video_end)
    both_video = q_in_video & k_in_video

    q_vid = q - context_len
    k_vid = k - context_len
    q_frame = q_vid // frame_size
    q_spatial = q_vid % frame_size
    k_frame = k_vid // frame_size
    k_spatial = k_vid % frame_size

    q_reorder = q_spatial * num_frames + q_frame
    k_reorder = k_spatial * num_frames + k_frame

    in_window = torch.abs(q_reorder - k_reorder) <= window_width
    is_sink = (k >= context_len) & (k < context_len + frame_size)

    video_mask = is_sink | in_window
    return (~both_video) | video_mask


def _build_svg_block_mask(head_choices, N, H, context_len, video_end,
                          frame_size, num_frames, window_width, device):
    """Build per-head flex_attention block mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        q_in_video = (q_idx >= context_len) & (q_idx < video_end)
        kv_in_video = (kv_idx >= context_len) & (kv_idx < video_end)
        both_video = q_in_video & kv_in_video

        is_sink = (kv_idx >= context_len) & (kv_idx < context_len + frame_size)

        # Mask A: sliding window in original order
        sw = torch.abs(q_idx - kv_idx) <= window_width

        # Mask B: sliding window in spatial-major order
        q_vid = q_idx - context_len
        kv_vid = kv_idx - context_len
        q_reorder = (q_vid % frame_size) * num_frames + (q_vid // frame_size)
        kv_reorder = (kv_vid % frame_size) * num_frames + (kv_vid // frame_size)
        tt = torch.abs(q_reorder - kv_reorder) <= window_width

        choice = head_choices[h]
        video_mask = is_sink | torch.where(choice == 0, sw, tt)

        return (~both_video) | video_mask

    return create_block_mask(mask_mod, B=None, H=H, Q_LEN=N, KV_LEN=N, device=device)


def _estimate_frame_size(video_len):
    for nf in (33, 25, 21, 17, 13, 9, 5):
        if video_len % nf == 0:
            return video_len // nf
    return video_len // 13 if video_len >= 13 else video_len
