from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class RadialMethod(SparseMethod):
    """Radial attention with logarithmic band decay per frame-pair distance.

    Port of: training_free/radial-attention/radial_attn/attn_mask.py
    """

    CONFIG_DEFAULTS = {
        "decay_factor": 0.5,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"radial not yet supported for {self.model_info.model_type}")

        decay_factor = self.config["decay_factor"]
        skip_steps = self.config["skip_first_steps"]
        skip_layers = self.config["skip_first_layers"]

        block_mask_cache = {}
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
            return _radial_attention(
                query, key, value,
                decay_factor=decay_factor,
                block_mask_cache=block_mask_cache,
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


def _radial_attention(query, key, value, decay_factor, block_mask_cache,
                      model_type="wan", text_len=0):
    """Radial attention via flex_attention with logarithmic band decay.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape

    if model_type == "hunyuan_video":
        context_len = 0
        video_len = N - text_len
    else:
        context_len = 226
        video_len = N - context_len

    if video_len <= 0:
        return dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
        )

    frame_size = _estimate_frame_size(video_len)
    num_frames = video_len // frame_size
    video_end = context_len + num_frames * frame_size

    cache_key = (N, context_len)
    if cache_key not in block_mask_cache:
        tpf_bits = frame_size.bit_length()

        def mask_mod(b, h, q_idx, kv_idx):
            q_in_video = (q_idx >= context_len) & (q_idx < video_end)
            kv_in_video = (kv_idx >= context_len) & (kv_idx < video_end)
            both_video = q_in_video & kv_in_video

            q_frame = (q_idx - context_len) // frame_size
            kv_frame = (kv_idx - context_len) // frame_size

            is_sink = (kv_idx >= context_len) & (kv_idx < context_len + frame_size)

            dist = torch.abs(q_frame - kv_frame)
            is_near = dist <= 1

            safe_dist = torch.clamp(dist, min=2)
            group = torch.floor(torch.log2(safe_dist.float())) + 1
            decay_length = (2.0 ** tpf_bits) / (2.0 ** group) * decay_factor
            window_width = torch.clamp(decay_length, min=128.0)

            decay_raw = (2.0 ** tpf_bits) / (2.0 ** group)
            below_thresh = decay_raw < 128.0
            split_factor = torch.where(
                below_thresh,
                torch.floor(128.0 / decay_raw).to(dist.dtype),
                torch.ones_like(dist),
            )
            split_ok = (dist % torch.clamp(split_factor, min=1)) == 0

            q_local = (q_idx - context_len) % frame_size
            kv_local = (kv_idx - context_len) % frame_size
            in_band = torch.abs(q_local - kv_local).float() <= window_width

            video_mask = is_sink | is_near | (in_band & split_ok)
            return (~both_video) | video_mask

        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=query.device)
        block_mask_cache[cache_key] = block_mask

    bm = block_mask_cache[cache_key]

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    out = flex_attention(q, k, v, block_mask=bm)
    return out.permute(0, 2, 1, 3)


def _estimate_frame_size(video_len):
    for nf in (33, 25, 21, 17, 13, 9, 5):
        if video_len % nf == 0:
            return video_len // nf
    return video_len // 13 if video_len >= 13 else video_len
