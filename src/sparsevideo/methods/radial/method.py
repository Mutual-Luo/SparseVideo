from __future__ import annotations

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from .._layout import infer_video_token_layout
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class RadialMethod(SparseMethod):
    """Radial attention with logarithmic band decay per frame-pair distance.

    Port of: training_free/radial-attention/radial_attn/attn_mask.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if self.config["use_sage_attention"]:
            raise NotImplementedError(
                "radial use_sage_attention is recognized from upstream but not ported in SparseVideo yet"
            )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"radial not yet supported for {self.model_info.model_type}")

        decay_factor = self.config["decay_factor"]
        dense_timesteps = self.config["dense_timesteps"]
        dense_layers = self.config["dense_layers"]

        block_mask_cache = {}
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            full_attention = (
                layer_idx < dense_layers
                or step_tracker.step <= dense_timesteps
            )
            if full_attention:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            if not query.is_cuda or attention_mask is not None:
                raise RuntimeError("radial sparse path requires CUDA self-attention without an attention mask")
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
    """Radial attention with logarithmic band decay per frame-pair distance.

    For HunyuanVideo (no context prefix, vid_len divisible by block_size):
        flashinfer variable-block-sparse with fixed block_size ≤ 128.
        Matches original radial-attention FlashInferBackend block granularity.
        Text tokens are appended and always fully attended via dense SDPA.

    For WAN:
        SparseVideo patches attn1 self-attention, so the sequence contains
        video tokens only.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    video_len = layout.video_len
    tail_len = layout.tail_len

    if video_len <= 0:
        raise RuntimeError("radial sparse path could not find video tokens")
    if not query.is_cuda:
        raise RuntimeError("radial sparse path requires CUDA")

    from ...kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    frame_size = _estimate_frame_size(video_len)
    num_frames = video_len // frame_size
    vid_len = num_frames * frame_size
    vid_start = context_len

    # --- flashinfer variable-block path (HunyuanVideo only: no context offset issue) ---
    if HAS_FLASHINFER and query.is_cuda and context_len == 0:
        block_size = _find_bsr_block_size(vid_len, frame_size)
        num_blocks = vid_len // block_size

        bsr_cache_key = (vid_len, block_size, frame_size, num_frames, decay_factor)
        if bsr_cache_key not in block_mask_cache:
            block_mask_cache[bsr_cache_key] = _radial_bsr_mask(
                vid_len, block_size, frame_size, num_frames, decay_factor,
            )
        bsr_2d = block_mask_cache[bsr_cache_key]  # [num_blocks, num_blocks] bool

        BH = B * H
        # All clusters have the same size: block_size (fixed-block)
        q_sizes = torch.full((BH, num_blocks), block_size, dtype=torch.int32, device=query.device)
        k_sizes = q_sizes

        # dynamic_map = bsr_2d broadcast across BH heads
        dmap = bsr_2d.to(query.device).unsqueeze(0).expand(BH, -1, -1)

        q_vid = query[:, vid_start:vid_start + vid_len, :, :]
        k_vid = key[:, vid_start:vid_start + vid_len, :, :]
        v_vid = value[:, vid_start:vid_start + vid_len, :, :]
        q_flat = q_vid.permute(0, 2, 1, 3).reshape(BH, vid_len, D).contiguous()
        k_flat = k_vid.permute(0, 2, 1, 3).reshape(BH, vid_len, D).contiguous()
        v_flat = v_vid.permute(0, 2, 1, 3).reshape(BH, vid_len, D).contiguous()

        out_flat = variable_block_sparse_attn(q_flat, k_flat, v_flat, dmap, q_sizes, k_sizes)
        out_vid = out_flat.reshape(B, H, vid_len, D).permute(0, 2, 1, 3)

        if tail_len == 0:
            return out_vid

        # Text tokens (HunyuanVideo): attend to everything densely
        q_all = query.permute(0, 2, 1, 3)
        k_all = key.permute(0, 2, 1, 3)
        v_all = value.permute(0, 2, 1, 3)
        out_text = torch.nn.functional.scaled_dot_product_attention(
            q_all[:, :, vid_len:, :], k_all, v_all, dropout_p=0.0,
        ).permute(0, 2, 1, 3)

        return torch.cat([out_vid, out_text], dim=1)

    # --- flex_attention fallback ---
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    video_end = context_len + num_frames * frame_size
    tpf_bits = frame_size.bit_length()

    flex_cache_key = (N, context_len, tail_len, "flex")
    if flex_cache_key not in block_mask_cache:
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

        block_mask_cache[flex_cache_key] = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=query.device,
        )

    bm = block_mask_cache[flex_cache_key]
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    out = flex_attention(q, k, v, block_mask=bm)
    return out.permute(0, 2, 1, 3)


def _find_bsr_block_size(vid_len: int, frame_size: int) -> int:
    """Find the largest block_size ≤ 128 that evenly divides vid_len."""
    for bs in (128, 64, 32, 16):
        if vid_len % bs == 0:
            return bs
    return 1


def _radial_bsr_mask(
    vid_len: int,
    block_size: int,
    frame_size: int,
    num_frames: int,
    decay_factor: float,
) -> torch.Tensor:
    """Build [num_blocks, num_blocks] bool BSR mask evaluated at block centers."""
    import math

    num_blocks = vid_len // block_size
    tpf_bits = frame_size.bit_length()
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)

    for qi in range(num_blocks):
        q_tok = qi * block_size + block_size // 2
        q_frame = q_tok // frame_size
        q_local = q_tok % frame_size

        for ki in range(num_blocks):
            k_tok = ki * block_size + block_size // 2
            k_frame = k_tok // frame_size
            k_local = k_tok % frame_size

            if k_frame == 0:
                mask[qi, ki] = True
                continue

            dist = abs(q_frame - k_frame)
            if dist <= 1:
                mask[qi, ki] = True
                continue

            group = math.floor(math.log2(max(dist, 2))) + 1
            decay_raw = (2.0 ** tpf_bits) / (2.0 ** group)
            window_width = max(decay_raw * decay_factor, 128.0)
            in_band = abs(q_local - k_local) <= window_width
            split_factor = max(1, int(128.0 / max(decay_raw, 1e-9))) if decay_raw < 128.0 else 1
            split_ok = (dist % split_factor) == 0

            mask[qi, ki] = bool(in_band and split_ok)

    return mask
def _estimate_frame_size(video_len):
    for nf in (33, 25, 21, 17, 13, 9, 5):
        if video_len % nf == 0:
            return video_len // nf
    return video_len // 13 if video_len >= 13 else video_len
