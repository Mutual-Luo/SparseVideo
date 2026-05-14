from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from diffusers.models.attention_dispatch import dispatch_attention_fn

from ._base import SparseMethod
from ..processors.wan import SparseWanAttnProcessor
from ..processors.hunyuan_video import SparseHunyuanVideoAttnProcessor


class FlashOmniMethod(SparseMethod):
    """FlashOmni: Block-level similarity → sparse pattern → efficient execution.

    Uses block-level mean pooling to build a sparse pattern, then executes
    via FlashOmni CUDA kernel when available, falling back to flex_attention.

    Port of: training_free/FlashOmni/flashomni/attention.py
    """

    CONFIG_DEFAULTS = {
        "budget": 0.5,
        "block_size": 128,
        "skip_first_steps": 0,
        "skip_first_layers": 0,
    }

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        self._has_flashomni = False
        try:
            import flashomni
            self._has_flashomni = True
        except ImportError:
            pass

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"flashomni not yet supported for {self.model_info.model_type}")

        cfg = self.config
        skip_steps = cfg["skip_first_steps"]
        skip_layers = cfg["skip_first_layers"]
        has_flashomni = self._has_flashomni

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
            return _flashomni_attention(
                query, key, value,
                budget=cfg["budget"],
                block_size=cfg["block_size"],
                has_flashomni=has_flashomni,
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _flashomni_attention(query, key, value, budget, block_size, has_flashomni):
    """FlashOmni sparse attention via block-level similarity.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    q = query.permute(0, 2, 1, 3)  # [B, H, N, D]
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    num_q_blocks = (N + block_size - 1) // block_size
    num_k_blocks = (N + block_size - 1) // block_size

    # Pad to block-aligned
    pad_n = num_q_blocks * block_size - N
    if pad_n > 0:
        q_padded = F.pad(q, (0, 0, 0, pad_n))
        k_padded = F.pad(k, (0, 0, 0, pad_n))
        v_padded = F.pad(v, (0, 0, 0, pad_n))
    else:
        q_padded = q
        k_padded = k
        v_padded = v

    N_padded = num_q_blocks * block_size

    # Block-level means: [B, H, num_blocks, D]
    q_blocks = q_padded.view(B, H, num_q_blocks, block_size, D).mean(dim=3)
    k_blocks = k_padded.view(B, H, num_k_blocks, block_size, D).mean(dim=3)

    block_scores = torch.matmul(q_blocks, k_blocks.transpose(-2, -1)) * scale
    block_attn = F.softmax(block_scores, dim=-1)

    k_keep = max(1, int(num_k_blocks * budget))
    _, topk_idx = torch.topk(block_attn, k=k_keep, dim=-1)
    # block_mask_pattern: [B, H, nqb, nkb] bool
    block_mask_pattern = torch.zeros_like(block_attn, dtype=torch.bool)
    block_mask_pattern.scatter_(dim=-1, index=topk_idx, value=True)

    if has_flashomni:
        try:
            import flashomni
            out = flashomni.attention(
                q_padded, k_padded, v_padded,
                block_mask=block_mask_pattern,
                block_size=block_size,
            )
            if pad_n > 0:
                out = out[:, :, :N, :]
            return out.permute(0, 2, 1, 3)
        except Exception:
            pass

    # Fallback: use flex_attention with block mask derived from pattern
    # block_mask_pattern is [B, H, nqb, nkb], use first batch element
    pattern = block_mask_pattern[0]  # [H, nqb, nkb]

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        q_block = torch.clamp(q_block, 0, num_q_blocks - 1)
        kv_block = torch.clamp(kv_block, 0, num_k_blocks - 1)
        return pattern[h, q_block, kv_block]

    bm = create_block_mask(
        mask_mod, B=None, H=H, Q_LEN=N_padded, KV_LEN=N_padded,
        device=query.device, BLOCK_SIZE=(block_size, block_size),
    )

    out = flex_attention(q_padded, k_padded, v_padded, block_mask=bm)
    if pad_n > 0:
        out = out[:, :, :N, :]
    return out.permute(0, 2, 1, 3)
