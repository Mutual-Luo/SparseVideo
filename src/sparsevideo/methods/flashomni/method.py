from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

from .._base import SparseMethod
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class FlashOmniMethod(SparseMethod):
    """FlashOmni: Block-level similarity → sparse pattern → efficient execution.

    Uses block-level mean pooling to build a sparse pattern, then executes
    via FlashOmni CUDA kernels. The slower flex_attention path is available
    only when explicitly selected with implementation="flex".

    Adapter for: training_free/FlashOmni/flashomni/attention.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if self.config["implementation"] not in ("upstream", "flex"):
            raise ValueError("flashomni implementation must be 'upstream' or 'flex'")
        unsupported = [
            name
            for name, default in method_config.UNPORTED_OPTION_DEFAULTS.items()
            if self.config[name] != default
        ]
        if unsupported:
            raise NotImplementedError(
                "These upstream FlashOmni sparse-info tensor inputs are recognized "
                f"but not wired through SparseVideo CLI/API yet: {unsupported}"
            )

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in ("wan", "hunyuan_video"):
            raise NotImplementedError(f"flashomni not yet supported for {self.model_info.model_type}")

        cfg = self.config

        def attn_fn(query, key, value, attention_mask, **kwargs):
            if cfg["is_full"]:
                return dispatch_attention_fn(
                    query, key, value,
                    attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                )
            return _flashomni_attention(
                query, key, value,
                sparse_kv_budget=cfg["sparse_kv_budget"],
                sparse_block_size_for_q=cfg["sparse_block_size_for_q"],
                sparse_block_size_for_kv=cfg["sparse_block_size_for_kv"],
                implementation=cfg["implementation"],
                backend=cfg["backend"],
                workspace_bytes=cfg["workspace_bytes"],
            )

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _flashomni_attention(query, key, value, sparse_kv_budget,
                         sparse_block_size_for_q, sparse_block_size_for_kv,
                         implementation, backend, workspace_bytes):
    """FlashOmni sparse attention via block-level similarity.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    q = query.permute(0, 2, 1, 3)  # [B, H, N, D]
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    q_block_size = int(sparse_block_size_for_q)
    kv_block_size = int(sparse_block_size_for_kv)
    num_q_blocks = (N + q_block_size - 1) // q_block_size
    num_k_blocks = (N + kv_block_size - 1) // kv_block_size

    # Pad to block-aligned
    q_pad_n = num_q_blocks * q_block_size - N
    kv_pad_n = num_k_blocks * kv_block_size - N
    if q_pad_n > 0:
        q_padded = F.pad(q, (0, 0, 0, q_pad_n))
    else:
        q_padded = q
    if kv_pad_n > 0:
        k_padded = F.pad(k, (0, 0, 0, kv_pad_n))
        v_padded = F.pad(v, (0, 0, 0, kv_pad_n))
    else:
        k_padded = k
        v_padded = v

    q_len_padded = num_q_blocks * q_block_size
    kv_len_padded = num_k_blocks * kv_block_size

    # Block-level means: [B, H, num_blocks, D]
    q_blocks = q_padded.view(B, H, num_q_blocks, q_block_size, D).mean(dim=3)
    k_blocks = k_padded.view(B, H, num_k_blocks, kv_block_size, D).mean(dim=3)

    block_scores = torch.matmul(q_blocks, k_blocks.transpose(-2, -1)) * scale
    block_attn = F.softmax(block_scores, dim=-1)

    k_keep = min(num_k_blocks, max(1, int(num_k_blocks * sparse_kv_budget)))
    _, topk_idx = torch.topk(block_attn, k=k_keep, dim=-1)
    # block_mask_pattern: [B, H, nqb, nkb] bool
    block_mask_pattern = torch.zeros_like(block_attn, dtype=torch.bool)
    block_mask_pattern.scatter_(dim=-1, index=topk_idx, value=True)

    if implementation == "upstream":
        if not query.is_cuda:
            raise RuntimeError("flashomni upstream sparse path requires CUDA")
        out = _flashomni_upstream_attention(
            q_padded, k_padded, v_padded, block_mask_pattern,
            q_len=N, q_block_size=q_block_size, kv_block_size=kv_block_size,
            backend=backend, workspace_bytes=workspace_bytes,
        )
        if q_pad_n > 0:
            out = out[:, :, :N, :]
        return out.permute(0, 2, 1, 3)

    if implementation != "flex":
        raise ValueError("flashomni implementation must be 'upstream' or 'flex'")

    # Explicit fallback: use flex_attention with block mask derived from pattern.
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # block_mask_pattern is [B, H, nqb, nkb], use first batch element
    pattern = block_mask_pattern[0]  # [H, nqb, nkb]

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // q_block_size
        kv_block = kv_idx // kv_block_size
        q_block = torch.clamp(q_block, 0, num_q_blocks - 1)
        kv_block = torch.clamp(kv_block, 0, num_k_blocks - 1)
        return pattern[h, q_block, kv_block]

    bm = create_block_mask(
        mask_mod, B=None, H=H, Q_LEN=q_len_padded, KV_LEN=kv_len_padded,
        device=query.device, BLOCK_SIZE=(q_block_size, kv_block_size),
    )

    out = flex_attention(q_padded, k_padded, v_padded, block_mask=bm)
    if q_pad_n > 0:
        out = out[:, :, :N, :]
    return out.permute(0, 2, 1, 3)


def _flashomni_import():
    try:
        if importlib.util.find_spec("flashomni") is None:
            raise ImportError("flashomni package is not installed")
        if (
            "FLASHOMNI_WORKSPACE_BASE" not in os.environ
            and not os.access(Path.home(), os.W_OK)
        ):
            os.environ["FLASHOMNI_WORKSPACE_BASE"] = tempfile.gettempdir()
        import flashomni
    except Exception as exc:
        raise ImportError(
            "flashomni implementation='upstream' requires the FlashOmni package "
            "with its CUDA/C++ ops built. Install/build flashomni-python with "
            "FLASHOMNI_ENABLE_AOT=1, or explicitly pass implementation=flex for "
            "the slower SparseVideo fallback."
        ) from exc
    return flashomni


def _flashomni_upstream_attention(q, k, v, block_mask_pattern,
                                  q_len, q_block_size, kv_block_size,
                                  backend, workspace_bytes):
    """Execute FlashOmni's BatchFlashOmniFAWithRaggedKVWrapper.

    q/k/v are [B, H, S, D]. FlashOmni uses NHD ragged tensors, so each video in
    the batch is represented by one indptr segment.
    """
    flashomni = _flashomni_import()
    B, H, q_len_padded, D = q.shape
    if q_len > q_len_padded:
        raise RuntimeError("flashomni q_len cannot exceed padded query length")
    kv_len_padded = k.shape[2]
    device = q.device

    q_nhd = q.transpose(1, 2).contiguous().view(B * q_len_padded, H, D)
    k_nhd = k.transpose(1, 2).contiguous().view(B * kv_len_padded, H, D)
    v_nhd = v.transpose(1, 2).contiguous().view(B * kv_len_padded, H, v.shape[-1])

    qo_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * q_len_padded
    kv_indptr = torch.arange(B + 1, device=device, dtype=torch.int32) * kv_len_padded
    workspace = torch.empty(int(workspace_bytes), dtype=torch.uint8, device=device)
    wrapper = flashomni.attention.BatchFlashOmniFAWithRaggedKVWrapper(
        workspace, kv_layout="NHD", backend=backend,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim_qk=D,
        sparse_block_size_for_q=q_block_size,
        sparse_block_size_for_kv=kv_block_size,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    num_q_blocks = block_mask_pattern.shape[2]
    num_kv_blocks = block_mask_pattern.shape[3]
    sparse_info = torch.ones((B, num_q_blocks, H), device=device, dtype=torch.uint8)
    sparse_info = sparse_info.contiguous().view(-1, H)
    sparse_kv_info = block_mask_pattern.to(torch.uint8).transpose(1, 2).contiguous().view(
        -1, H, num_kv_blocks,
    )
    sparse_kv_info = sparse_kv_info * sparse_info.unsqueeze(-1)

    packed_sparse_info, sparse_info_indptr = flashomni.segment_packbits(
        sparse_info.contiguous().view(-1),
        wrapper._sparse_info_indptr_base,
        bitorder="little",
    )
    packed_sparse_kv_info, sparse_kv_info_indptr = flashomni.segment_packbits(
        sparse_kv_info.contiguous().view(-1),
        wrapper._sparse_kv_info_indptr_base,
        bitorder="little",
    )
    out = wrapper.run(
        q_nhd, k_nhd, v_nhd,
        packed_sparse_info,
        packed_sparse_kv_info,
        sparse_info_indptr,
        sparse_kv_info_indptr,
        False,
    )
    return out.view(B, q_len_padded, H, v.shape[-1]).transpose(1, 2).contiguous()
