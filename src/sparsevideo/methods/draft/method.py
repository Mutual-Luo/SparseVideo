from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_frame_shape, infer_video_token_layout
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

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

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
            full_attention = (
                query.shape[1] != key.shape[1]
                or _draft_is_dense_layer_or_timestep(
                    model_type=model_type,
                    layer_idx=layer_idx,
                    timestep=step_tracker.timestep,
                )
            )
            if full_attention:
                backend_trace = []
                out = _draft_dense_attention(
                    query, key, value,
                    attention_mask=attention_mask,
                    model_type=model_type,
                    text_len=kwargs.get("text_len", 0),
                    backend_trace=backend_trace,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=backend_trace[-1] if backend_trace else None,
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if attention_mask is not None and model_type != "hunyuan_video":
                raise RuntimeError("draft sparse path requires self-attention without an attention mask")
            backend_trace = []
            out = _draft_attention(
                query, key, value,
                sparsity_ratio=cfg["sparsity_ratio"],
                pool_h=cfg["pool_h"],
                pool_w=cfg["pool_w"],
                model_type=model_type,
                text_len=kwargs.get("text_len", 0),
                latent_h=cfg["latent_h"],
                latent_w=cfg["latent_w"],
                visual_len=cfg["visual_len"],
                expected_text_len=cfg["text_len"],
                batch_size=cfg["batch_size"],
                attention_mask=attention_mask,
                allow_triton_fallback=cfg["allow_triton_fallback"],
                backend_trace=backend_trace,
            )
            self.record_runtime_dispatch(
                "sparse",
                backend=backend_trace[-1] if backend_trace else None,
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _draft_dense_attention(query, key, value, attention_mask=None, model_type="wan", text_len=0,
                           backend_trace=None):
    if query.is_cuda:
        try:
            flash_attn_varlen_func = _load_flash_attn_varlen_func()
        except ImportError:
            pass
        else:
            B, query_len, H, D = query.shape
            key_len = key.shape[1]
            if query_len == key_len and model_type == "hunyuan_video":
                video_len = query_len - int(text_len or 0)
                cu_seqlens_q = _draft_cu_seqlens(
                    attention_mask=attention_mask,
                    batch_size=B,
                    total_len=query_len,
                    video_len=video_len,
                    text_len=query_len - video_len,
                    device=query.device,
                )
                cu_seqlens_kv = cu_seqlens_q
            else:
                cu_seqlens_q = torch.arange(
                    0, (B + 1) * query_len, query_len, device=query.device, dtype=torch.int32,
                )
                cu_seqlens_kv = torch.arange(
                    0, (B + 1) * key_len, key_len, device=query.device, dtype=torch.int32,
                )
            q = query.reshape(B * query_len, H, D).contiguous()
            k = key.reshape(B * key_len, H, D).contiguous()
            v = value.reshape(B * key_len, H, D).contiguous()
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                query_len,
                key_len,
            )
            if backend_trace is not None:
                backend_trace.append("flash_attn_varlen")
            return out.view(B, query_len, H, D)

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        dropout_p=0.0, is_causal=False,
    )
    if backend_trace is not None:
        backend_trace.append("torch_sdpa")
    return out.permute(0, 2, 1, 3).contiguous()


def _load_flash_attn_varlen_func():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    return flash_attn_varlen_func


def _draft_attention(query, key, value, sparsity_ratio, pool_h, pool_w,
                     model_type="wan", text_len=0, latent_h=None, latent_w=None,
                     visual_len=None, expected_text_len=None, batch_size=None,
                     attention_mask=None,
                     allow_triton_fallback=False, backend_trace=None):
    """Draft Attention: upstream reorg + percentile mask + block-sparse execution.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    video_len = layout.video_len
    tail_len = layout.tail_len

    if video_len <= 1:
        raise RuntimeError("draft sparse path could not find video tokens")
    if context_len != 0:
        raise RuntimeError("draft upstream layout expects video tokens first and optional text tokens at the end")
    _validate_configured_int("visual_len", visual_len, video_len)
    _validate_configured_int("text_len", expected_text_len, tail_len)
    _validate_configured_int("batch_size", batch_size, B)

    try:
        T, frame_h, frame_w = infer_video_frame_shape(video_len, model_type=model_type)
        _validate_configured_int("latent_h", latent_h, frame_h)
        _validate_configured_int("latent_w", latent_w, frame_w)
        _validate_upstream_draft_layout(
            video_len, frame_h, frame_w, pool_h, pool_w, model_type, text_len=tail_len
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "draft sparse path only supports upstream-compatible latent layouts; "
            f"got video_len={video_len}, shape={locals().get('T', '?')}x"
            f"{locals().get('frame_h', '?')}x{locals().get('frame_w', '?')}"
        ) from exc

    frame_size = frame_h * frame_w
    video_end = context_len + T * frame_size
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=frame_h,
        latent_w=frame_w,
        visual_len=video_len,
        text_len=tail_len,
        device=query.device,
    )

    if query.is_cuda:
        try:
            out = _draft_mit_path(
                query, key, value, B, N, H, D,
                context_len, video_end, T, frame_h, frame_w, frame_size,
                sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx,
                attention_mask=attention_mask,
            )
            if backend_trace is not None:
                backend_trace.append("mit_block_sparse")
            return out
        except Exception as exc:
            if not allow_triton_fallback:
                raise RuntimeError(
                    "draft upstream MIT Block-Sparse-Attention path failed; "
                    "set allow_triton_fallback=True only for debug fallback runs."
                ) from exc
        try:
            out = _draft_triton_path(
                query, key, value, B, N, H, D, scale,
                context_len, video_end, T, frame_h, frame_w, frame_size,
                sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx,
            )
            if backend_trace is not None:
                backend_trace.append("triton_debug_fallback")
            return out
        except Exception as exc:
            raise RuntimeError("draft Triton sparse path failed") from exc

    raise RuntimeError("draft sparse path requires CUDA")


def _validate_configured_int(name, configured, actual):
    if configured is None:
        return
    expected = int(configured)
    if expected != int(actual):
        raise RuntimeError(
            f"draft upstream {name} config mismatch: expected {expected}, got {int(actual)}"
        )


def _draft_mit_path(query, key, value, B, N, H, D,
                    context_len, video_end, T, frame_h, frame_w, frame_size,
                    sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx,
                    attention_mask=None):
    """Upstream Draft path backed by MIT Han Lab Block-Sparse-Attention."""
    from ...kernels.draft_block_sparse_runtime import load_block_sparse_attn_func

    if D != 128:
        raise RuntimeError(
            "draft MIT Block-Sparse-Attention path requires head_dim=128; "
            f"got head_dim={D}"
        )

    video_len = video_end - context_len
    q_vid = query[:, context_len:video_end, :, :]
    k_vid = key[:, context_len:video_end, :, :]

    draft_attn = _sample_qk_attention_2d(q_vid, k_vid, frame_h, frame_w, pool_h, pool_w)
    S = draft_attn.shape[-1]
    m_block_dim = (video_len + S - 1) // S
    n_block_dim = (video_len + S - 1) // S
    if m_block_dim != 128 or n_block_dim != 128:
        raise RuntimeError(
            "draft MIT Block-Sparse-Attention wrapper uses hard-coded 128x128 blocks; "
            f"got m_block_dim={m_block_dim}, n_block_dim={n_block_dim}"
        )

    block_sparse_attn_func = load_block_sparse_attn_func()
    q_block_num = (N + m_block_dim - 1) // m_block_dim
    k_block_num = (N + n_block_dim - 1) // n_block_dim

    base_visual = _attention_percentile_mask_headwise(draft_attn, 1.0 - float(sparsity_ratio))
    cu_seqlens = _draft_cu_seqlens(
        attention_mask=attention_mask,
        batch_size=B,
        total_len=N,
        video_len=video_len,
        text_len=N - video_end,
        device=query.device,
    )
    segment_count = cu_seqlens.numel() - 1
    base_visual = _draft_expand_visual_mask_for_segments(base_visual, segment_count)
    base_blockmask = torch.ones(segment_count, H, q_block_num, k_block_num, dtype=torch.bool, device=query.device)
    base_blockmask[:, :, :base_visual.shape[2], :base_visual.shape[3]] = base_visual

    q_reorg = query.index_select(1, reorg_idx).reshape(B * N, H, D).contiguous()
    k_reorg = key.index_select(1, reorg_idx).reshape(B * N, H, D).contiguous()
    v_reorg = value.index_select(1, reorg_idx).reshape(B * N, H, D).contiguous()
    head_mask_type = torch.ones(H, device=query.device, dtype=torch.int32)

    out = block_sparse_attn_func(
        q_reorg,
        k_reorg,
        v_reorg,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        None,
        base_blockmask,
        N,
        N,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = out.reshape(B, N, H, D)
    return out.index_select(1, restore_idx)


def _draft_cu_seqlens(attention_mask, batch_size, total_len, video_len, text_len, device):
    if attention_mask is None or text_len <= 0:
        return torch.arange(0, (batch_size + 1) * total_len, total_len, device=device, dtype=torch.int32)

    mask = attention_mask.reshape(attention_mask.shape[0], -1)
    if mask.shape[0] != batch_size:
        raise RuntimeError(
            f"draft Hunyuan attention_mask batch mismatch: got {mask.shape[0]}, expected {batch_size}"
        )
    if mask.shape[1] == total_len:
        text_mask = mask[:, video_len:video_len + text_len]
    elif mask.shape[1] == text_len:
        text_mask = mask
    else:
        raise RuntimeError(
            "draft Hunyuan attention_mask must cover either the full sequence "
            f"({total_len}) or the text tail ({text_len}); got {mask.shape[1]}"
        )

    prompt_lengths = text_mask.to(torch.int64).sum(dim=1).clamp(min=0, max=int(text_len))
    cu_seqlens = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=device)
    for i in range(batch_size):
        valid_len = int(video_len) + int(prompt_lengths[i].item())
        cu_seqlens[2 * i + 1] = i * int(total_len) + valid_len
        cu_seqlens[2 * i + 2] = (i + 1) * int(total_len)
    return cu_seqlens


def _draft_expand_visual_mask_for_segments(base_visual, segment_count):
    batch = base_visual.shape[0]
    if segment_count == batch:
        return base_visual
    if segment_count == 2 * batch:
        return base_visual.repeat_interleave(2, dim=0)
    raise RuntimeError(
        f"draft block mask segment mismatch: got segment_count={segment_count}, batch={batch}"
    )


def _draft_triton_path(query, key, value, B, N, H, D, scale,
                       context_len, video_end, T, frame_h, frame_w, frame_size,
                       sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx):
    """Pool-guided block-sparse attention.

    Backend: SparseVideo-owned generic Triton block_sparse_attention. This is
    debug fallback only; the parity path is _draft_mit_path().
    """
    from ...kernels.block_sparse_attn import block_sparse_attention

    video_len = video_end - context_len
    q_vid = query[:, context_len:video_end, :, :]
    k_vid = key[:, context_len:video_end, :, :]

    draft_attn = _sample_qk_attention_2d(q_vid, k_vid, frame_h, frame_w, pool_h, pool_w)
    S = draft_attn.shape[-1]
    m_block_dim = (video_len + S - 1) // S
    n_block_dim = (video_len + S - 1) // S
    q_block_num = (N + m_block_dim - 1) // m_block_dim
    k_block_num = (N + n_block_dim - 1) // n_block_dim

    base_visual = _attention_percentile_mask_headwise(draft_attn, 1.0 - float(sparsity_ratio))
    base_blockmask = torch.ones(B, H, q_block_num, k_block_num, dtype=torch.bool, device=query.device)
    base_blockmask[:, :, :base_visual.shape[2], :base_visual.shape[3]] = base_visual

    q_reorg = query.index_select(1, reorg_idx)
    k_reorg = key.index_select(1, reorg_idx)
    v_reorg = value.index_select(1, reorg_idx)

    q_sorted = q_reorg.permute(0, 2, 1, 3).reshape(B * H, N, D).contiguous()
    k_sorted = k_reorg.permute(0, 2, 1, 3).reshape(B * H, N, D).contiguous()
    v_sorted = v_reorg.permute(0, 2, 1, 3).reshape(B * H, N, D).contiguous()

    q_sizes = _fixed_block_sizes(N, m_block_dim, q_block_num, B * H, query.device)
    k_sizes = _fixed_block_sizes(N, n_block_dim, k_block_num, B * H, query.device)
    dynamic_map = base_blockmask.reshape(B * H, q_block_num, k_block_num)

    out_sorted = block_sparse_attention(
        q_sorted, k_sorted, v_sorted,
        q_sizes.to(torch.long), k_sizes.to(torch.long), dynamic_map, scale,
    )

    out_reorg = out_sorted.reshape(B, H, N, D).permute(0, 2, 1, 3)
    return out_reorg.index_select(1, restore_idx)


def _draft_is_dense_layer_or_timestep(model_type, layer_idx, timestep):
    if model_type == "wan":
        return layer_idx < 1 or timestep > 925
    if model_type == "hunyuan_video":
        return layer_idx < 2 or timestep > 945
    return False


def _validate_upstream_draft_layout(video_len, frame_h, frame_w, pool_h, pool_w, model_type, text_len=None):
    part_size = frame_w * pool_h
    block_size = frame_w
    if frame_h % pool_h != 0:
        raise ValueError("latent_h must be multiple of pool_h")
    if video_len % part_size != 0:
        raise ValueError("visual_len must be multiple of latent_w * pool_h")
    if block_size % pool_w != 0:
        raise ValueError("latent_w must be multiple of pool_w")
    if model_type == "wan" and video_len not in (21 * 32 * 48, 21 * 48 * 80):
        raise ValueError("upstream Wan draft path supports 768x512 or 1280x768 latent layouts")
    if model_type == "hunyuan_video":
        if video_len != 33 * 48 * 80:
            raise ValueError("upstream Hunyuan draft path supports 129-frame 1280x768 latent layout")
        if text_len is not None and text_len != 256:
            raise ValueError("upstream Hunyuan draft path expects a 256-token text tail")


def _generate_reorg_restore_indices(pool_h, pool_w, latent_h, latent_w,
                                    visual_len, text_len, device):
    part_size = latent_w * pool_h
    block_size = latent_w
    sub_block_size = pool_w

    _validate_upstream_draft_layout(visual_len, latent_h, latent_w, pool_h, pool_w, model_type="")

    num_parts = visual_len // part_size
    blocks_per_part = part_size // block_size
    subs_per_block = block_size // sub_block_size

    part_pattern = []
    for c in range(subs_per_block):
        for b in range(blocks_per_part):
            start = b * block_size + c * sub_block_size
            part_pattern.extend(range(start, start + sub_block_size))

    reorg_idx = []
    for p in range(num_parts):
        base = p * part_size
        reorg_idx.extend(base + i for i in part_pattern)

    restore_idx = [0] * visual_len
    for new_pos, orig_pos in enumerate(reorg_idx):
        restore_idx[orig_pos] = new_pos

    if text_len > 0:
        reorg_idx.extend(range(visual_len, visual_len + text_len))
        restore_idx.extend(range(visual_len, visual_len + text_len))

    return (
        torch.tensor(reorg_idx, dtype=torch.long, device=device),
        torch.tensor(restore_idx, dtype=torch.long, device=device),
    )


def _sample_qk_attention_2d(q, k, frame_h, frame_w, pool_h, pool_w):
    B, L, H, D = q.shape
    frame_tokens = frame_h * frame_w
    if L % frame_tokens != 0:
        raise ValueError("L must be multiple of frame_h*frame_w")
    num_frames = L // frame_tokens

    q_vid = q.view(B, num_frames, frame_h, frame_w, H, D)
    k_vid = k.view(B, num_frames, frame_h, frame_w, H, D)
    q_vid = q_vid.permute(0, 1, 4, 5, 2, 3).reshape(B * num_frames, H * D, frame_h, frame_w)
    k_vid = k_vid.permute(0, 1, 4, 5, 2, 3).reshape(B * num_frames, H * D, frame_h, frame_w)

    q_pooled = F.avg_pool2d(q_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)
    k_pooled = F.avg_pool2d(k_vid, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w), ceil_mode=True)

    pooled_h, pooled_w = q_pooled.shape[-2:]
    pooled_len = num_frames * pooled_h * pooled_w

    q_pooled = q_pooled.reshape(B, num_frames, H, D, pooled_h, pooled_w)
    k_pooled = k_pooled.reshape(B, num_frames, H, D, pooled_h, pooled_w)
    q_pooled = q_pooled.permute(0, 1, 4, 5, 2, 3).reshape(B, pooled_len, H, D)
    k_pooled = k_pooled.permute(0, 1, 4, 5, 2, 3).reshape(B, pooled_len, H, D)

    q_heads = q_pooled.permute(0, 2, 1, 3)
    k_heads = k_pooled.permute(0, 2, 1, 3)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * (D ** -0.5)
    return torch.softmax(scores, dim=-1)


def _attention_percentile_mask_headwise(attn_map, keep_ratio):
    B, H, S, _ = attn_map.shape
    mask = torch.zeros_like(attn_map, dtype=torch.bool)
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))

    for b in range(B):
        for h in range(H):
            head_scores = attn_map[b, h]
            flat = head_scores.flatten()
            n = flat.numel()
            k = int((1.0 - keep_ratio) * n)
            if k == 0:
                mask[b, h] = True
                continue
            if k >= n:
                continue
            threshold = torch.topk(flat, k, largest=False).values.max()
            mask[b, h] = head_scores >= threshold

    return mask


def _fixed_block_sizes(total_len, block_dim, block_num, batch_heads, device):
    starts = torch.arange(block_num, device=device, dtype=torch.int64) * int(block_dim)
    sizes = torch.clamp(total_len - starts, min=0, max=int(block_dim)).to(torch.int32)
    return sizes.unsqueeze(0).expand(batch_heads, -1).contiguous()
