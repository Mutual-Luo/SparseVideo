from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_frame_shape, infer_video_token_layout
from .._schedule import configured_dense_warmup_layer_count, configured_dense_warmup_requires_dense, runtime_num_inference_steps
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
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
        if self.model_info.model_type not in (
            "wan",
            "hunyuan_video",
            "cogvideox",
            "ltx_video",
            "allegro",
            "mochi",
            "easyanimate",
        ):
            raise NotImplementedError(f"draft not yet supported for {self.model_info.model_type}")

        cfg = self.config

        model_type = self.model_info.model_type
        dense_warmup_layer_count = configured_dense_warmup_layer_count(cfg, total_layers)
        def attn_fn(query, key, value, attention_mask, **kwargs):
            full_attention = (
                layer_idx < dense_warmup_layer_count
                or configured_dense_warmup_requires_dense(
                    cfg,
                    runtime_num_inference_steps(step_tracker),
                    getattr(step_tracker, "step", None),
                    notifier=self.warmup_notifier,
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
            if query.shape[1] != key.shape[1]:
                raise RuntimeError(
                    "draft sparse path requires self-attention with matching query/key lengths; "
                    "dense fallback is controlled only by the common dense warmup ratios"
                )
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
        if self.model_info.model_type == "hunyuan_video":
            return SparseHunyuanVideoAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "cogvideox":
            return SparseCogVideoXAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "ltx_video":
            return SparseLTXVideoAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "allegro":
            return SparseAllegroAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        if self.model_info.model_type == "mochi":
            return SparseMochiAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)
        return SparseEasyAnimateAttnProcessor(attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker)


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
    from ..._flash_attn import require_flash_attn_varlen_func
    return require_flash_attn_varlen_func()


def _draft_attention(query, key, value, sparsity_ratio, pool_h, pool_w,
                     model_type="wan", text_len=0, latent_h=None, latent_w=None,
                     visual_len=None, expected_text_len=None, batch_size=None,
                     attention_mask=None,
                     backend_trace=None):
    """Draft Attention: upstream reorg + percentile mask + block-sparse execution.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape

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
        T, frame_h, frame_w = _infer_draft_frame_shape(
            video_len,
            model_type=model_type,
            latent_h=latent_h,
            latent_w=latent_w,
        )
        _validate_configured_int("latent_h", latent_h, frame_h)
        _validate_configured_int("latent_w", latent_w, frame_w)
        canvas_h = _ceil_to_multiple(frame_h, pool_h)
        canvas_w = _ceil_to_multiple(frame_w, pool_w)
        canvas_video_len = T * canvas_h * canvas_w
        _validate_upstream_draft_layout(
            canvas_video_len,
            canvas_h,
            canvas_w,
            pool_h,
            pool_w,
            model_type,
            text_len=tail_len,
            strict_upstream_shape=False,
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "draft sparse path only supports upstream-compatible latent layouts; "
            f"got video_len={video_len}, shape={locals().get('T', '?')}x"
            f"{locals().get('frame_h', '?')}x{locals().get('frame_w', '?')}"
        ) from exc

    frame_size = frame_h * frame_w
    video_end = context_len + T * frame_size
    kernel_query, kernel_key, kernel_value = query, key, value
    kernel_N = N
    kernel_frame_h = canvas_h
    kernel_frame_w = canvas_w
    kernel_frame_size = canvas_h * canvas_w
    kernel_video_end = context_len + canvas_video_len
    padded_canvas = (canvas_h != frame_h or canvas_w != frame_w)
    if padded_canvas:
        kernel_query, kernel_key, kernel_value = _pad_draft_video_canvas(
            query, key, value,
            context_len=context_len,
            video_end=video_end,
            tail_len=tail_len,
            T=T,
            frame_h=frame_h,
            frame_w=frame_w,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
        kernel_N = kernel_query.shape[1]
    reorg_idx, restore_idx = _generate_reorg_restore_indices(
        pool_h=pool_h,
        pool_w=pool_w,
        latent_h=canvas_h,
        latent_w=canvas_w,
        visual_len=canvas_video_len,
        text_len=tail_len,
        device=query.device,
    )
    if model_type == "hunyuan_video":
        attention_mask = _draft_hunyuan_text_mask_for_mit(
            attention_mask,
            batch_size=B,
            original_total_len=N,
            original_video_len=video_len,
            text_len=tail_len,
        )

    if query.is_cuda:
        out = _draft_mit_path(
            kernel_query, kernel_key, kernel_value, B, kernel_N, H, D,
            context_len, kernel_video_end, T, kernel_frame_h, kernel_frame_w, kernel_frame_size,
            sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx,
            attention_mask=attention_mask,
        )
        if padded_canvas:
            out = _crop_draft_video_canvas(
                out,
                context_len=context_len,
                tail_len=tail_len,
                T=T,
                frame_h=frame_h,
                frame_w=frame_w,
                canvas_h=canvas_h,
                canvas_w=canvas_w,
            )
        if backend_trace is not None:
            backend_trace.append("mit_block_sparse")
        return out

    raise RuntimeError("draft sparse path requires CUDA")


def _validate_configured_int(name, configured, actual):
    if configured is None:
        return
    expected = int(configured)
    if expected != int(actual):
        raise RuntimeError(
            f"draft upstream {name} config mismatch: expected {expected}, got {int(actual)}"
        )


def _draft_hunyuan_text_mask_for_mit(attention_mask, *, batch_size, original_total_len, original_video_len, text_len):
    if attention_mask is None or text_len <= 0:
        return attention_mask

    mask = attention_mask.reshape(attention_mask.shape[0], -1)
    if mask.shape[0] != int(batch_size):
        raise RuntimeError(
            f"draft Hunyuan attention_mask batch mismatch: got {mask.shape[0]}, expected {int(batch_size)}"
        )
    if mask.shape[1] == int(text_len):
        return mask.contiguous()
    if mask.shape[1] == int(original_total_len):
        start = int(original_video_len)
        return mask[:, start:start + int(text_len)].contiguous()
    return attention_mask


def _draft_mit_path(query, key, value, B, N, H, D,
                    context_len, video_end, T, frame_h, frame_w, frame_size,
                    sparsity_ratio, pool_h, pool_w, reorg_idx, restore_idx,
                    attention_mask=None):
    """Upstream Draft path backed by MIT Han Lab Block-Sparse-Attention."""
    from ...kernels.draft_block_sparse_runtime import load_block_sparse_attn_func

    original_D = D
    kernel_D = _draft_mit_head_dim(D)
    softmax_scale = None
    if kernel_D != D:
        pad_dim = kernel_D - D
        query = F.pad(query, (0, pad_dim))
        key = F.pad(key, (0, pad_dim))
        value = F.pad(value, (0, pad_dim))
        D = kernel_D
        softmax_scale = original_D ** -0.5

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
        softmax_scale=softmax_scale,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = out.reshape(B, N, H, D)
    out = out.index_select(1, restore_idx)
    if D != original_D:
        out = out[..., :original_D]
    return out


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


def _draft_mit_head_dim(head_dim):
    head_dim = int(head_dim)
    if head_dim == 128:
        return 128
    if head_dim < 128:
        return 128
    raise RuntimeError(
        "draft MIT Block-Sparse-Attention path requires head_dim <= 128; "
        f"got head_dim={head_dim}"
    )


def _ceil_to_multiple(value, multiple):
    value = int(value)
    multiple = int(multiple)
    if multiple <= 0:
        raise RuntimeError("draft pool_h and pool_w must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def _pad_draft_video_canvas(query, key, value, *, context_len, video_end, tail_len,
                            T, frame_h, frame_w, canvas_h, canvas_w):
    return tuple(
        _pad_draft_tensor_video_canvas(
            tensor,
            context_len=context_len,
            video_end=video_end,
            tail_len=tail_len,
            T=T,
            frame_h=frame_h,
            frame_w=frame_w,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
        for tensor in (query, key, value)
    )


def _pad_draft_tensor_video_canvas(tensor, *, context_len, video_end, tail_len,
                                   T, frame_h, frame_w, canvas_h, canvas_w):
    B, _, H, D = tensor.shape
    prefix = tensor[:, :context_len, :, :] if context_len else None
    video = tensor[:, context_len:video_end, :, :]
    tail = tensor[:, video_end:video_end + tail_len, :, :] if tail_len else None
    video = video.view(B, T, frame_h, frame_w, H, D)
    video = F.pad(video, (0, 0, 0, 0, 0, canvas_w - frame_w, 0, canvas_h - frame_h))
    video = video.reshape(B, T * canvas_h * canvas_w, H, D)
    parts = []
    if prefix is not None:
        parts.append(prefix)
    parts.append(video)
    if tail is not None:
        parts.append(tail)
    return torch.cat(parts, dim=1)


def _crop_draft_video_canvas(out, *, context_len, tail_len, T, frame_h, frame_w, canvas_h, canvas_w):
    B, _, H, D = out.shape
    prefix = out[:, :context_len, :, :] if context_len else None
    video_len = T * frame_h * frame_w
    canvas_video_len = T * canvas_h * canvas_w
    video = out[:, context_len:context_len + canvas_video_len, :, :]
    video = video.view(B, T, canvas_h, canvas_w, H, D)
    video = video[:, :, :frame_h, :frame_w, :, :].reshape(B, video_len, H, D)
    tail = out[:, context_len + canvas_video_len:context_len + canvas_video_len + tail_len, :, :] if tail_len else None
    parts = []
    if prefix is not None:
        parts.append(prefix)
    parts.append(video)
    if tail is not None:
        parts.append(tail)
    return torch.cat(parts, dim=1)


def _infer_draft_frame_shape(video_len, model_type="wan", latent_h=None, latent_w=None):
    if latent_h is not None and latent_w is not None:
        frame_h = int(latent_h)
        frame_w = int(latent_w)
        if frame_h <= 0 or frame_w <= 0:
            raise RuntimeError("draft latent_h and latent_w must be positive")
        frame_size = frame_h * frame_w
        if video_len % frame_size != 0:
            raise RuntimeError(
                f"draft configured latent_h/latent_w={frame_h}x{frame_w} do not divide "
                f"video_len={video_len}"
            )
        return video_len // frame_size, frame_h, frame_w
    return infer_video_frame_shape(video_len, model_type=model_type)


def _validate_upstream_draft_layout(
    video_len,
    frame_h,
    frame_w,
    pool_h,
    pool_w,
    model_type,
    text_len=None,
    *,
    strict_upstream_shape=True,
):
    part_size = frame_w * pool_h
    block_size = frame_w
    if frame_h % pool_h != 0:
        raise ValueError("latent_h must be multiple of pool_h")
    if video_len % part_size != 0:
        raise ValueError("visual_len must be multiple of latent_w * pool_h")
    if block_size % pool_w != 0:
        raise ValueError("latent_w must be multiple of pool_w")
    if strict_upstream_shape and model_type == "wan" and video_len not in (21 * 32 * 48, 21 * 48 * 80):
        raise ValueError("upstream Wan draft path supports 768x512 or 1280x768 latent layouts")
    if strict_upstream_shape and model_type == "hunyuan_video":
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
