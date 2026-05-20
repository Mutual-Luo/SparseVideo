from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._layout import infer_video_frame_count, infer_video_token_layout
from .._schedule import first_times_fp_requires_dense, resolve_first_layers, scheduler_timestep_from_tracker
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config

_SVG_FLEX_ATTENTION = {}
_TEXT_TAIL_MODELS = {"hunyuan_video", "cogvideox", "mochi", "easyanimate"}


class SVG1Method(SparseMethod):
    """SVG1: Sparse VideoGen stripe-based attention with online MSE profiling.

    Port of: training_free/Sparse-VideoGen/svg/models/wan/attention.py + utils.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in (
            "wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate",
        ):
            raise NotImplementedError(f"svg1 not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = resolve_first_layers(cfg["first_layers_fp"], total_layers)

        state = {"block_mask": None, "profiled_step": -1}
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            scheduler_timestep = scheduler_timestep_from_tracker(step_tracker, kwargs)
            prompt_length = kwargs.get("prompt_length")
            if prompt_length is None:
                prompt_length = cfg.get("prompt_length")
            full_attention = (
                layer_idx < first_layer_count
                or first_times_fp_requires_dense(
                    cfg["first_times_fp"],
                    cfg["num_inference_steps"],
                    step_tracker.step,
                    scheduler_timestep,
                )
            )
            if full_attention:
                out = _svg1_dense_attention(
                    query, key, value, attention_mask,
                    model_type=self.model_info.model_type,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_svg1_dense_backend_name(query, attention_mask, self.model_info.model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("svg1 sparse path requires CUDA self-attention without an attention mask")
            if attention_mask is not None and model_type not in _TEXT_TAIL_MODELS:
                raise RuntimeError("svg1 sparse path requires CUDA self-attention without an attention mask")
            out = _svg_attention(
                query, key, value,
                sparsity=cfg["sparsity"],
                num_sampled_rows=cfg["num_sampled_rows"],
                sample_mse_max_row=cfg["sample_mse_max_row"],
                state=state,
                step_tracker_step=step_tracker.step,
                model_type=model_type,
                text_len=kwargs.get("text_len", 0),
                prompt_length=prompt_length,
                context_length=cfg.get("context_length"),
            )
            self.record_runtime_dispatch(
                "sparse",
                backend="flex_attention",
                layer_idx=layer_idx,
                step=getattr(step_tracker, "step", None),
            )
            return out

        if self.model_info.model_type == "wan":
            return SparseWanAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "cogvideox":
            return SparseCogVideoXAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "ltx_video":
            return SparseLTXVideoAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "allegro":
            return SparseAllegroAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "mochi":
            return SparseMochiAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        if self.model_info.model_type == "easyanimate":
            return SparseEasyAnimateAttnProcessor(
                attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
            )
        return SparseHunyuanVideoAttnProcessor(
            attn_fn=attn_fn, layer_idx=layer_idx, step_tracker=step_tracker,
        )


def _svg1_dense_backend_name(query, attention_mask, model_type):
    if model_type == "hunyuan_video" and attention_mask is not None and query.is_cuda:
        return "flash_attn_varlen"
    if attention_mask is not None:
        return "diffusers_dispatch"
    return "torch_sdpa"


def _svg_attention(query, key, value, sparsity, num_sampled_rows,
                   sample_mse_max_row, state, step_tracker_step,
                   model_type="wan", text_len=0, prompt_length=None,
                   context_length=None):
    """SVG stripe-based sparse attention with per-head profiling.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    scale = D ** -0.5

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    tail_len = layout.tail_len
    video_len = layout.video_len
    if model_type in _TEXT_TAIL_MODELS and tail_len > 0:
        if context_length is not None and int(context_length) != tail_len:
            raise RuntimeError(
                "svg1 context_length must match the text token tail length "
                f"seen by the processor; got context_length={int(context_length)}, text_len={tail_len}"
            )

    if video_len <= 0:
        raise RuntimeError("svg1 sparse path could not find video tokens")
    if not query.is_cuda:
        raise RuntimeError("svg1 sparse path requires CUDA")

    num_frames = infer_video_frame_count(video_len, model_type=model_type)
    frame_size = video_len // num_frames
    video_end = context_len + num_frames * frame_size
    if context_len != 0:
        raise RuntimeError("svg1 sparse path currently expects video tokens before any text/context tail")
    window_width = _svg_window_width(sparsity, model_type, tail_len, num_frames, frame_size)

    head_choices = _profile_masks(
        query, key, value, scale, context_len, video_end,
        frame_size, num_frames, num_sampled_rows,
        sample_mse_max_row, model_type=model_type,
    )

    prompt_length = _resolve_prompt_length(prompt_length, text_len)
    block_mask_key = (
        N,
        num_frames,
        frame_size,
        float(window_width),
        model_type,
        prompt_length,
    )
    if state.get("block_mask") is None or state.get("block_mask_key") != block_mask_key:
        state["block_mask"] = _build_svg_block_mask(
            N, video_len, frame_size, num_frames, window_width,
            query.device, model_type=model_type, prompt_length=prompt_length,
        )
        state["block_mask_key"] = block_mask_key
    state["profiled_step"] = step_tracker_step

    bm = state["block_mask"]

    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()

    q, k, v = _place_svg_heads(q, k, v, head_choices, video_len, num_frames, frame_size, text_len)
    out = _svg_flex_attention(q, k, v, block_mask=bm, model_type=model_type)
    out = _restore_svg_heads(out, head_choices, video_len, num_frames, frame_size, text_len)
    return out.permute(0, 2, 1, 3)


def _svg_flex_attention(query, key, value, block_mask, model_type="wan"):
    global _SVG_FLEX_ATTENTION

    if model_type not in _SVG_FLEX_ATTENTION:
        from torch.nn.attention.flex_attention import flex_attention

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 192 * 3)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit,
            192 * 3,
        )
        if model_type == "hunyuan_video":
            # Match Sparse-VideoGen's Hunyuan SVG1 compile call.
            _SVG_FLEX_ATTENTION[model_type] = torch.compile(flex_attention, dynamic=False)
        else:
            # Match Sparse-VideoGen's Wan SVG1 compile call.
            _SVG_FLEX_ATTENTION[model_type] = torch.compile(
                flex_attention,
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )
    return _SVG_FLEX_ATTENTION[model_type](query, key, value, block_mask=block_mask)


def _svg1_dense_attention(query, key, value, attention_mask, *, model_type):
    if model_type == "hunyuan_video" and attention_mask is not None and query.is_cuda:
        return _svg1_hunyuan_flash_attn_varlen(query, key, value, attention_mask)
    if attention_mask is not None:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        return dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        dropout_p=0.0, is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _svg1_hunyuan_flash_attn_varlen(query, key, value, attention_mask):
    if query.shape[0] != 1:
        raise RuntimeError("SVG1 Hunyuan FlashAttention varlen path follows upstream batch size 1")

    flash_attn_varlen_func = _load_flash_attn_varlen_func()
    batch, seq_len, heads, dim = query.shape
    q = query.permute(1, 0, 2, 3).reshape(seq_len, batch * heads, dim).contiguous()
    k = key.permute(1, 0, 2, 3).reshape(seq_len, batch * heads, dim).contiguous()
    v = value.permute(1, 0, 2, 3).reshape(seq_len, batch * heads, dim).contiguous()

    valid_len = int(attention_mask.sum().item())
    total_len = int(attention_mask.numel())
    cu_seqlens = torch.tensor([0, valid_len, total_len], dtype=torch.int32, device=query.device)
    hidden_states = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens, cu_seqlens,
        total_len, total_len,
    )
    return hidden_states.permute(1, 0, 2).reshape(batch, heads, seq_len, dim).permute(0, 2, 1, 3)


def _load_flash_attn_varlen_func():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    return flash_attn_varlen_func


def _profile_masks(query, key, value, scale, context_len, video_end,
                   frame_size, num_frames, num_sampled_rows,
                   sample_mse_max_row, model_type="wan"):
    """Profile two mask candidates on sampled rows and select best per head.

    Returns head_choices: [B, H] tensor (0=spatial, 1=temporal).
    """
    B, N, H, D = query.shape
    device = query.device

    num_sample = min(num_sampled_rows, N)
    sample_high = max(1, min(int(sample_mse_max_row), N))
    # Upstream Sparse-VideoGen samples rows on the default CPU generator before
    # indexing CUDA tensors; keep that RNG stream so fixed seeds choose the same
    # profiling rows.
    sampled_idx = torch.randint(0, sample_high, (num_sample,), device="cpu").to(device)

    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)

    sampled_q = q[:, :, sampled_idx, :]
    scores_sample = torch.matmul(sampled_q, k.transpose(-2, -1)) * scale

    weights_dense = F.softmax(scores_sample, dim=-1)
    out_dense = torch.matmul(weights_dense, v)

    all_idx = torch.arange(N, device=device)

    mask_a = _svg_profile_mask_rows(
        "spatial", sampled_idx, all_idx, context_len, video_end,
        frame_size, num_frames, model_type=model_type,
    )
    mask_b = _svg_profile_mask_rows(
        "temporal", sampled_idx, all_idx, context_len, video_end,
        frame_size, num_frames, model_type=model_type,
    )

    mses = []
    for mask in [mask_a, mask_b]:
        masked_scores = scores_sample.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        w = F.softmax(masked_scores, dim=-1)
        out_sp = torch.matmul(w, v)
        mse = ((out_sp - out_dense) ** 2).mean(dim=(-2, -1))
        mses.append(mse)

    mses = torch.stack(mses, dim=0)
    return mses.argmin(dim=0)


def _svg_profile_mask_rows(mask_name, q_idx, all_idx, context_len, video_end,
                           frame_size, num_frames, model_type="wan"):
    """Build upstream SVG profiling mask rows. [num_sample, N]"""
    q = q_idx.unsqueeze(1)
    k = all_idx.unsqueeze(0)
    q_in_video = (q >= context_len) & (q < video_end)
    k_in_video = (k >= context_len) & (k < video_end)
    both_video = q_in_video & k_in_video

    q_pos = q - context_len
    k_pos = k - context_len
    if mask_name == "temporal":
        q_pos = _frame_major_to_token_major(q_pos, num_frames, frame_size)
        k_pos = _frame_major_to_token_major(k_pos, num_frames, frame_size)
    elif mask_name != "spatial":
        raise ValueError(f"Unknown svg1 profiling mask {mask_name!r}")

    block_size = 128
    if model_type in _TEXT_TAIL_MODELS:
        block_thres = frame_size * 1.5
        is_sink = torch.zeros_like(k_in_video)
    else:
        block_thres = frame_size * 2
        sink_pos = k_pos if mask_name == "temporal" else (k - context_len)
        is_sink = k_in_video & (sink_pos < frame_size)

    block_window = block_thres // block_size
    in_window = torch.abs(q_pos // block_size - k_pos // block_size) < block_window
    video_mask = is_sink | in_window
    return (~both_video) | video_mask


def _build_svg_block_mask(N, video_len, frame_size, num_frames, window_width,
                          device, model_type="wan", prompt_length=0):
    """Build the common upstream SVG FlexAttention block mask."""
    from torch.nn.attention.flex_attention import BlockMask

    def mask_mod(b, h, q_idx, kv_idx):
        return _svg_common_mask(q_idx, kv_idx, video_len, frame_size, window_width, model_type, prompt_length)

    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = _svg_kv_block_partitions(
        N,
        video_len,
        frame_size,
        int(window_width),
        model_type=model_type,
        prompt_length=prompt_length,
        device=device,
    )
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=(128, 128),
        mask_mod=mask_mod,
        seq_lengths=(N, N),
    )


def _svg_kv_blocks(N, video_len, frame_size, window_width,
                   model_type="wan", prompt_length=0, device=None):
    block_size = 128
    num_q_blocks = math.ceil(N / block_size)
    num_kv_blocks = math.ceil(N / block_size)
    per_q_blocks = []
    for q_block in range(num_q_blocks):
        q_start = q_block * block_size
        q_end = min(N, (q_block + 1) * block_size) - 1
        blocks: set[int] = set()
        if model_type in _TEXT_TAIL_MODELS:
            _add_hunyuan_svg_blocks(
                blocks, q_start, q_end, N, video_len, int(prompt_length or 0),
                window_width, block_size, num_kv_blocks,
            )
        else:
            _add_block_range(blocks, 0, frame_size - 1, block_size, num_kv_blocks)
            _add_block_range(
                blocks,
                q_start - window_width,
                q_end + window_width,
                block_size,
                num_kv_blocks,
            )
        per_q_blocks.append(sorted(blocks))

    # BlockMask.from_kv_blocks currently interprets the last dimension of
    # kv_indices as the full KV block count when deriving the transposed Q
    # metadata, so pad to num_kv_blocks rather than to the per-row maximum.
    max_blocks = num_kv_blocks
    kv_num_blocks = torch.empty((1, 1, num_q_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((1, 1, num_q_blocks, max_blocks), dtype=torch.int32, device=device)
    for q_block, blocks in enumerate(per_q_blocks):
        kv_num_blocks[0, 0, q_block] = len(blocks)
        if blocks:
            kv_indices[0, 0, q_block, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
    return kv_num_blocks, kv_indices


def _svg_kv_block_partitions(N, video_len, frame_size, window_width,
                             model_type="wan", prompt_length=0, device=None):
    block_size = 128
    num_q_blocks = math.ceil(N / block_size)
    num_kv_blocks = math.ceil(N / block_size)
    partial_rows = []
    full_rows = []
    for q_block in range(num_q_blocks):
        q_start = q_block * block_size
        q_end = min(N, (q_block + 1) * block_size) - 1
        candidates = set()
        if model_type in _TEXT_TAIL_MODELS:
            _add_hunyuan_svg_blocks(
                candidates, q_start, q_end, N, video_len, int(prompt_length or 0),
                window_width, block_size, num_kv_blocks,
            )
        else:
            _add_block_range(candidates, 0, frame_size - 1, block_size, num_kv_blocks)
            _add_block_range(
                candidates,
                q_start - window_width,
                q_end + window_width,
                block_size,
                num_kv_blocks,
            )

        partial_blocks = []
        full_blocks = []
        for kv_block in sorted(candidates):
            kv_start = kv_block * block_size
            kv_end = min(N, (kv_block + 1) * block_size) - 1
            if _svg_block_is_full(
                q_start, q_end, kv_start, kv_end, N, video_len, frame_size,
                window_width, model_type=model_type, prompt_length=prompt_length,
            ):
                full_blocks.append(kv_block)
            else:
                partial_blocks.append(kv_block)
        partial_rows.append(partial_blocks)
        full_rows.append(full_blocks)

    kv_num_blocks, kv_indices = _svg_block_rows_to_tensors(partial_rows, num_q_blocks, num_kv_blocks, device)
    full_kv_num_blocks, full_kv_indices = _svg_block_rows_to_tensors(full_rows, num_q_blocks, num_kv_blocks, device)
    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices


def _svg_block_rows_to_tensors(rows, num_q_blocks, num_kv_blocks, device=None):
    # BlockMask.from_kv_blocks currently interprets the last dimension of
    # kv_indices as the full KV block count when deriving the transposed Q
    # metadata, so pad to num_kv_blocks rather than to the per-row maximum.
    kv_num_blocks = torch.empty((1, 1, num_q_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((1, 1, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=device)
    for q_block, blocks in enumerate(rows):
        kv_num_blocks[0, 0, q_block] = len(blocks)
        if blocks:
            kv_indices[0, 0, q_block, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=device)
    return kv_num_blocks, kv_indices


def _svg_block_is_full(q_start, q_end, kv_start, kv_end, N, video_len, frame_size,
                       window_width, model_type="wan", prompt_length=0):
    # create_block_mask pads out-of-sequence tokens with False, so padded edge
    # blocks are always partial even if every valid token in the block is kept.
    block_size = 128
    if q_end - q_start + 1 != block_size or kv_end - kv_start + 1 != block_size:
        return False

    if model_type in _TEXT_TAIL_MODELS:
        real_length = min(N, video_len + int(prompt_length or 0))
        q_fake_full = q_start >= real_length
        kv_fake_full = kv_start >= real_length
        if q_fake_full and kv_fake_full:
            return True
        q_real_full = q_end < real_length
        kv_real_full = kv_end < real_length
        if not (q_real_full and kv_real_full):
            return False
        q_text_full = q_start >= video_len
        kv_text_full = kv_start >= video_len
        temporal_full = _max_abs_between_intervals(q_start, q_end, kv_start, kv_end) < window_width
        return q_text_full or kv_text_full or temporal_full

    first_frame_full = kv_end < frame_size
    temporal_full = _max_abs_between_intervals(q_start, q_end, kv_start, kv_end) <= window_width
    return first_frame_full or temporal_full


def _max_abs_between_intervals(a_start, a_end, b_start, b_end):
    return max(abs(a_start - b_end), abs(a_end - b_start))


def _add_hunyuan_svg_blocks(blocks, q_start, q_end, N, video_len, prompt_length,
                            window_width, block_size, num_kv_blocks):
    real_length = min(N, video_len + int(prompt_length or 0))
    q_intersects_real = q_start < real_length and q_end >= 0
    q_intersects_text = q_start < real_length and q_end >= video_len
    q_intersects_fake = q_end >= real_length

    if q_intersects_real:
        if q_intersects_text:
            _add_block_range(blocks, 0, real_length - 1, block_size, num_kv_blocks)
        else:
            _add_block_range(
                blocks,
                q_start - window_width + 1,
                min(real_length - 1, q_end + window_width - 1),
                block_size,
                num_kv_blocks,
            )
            _add_block_range(blocks, video_len, real_length - 1, block_size, num_kv_blocks)
    if q_intersects_fake:
        _add_block_range(blocks, real_length, N - 1, block_size, num_kv_blocks)


def _add_block_range(blocks, token_start, token_end, block_size, num_blocks):
    start = max(0, int(token_start))
    end = min(num_blocks * block_size - 1, int(token_end))
    if end < start:
        return
    first = start // block_size
    last = min(num_blocks - 1, end // block_size)
    blocks.update(range(first, last + 1))


def _svg_common_mask(q_idx, kv_idx, video_len, frame_size, window_width,
                     model_type="wan", prompt_length=0):
    if model_type in _TEXT_TAIL_MODELS:
        real_length = video_len + int(prompt_length or 0)
        real_mask = (kv_idx < real_length) & (q_idx < real_length)
        fake_mask = (kv_idx >= real_length) & (q_idx >= real_length)
        text_column_mask = (video_len <= kv_idx) & (kv_idx < real_length)
        text_row_mask = (video_len <= q_idx) & (q_idx < real_length)
        temporal_head_mask = torch.abs(q_idx - kv_idx) < window_width
        return (real_mask & (temporal_head_mask | text_column_mask | text_row_mask)) | fake_mask

    temporal_head_mask = torch.abs(q_idx - kv_idx) <= window_width
    first_frame_mask = kv_idx < frame_size
    return first_frame_mask | temporal_head_mask


def _place_svg_heads(query, key, value, head_choices, video_len, num_frames, frame_size, context_length=0):
    if query.is_cuda and _svg_placement_triton_supported(query):
        from .placement import sparse_head_placement

        return sparse_head_placement(query, key, value, head_choices, context_length, num_frames, frame_size)
    return _place_svg_heads_pytorch(query, key, value, head_choices, video_len, num_frames, frame_size)


def _place_svg_heads_pytorch(query, key, value, head_choices, video_len, num_frames, frame_size):
    return tuple(
        _select_temporal_heads(
            tensor,
            _to_token_major(tensor, video_len, num_frames, frame_size),
            head_choices,
        )
        for tensor in (query, key, value)
    )


def _restore_svg_heads(hidden_states, head_choices, video_len, num_frames, frame_size, context_length=0):
    if hidden_states.is_cuda and _svg_placement_triton_supported(hidden_states):
        from .placement import hidden_states_placement

        return hidden_states_placement(hidden_states, head_choices, context_length, num_frames, frame_size)
    return _restore_svg_heads_pytorch(hidden_states, head_choices, video_len, num_frames, frame_size)


def _restore_svg_heads_pytorch(hidden_states, head_choices, video_len, num_frames, frame_size):
    return _select_temporal_heads(
        hidden_states,
        _to_frame_major(hidden_states, video_len, num_frames, frame_size),
        head_choices,
    )


def _svg_placement_triton_supported(tensor):
    head_dim = int(tensor.shape[-1])
    return head_dim > 0 and (head_dim & (head_dim - 1)) == 0


def _select_temporal_heads(spatial_tensor, temporal_tensor, head_choices):
    temporal = head_choices.to(device=spatial_tensor.device, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
    return torch.where(temporal, temporal_tensor, spatial_tensor)


def _to_token_major(tensor, video_len, num_frames, frame_size):
    token_major = _token_major_indices(video_len, num_frames, frame_size, tensor.device)
    out = tensor.clone()
    out[:, :, token_major, :] = tensor[:, :, :video_len, :]
    return out


def _to_frame_major(tensor, video_len, num_frames, frame_size):
    token_major = _token_major_indices(video_len, num_frames, frame_size, tensor.device)
    out = tensor.clone()
    out[:, :, :video_len, :] = tensor[:, :, token_major, :]
    return out


def _token_major_indices(video_len, num_frames, frame_size, device):
    idx = torch.arange(video_len, device=device)
    return _frame_major_to_token_major(idx, num_frames, frame_size)


def _frame_major_to_token_major(idx, num_frames, frame_size):
    return (idx % frame_size) * num_frames + (idx // frame_size)


def _round_svg_window_width(window_width, model_type="wan"):
    if model_type == "hunyuan_video":
        return math.floor(float(window_width) / 128) * 128
    return math.ceil(float(window_width) / 128) * 128


def _svg_window_width(sparsity, model_type, tail_len, num_frames, frame_size):
    context_length = int(tail_len or 0) if model_type in _TEXT_TAIL_MODELS else 0
    return _round_svg_window_width(
        _sparsity_to_width(sparsity, context_length, num_frames, frame_size) * frame_size,
        model_type=model_type,
    )


def _resolve_prompt_length(prompt_length, text_len):
    if prompt_length is None:
        return int(text_len or 0)
    if isinstance(prompt_length, torch.Tensor):
        return int(prompt_length.detach().flatten()[0].item())
    return int(prompt_length)


def _sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2
    adjusted = (float(sparsity) * total_elements - 2 * seq_len * context_length) / total_elements
    adjusted = min(max(adjusted, 0.0), 1.0 - 1e-12)
    width = seq_len * (1 - math.sqrt(1 - adjusted))
    return width / frame_size
