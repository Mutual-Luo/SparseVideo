from __future__ import annotations

import torch

from .._base import SparseMethod
from .._layout import infer_video_frame_shape_for_attention, infer_video_token_layout
from .._schedule import configured_dense_warmup_layer_count, configured_dense_warmup_requires_dense, runtime_or_config_num_inference_steps
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from ...kernels.sageattention_runtime import load_sageattn_function
from ...kernels.spas_sage_runtime import load_block_sparse_sage2_attn_function
from . import config as method_config


class RadialMethod(SparseMethod):
    """Radial attention with logarithmic band decay per frame-pair distance.

    Port of: training_free/radial-attention/radial_attn/attn_mask.py
    """

    CONFIG_DEFAULTS = method_config.CONFIG_DEFAULTS
    CONFIG_ALIASES = method_config.CONFIG_ALIASES

    @classmethod
    def default_config(cls, **context):
        return method_config.default_config(**context)

    def __init__(self, config, model_info):
        super().__init__(config, model_info)
        if self.config["block_size"] not in (64, 128):
            raise ValueError("radial block_size must be 64 or 128")
        self._block_sparse_sage2_attn_fn = None
        self._sageattn_fn = None
        if self.config["use_sage_attention"]:
            try:
                self._block_sparse_sage2_attn_fn = load_block_sparse_sage2_attn_function()
                self._sageattn_fn = load_sageattn_function()
            except ImportError as exc:
                raise ImportError(
                    "radial use_sage_attention requires SparseVideo-owned spas_sage_attn "
                    "with block_sparse_sage2_attn_cuda and SparseVideo-owned SageAttention "
                    "with sageattn under src/sparsevideo/kernels/native. Do not rely on "
                    "training_free/ or environment packages."
                ) from exc

    def create_processor(self, layer_idx, total_layers, original_processor, step_tracker):
        if self.model_info.model_type not in (
            "wan", "hunyuan_video", "cogvideox", "ltx_video", "allegro", "mochi", "easyanimate",
        ):
            raise NotImplementedError(f"radial not yet supported for {self.model_info.model_type}")

        decay_factor = self.config["decay_factor"]
        block_size = self.config["block_size"]
        use_sage_attention = self.config["use_sage_attention"]
        block_sparse_sage2_attn_fn = self._block_sparse_sage2_attn_fn
        sageattn_fn = self._sageattn_fn
        dense_warmup_layer_count = configured_dense_warmup_layer_count(self.config, total_layers)

        block_mask_cache = {}
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            full_attention = (
                layer_idx < dense_warmup_layer_count
                or configured_dense_warmup_requires_dense(
                    self.config,
                    runtime_or_config_num_inference_steps(step_tracker, self.config),
                    getattr(step_tracker, "step", None),
                    notifier=self.warmup_notifier,
                )
            )
            if full_attention:
                text_len = kwargs.get("text_len", 0)
                out = _radial_attention(
                    query, key, value,
                    decay_factor=decay_factor,
                    block_mask_cache=block_mask_cache,
                    block_size=block_size,
                    model_type=model_type,
                    text_len=text_len,
                    seq_shape=kwargs.get("seq_shape"),
                    attention_mask=attention_mask,
                    use_sage_attention=use_sage_attention,
                    block_sparse_sage2_attn_fn=block_sparse_sage2_attn_fn,
                    sageattn_fn=sageattn_fn,
                    force_dense=True,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_radial_backend_name(
                        query, block_size, model_type, text_len,
                        use_sage_attention=use_sage_attention,
                        force_dense=True,
                    ),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("radial sparse path requires CUDA self-attention")
            if (
                attention_mask is not None
                and model_type not in ("hunyuan_video", "cogvideox", "mochi", "easyanimate")
            ):
                raise RuntimeError("radial sparse path only supports attention masks for Hunyuan-style text tails")
            text_len = kwargs.get("text_len", 0)
            out = _radial_attention(
                query, key, value,
                decay_factor=decay_factor,
                block_mask_cache=block_mask_cache,
                block_size=block_size,
                model_type=model_type,
                text_len=text_len,
                seq_shape=kwargs.get("seq_shape"),
                attention_mask=attention_mask,
                use_sage_attention=use_sage_attention,
                block_sparse_sage2_attn_fn=block_sparse_sage2_attn_fn,
                sageattn_fn=sageattn_fn,
            )
            self.record_runtime_dispatch(
                "sparse",
                backend=_radial_backend_name(
                    query, block_size, model_type, text_len,
                    use_sage_attention=use_sage_attention,
                    force_dense=False,
                ),
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


def _radial_backend_name(
    query,
    block_size,
    model_type,
    text_len,
    *,
    use_sage_attention,
    force_dense,
):
    if use_sage_attention:
        return "sage_dense" if force_dense else "sage_block_sparse"
    if force_dense:
        return "flashinfer_dense"

    layout = infer_video_token_layout(query.shape[1], model_type=model_type, text_len=text_len)
    return "flashinfer"


def _radial_attention(query, key, value, decay_factor, block_mask_cache, block_size=128,
                      model_type="wan", text_len=0, attention_mask=None,
                      seq_shape=None,
                      use_sage_attention=False,
                      block_sparse_sage2_attn_fn=None, sageattn_fn=None,
                      force_dense=False):
    """Radial attention with logarithmic band decay per frame-pair distance.

    For HunyuanVideo (no context prefix):
        flashinfer block-sparse with fixed block_size ≤ 128, or variable final
        block size when the video token count is not exactly divisible.
        Text tokens are appended and always fully attended via dense SDPA.

    For WAN:
        SparseVideo patches attn1 self-attention, so the sequence contains
        video tokens only.

    query/key/value: [B, N, H, D]
    """
    B, N, H, D = query.shape
    kv_len = key.shape[1]
    if value.shape[1] != kv_len:
        raise RuntimeError(
            "radial sparse path requires key/value lengths to match; "
            f"got key_len={kv_len}, value_len={value.shape[1]}"
        )

    layout = infer_video_token_layout(N, model_type=model_type, text_len=text_len)
    context_len = layout.context_len
    video_len = layout.video_len
    tail_len = layout.tail_len

    if video_len <= 0:
        raise RuntimeError("radial sparse path could not find video tokens")
    if not query.is_cuda:
        raise RuntimeError("radial sparse path requires CUDA")

    num_frames, frame_h, frame_w = infer_video_frame_shape_for_attention(
        video_len,
        model_type=model_type,
        seq_shape=seq_shape,
    )
    frame_size = frame_h * frame_w
    vid_len = num_frames * frame_size
    q_kv_offset = 0
    kv_video_len = vid_len
    kv_num_frames = num_frames
    if model_type not in ("hunyuan_video", "cogvideox", "mochi", "easyanimate") and kv_len != N:
        if kv_len < N:
            raise RuntimeError(
                "radial rectangular sparse path requires key/value length >= query length; "
                f"got query_len={N}, key_len={kv_len}"
            )
        q_kv_offset = kv_len - N
        if q_kv_offset % frame_size != 0 or kv_len % frame_size != 0:
            raise RuntimeError(
                "radial rectangular sparse path requires the condition prefix "
                "and KV sequence to align to whole video frames; "
                f"got query_len={N}, key_len={kv_len}, frame_size={frame_size}"
            )
        kv_video_len = kv_len
        kv_num_frames = kv_video_len // frame_size
    vid_start = context_len
    pre_defined_mask = _expand_attention_mask(attention_mask, N, query.device)

    if force_dense:
        if context_len != 0:
            raise RuntimeError("radial dense path only supports layouts without a context prefix")
        if use_sage_attention:
            dense_mask_key = (vid_len, kv_video_len, block_size, model_type, "sage-dense")
            if dense_mask_key not in block_mask_cache:
                dense_q_blocks = _ceil_div(vid_len, block_size)
                dense_kv_blocks = _ceil_div(kv_video_len, block_size)
                block_mask_cache[dense_mask_key] = torch.ones(
                    (dense_q_blocks, dense_kv_blocks),
                    dtype=torch.bool,
                    device=query.device,
                )
            if kv_video_len != vid_len:
                return _radial_sage_dense_attention(
                    query,
                    key,
                    value,
                    vid_len,
                    tail_len,
                    pre_defined_mask,
                    sageattn_fn,
                )
            return _radial_sage_attention(
                query,
                key,
                value,
                block_mask_cache[dense_mask_key],
                vid_len,
                tail_len,
                block_size,
                pre_defined_mask,
                block_sparse_sage2_attn_fn,
                sageattn_fn,
            )
        dense_mask_key = (vid_len, kv_video_len, block_size, model_type, "flashinfer-dense")
        if dense_mask_key not in block_mask_cache:
            dense_q_blocks = _ceil_div(vid_len, block_size)
            dense_kv_blocks = _ceil_div(kv_video_len, block_size)
            block_mask_cache[dense_mask_key] = torch.ones(
                (dense_q_blocks, dense_kv_blocks),
                dtype=torch.bool,
                device=query.device,
            )
        return _radial_flashinfer_attention(
            query, key, value, block_mask_cache[dense_mask_key], vid_len, tail_len, block_size, pre_defined_mask,
            kv_video_len=kv_video_len,
        )

    # --- Sparge/Sage block-sparse path ---
    # Mirrors upstream radial_attn.attn_mask.SpargeSageAttnBackend. This path
    # uses the owned spas_sage_attn block_sparse_sage2_attn_cuda runtime instead
    # of importing training_free/ or an untracked checkout. The owned kernel
    # accepts partial final Q/K blocks, so non-upstream resolutions do not need
    # to be rejected at Python preflight time.
    if use_sage_attention:
        if context_len != 0:
            raise RuntimeError("radial use_sage_attention only supports layouts without a context prefix")
        if kv_video_len != vid_len:
            raise RuntimeError("radial use_sage_attention does not support rectangular LongCat q/kv lengths")
        bsr_cache_key = (vid_len, block_size, frame_size, num_frames, decay_factor, model_type, "sage")
        if bsr_cache_key not in block_mask_cache:
            block_mask_cache[bsr_cache_key] = _radial_bsr_mask(
                vid_len, block_size, frame_size, num_frames, decay_factor, model_type, query.device,
            )
        return _radial_sage_attention(
            query,
            key,
            value,
            block_mask_cache[bsr_cache_key],
            vid_len,
            tail_len,
            block_size,
            pre_defined_mask,
            block_sparse_sage2_attn_fn,
            sageattn_fn,
        )

    # --- flashinfer block-sparse path ---
    # Upstream radial-attention benchmarks use fixed BSR blocks; SparseVideo
    # keeps that path for divisible shapes and uses the same FlashInfer
    # variable-block wrapper as SVG2/SVOO for a partial final block.
    if query.is_cuda and context_len == 0:
        bsr_cache_key = (
            vid_len,
            kv_video_len,
            block_size,
            frame_size,
            num_frames,
            kv_num_frames,
            q_kv_offset,
            decay_factor,
            model_type,
        )
        if bsr_cache_key not in block_mask_cache:
            if kv_video_len == vid_len and q_kv_offset == 0:
                block_mask_cache[bsr_cache_key] = _radial_bsr_mask(
                    vid_len, block_size, frame_size, num_frames, decay_factor, model_type, query.device,
                )
            else:
                block_mask_cache[bsr_cache_key] = _radial_rectangular_bsr_mask(
                    vid_len,
                    kv_video_len,
                    block_size,
                    frame_size,
                    num_frames,
                    kv_num_frames,
                    q_kv_offset,
                    decay_factor,
                    model_type,
                    query.device,
                )
        bsr_2d = block_mask_cache[bsr_cache_key]  # [num_blocks, num_blocks] bool

        return _radial_flashinfer_attention(
            query, key, value, bsr_2d, vid_len, tail_len, block_size, pre_defined_mask,
            kv_video_len=kv_video_len,
        )

    reasons = []
    reasons = []
    if context_len != 0:
        reasons.append(f"context_len={context_len}")
    if not query.is_cuda:
        reasons.append("input is not on CUDA")
    reason_text = "; ".join(reasons) if reasons else "unsupported layout"
    raise RuntimeError(
        "radial sparse path requires FlashInfer block-sparse backend. "
        f"{reason_text}."
    )


def _radial_flashinfer_attention(
    query,
    key,
    value,
    video_mask,
    video_len,
    tail_len,
    block_size,
    pre_defined_mask=None,
    kv_video_len=None,
):
    from ...kernels.flashinfer_block_sparse import _ensure_cuda_home_for_flashinfer_jit, get_flashinfer

    _ensure_cuda_home_for_flashinfer_jit()
    flashinfer = get_flashinfer()

    B, _N, _H, _D = query.shape
    kv_video_len = video_len if kv_video_len is None else int(kv_video_len)
    if tail_len and pre_defined_mask is not None and B != 1:
        raise RuntimeError("radial Hunyuan attention_mask path currently supports batch size 1")

    video_result = _radial_flashinfer_video_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        kv_video_len,
        block_size,
        return_lse=bool(tail_len),
    )
    if not tail_len:
        return video_result

    video_video_o_all, video_video_o_lse_all = video_result

    outputs = []
    for batch_idx in range(B):
        q_b = query[batch_idx]
        k_b = key[batch_idx]
        v_b = value[batch_idx]
        custom_video_text = None
        custom_text = None
        if pre_defined_mask is not None:
            custom_video_text = pre_defined_mask[:video_len, video_len:]
            custom_text = pre_defined_mask[video_len:, :]

        video_text_o, video_text_o_lse = flashinfer.single_prefill_with_kv_cache(
            q=q_b[:video_len],
            k=k_b[video_len:],
            v=v_b[video_len:],
            causal=False,
            return_lse=True,
            custom_mask=custom_video_text,
        )
        out_video, _ = flashinfer.merge_state(
            v_a=video_video_o_all[batch_idx],
            s_a=video_video_o_lse_all[batch_idx],
            v_b=video_text_o,
            s_b=video_text_o_lse,
        )
        out_text = flashinfer.single_prefill_with_kv_cache(
            q=q_b[video_len:],
            k=k_b,
            v=v_b,
            causal=False,
            return_lse=False,
            custom_mask=custom_text,
        )
        outputs.append(torch.cat([out_video, out_text], dim=0))

    return torch.stack(outputs, dim=0)


def _radial_flashinfer_video_attention(
    query,
    key,
    value,
    video_mask,
    video_len,
    kv_video_len,
    block_size,
    *,
    return_lse=False,
):
    if video_len % block_size == 0 and kv_video_len % block_size == 0:
        return _radial_flashinfer_fixed_bsr_video_attention(
            query,
            key,
            value,
            video_mask,
            video_len,
            kv_video_len,
            block_size,
            return_lse=return_lse,
        )
    return _radial_flashinfer_variable_video_attention(
        query,
        key,
        value,
        video_mask,
        video_len,
        kv_video_len,
        block_size,
        return_lse=return_lse,
    )


def _radial_flashinfer_fixed_bsr_video_attention(
    query,
    key,
    value,
    video_mask,
    video_len,
    kv_video_len,
    block_size,
    *,
    return_lse=False,
):
    from ...kernels.flashinfer_block_sparse import build_bsr_from_mask, get_flashinfer

    flashinfer = get_flashinfer()

    B, _N, H, D = query.shape
    indptr, indices = build_bsr_from_mask(video_mask, query.device)
    workspace_buffer = torch.empty(128 * 1024 * 1024, device=query.device, dtype=torch.uint8)
    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer, backend="fa2")
    bsr_wrapper.plan(
        indptr=indptr,
        indices=indices,
        M=video_len,
        N=kv_video_len,
        R=block_size,
        C=block_size,
        num_qo_heads=H,
        num_kv_heads=H,
        head_dim=D,
        q_data_type=query.dtype,
        kv_data_type=key.dtype,
        o_data_type=query.dtype,
    )

    outputs = []
    lses = []
    for batch_idx in range(B):
        result = bsr_wrapper.run(
            query[batch_idx, :video_len].contiguous(),
            key[batch_idx, :kv_video_len].contiguous(),
            value[batch_idx, :kv_video_len].contiguous(),
            return_lse=return_lse,
        )
        if return_lse:
            out, lse = result
            outputs.append(out)
            lses.append(lse)
        else:
            outputs.append(result)
    del bsr_wrapper, workspace_buffer
    if return_lse:
        return torch.stack(outputs, dim=0), torch.stack(lses, dim=0)
    return torch.stack(outputs, dim=0)


def _radial_flashinfer_variable_video_attention(
    query,
    key,
    value,
    video_mask,
    video_len,
    kv_video_len,
    block_size,
    *,
    return_lse=False,
):
    from ...kernels.flashinfer_block_sparse import variable_block_sparse_attn

    B, _N, H, D = query.shape
    q_block_sizes = _radial_block_sizes(video_len, block_size, query.device)
    k_block_sizes = _radial_block_sizes(kv_video_len, block_size, query.device)
    num_q_blocks = int(q_block_sizes.shape[0])
    num_k_blocks = int(k_block_sizes.shape[0])
    if tuple(video_mask.shape) != (num_q_blocks, num_k_blocks):
        raise RuntimeError(
            "radial FlashInfer variable-block path expected video_mask shape "
            f"{num_q_blocks}x{num_k_blocks}; got {tuple(video_mask.shape)}"
        )

    batch_heads = B * H
    q = query[:, :video_len].permute(0, 2, 1, 3).reshape(batch_heads, video_len, D).contiguous()
    k = key[:, :kv_video_len].permute(0, 2, 1, 3).reshape(batch_heads, kv_video_len, D).contiguous()
    v = value[:, :kv_video_len].permute(0, 2, 1, 3).reshape(batch_heads, kv_video_len, D).contiguous()
    q_sizes = q_block_sizes.unsqueeze(0).expand(batch_heads, -1).contiguous()
    k_sizes = k_block_sizes.unsqueeze(0).expand(batch_heads, -1).contiguous()
    dynamic_map = (
        video_mask.to(device=query.device, dtype=torch.bool)
        .unsqueeze(0)
        .expand(batch_heads, -1, -1)
        .contiguous()
    )
    result = variable_block_sparse_attn(
        q,
        k,
        v,
        dynamic_map,
        q_sizes,
        k_sizes,
        return_lse=return_lse,
    )
    if return_lse:
        out, lse = result
        out = out.reshape(B, H, video_len, D).permute(0, 2, 1, 3).contiguous()
        lse = lse.reshape(B, H, video_len).permute(0, 2, 1).contiguous()
        return out, lse
    return result.reshape(B, H, video_len, D).permute(0, 2, 1, 3).contiguous()


def _radial_block_sizes(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = _ceil_div(int(length), int(block_size))
    sizes = torch.full((num_blocks,), int(block_size), dtype=torch.int32, device=device)
    remainder = int(length) % int(block_size)
    if remainder:
        sizes[-1] = remainder
    return sizes


def _radial_sage_attention(
    query,
    key,
    value,
    video_mask,
    video_len,
    tail_len,
    block_size,
    pre_defined_mask,
    block_sparse_sage2_attn_fn=None,
    sageattn_fn=None,
):
    if block_sparse_sage2_attn_fn is None:
        block_sparse_sage2_attn_fn = load_block_sparse_sage2_attn_function()

    B, N, H, _D = query.shape
    if tail_len and pre_defined_mask is None:
        pre_defined_mask = torch.ones((N, N), dtype=torch.bool, device=query.device)
    if tail_len and pre_defined_mask is not None and B != 1:
        raise RuntimeError("radial use_sage_attention Hunyuan attention_mask path currently supports batch size 1")

    if video_mask.all():
        if sageattn_fn is None:
            sageattn_fn = load_sageattn_function()
        return _radial_sage_dense_attention(
            query,
            key,
            value,
            video_len,
            tail_len,
            pre_defined_mask,
            sageattn_fn,
        )

    arch = _cuda_arch(query.device)
    q_block, k_block = _sparge_sage_qk_block_sizes(arch)
    video_mask = _radial_append_tail_blocks(video_mask, video_len, tail_len, block_size)
    converted_mask = _sparge_mask_convert(video_mask, block_size=block_size, arch=arch)
    converted_mask = (
        converted_mask[None, None, :, :]
        .expand(B, H, converted_mask.shape[0], converted_mask.shape[1])
        .to(torch.int8)
        .contiguous()
    )

    query_bhsd = query.permute(0, 2, 1, 3).contiguous()
    key_bhsd = key.permute(0, 2, 1, 3).contiguous()
    value_bhsd = value.permute(0, 2, 1, 3).contiguous()

    if pre_defined_mask is None:
        output = block_sparse_sage2_attn_fn(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            mask_id=converted_mask[
                :, :, :_ceil_div(query_bhsd.shape[2], q_block), :_ceil_div(key_bhsd.shape[2], k_block)
            ].contiguous(),
            tensor_layout="HND",
        )
        return output.permute(0, 2, 1, 3).contiguous()

    from ...kernels.flashinfer_block_sparse import _ensure_cuda_home_for_flashinfer_jit, get_flashinfer

    _ensure_cuda_home_for_flashinfer_jit()
    flashinfer = get_flashinfer()

    converted_mask = converted_mask.clone()
    kv_border = _ceil_div(int(pre_defined_mask[0].sum().item()), k_block)
    converted_mask[:, :, :, kv_border:] = False
    query_video_blocks = _ceil_div(video_len, q_block)
    output_video = block_sparse_sage2_attn_fn(
        query_bhsd[:, :, :video_len, :],
        key_bhsd,
        value_bhsd,
        mask_id=converted_mask[:, :, :query_video_blocks, :kv_border].contiguous(),
        tensor_layout="HND",
    )
    output_video = output_video.permute(0, 2, 1, 3).contiguous()

    output_text = []
    for batch_idx in range(B):
        text_out = flashinfer.single_prefill_with_kv_cache(
            q=query[batch_idx, video_len:],
            k=key[batch_idx, : pre_defined_mask[0].sum()],
            v=value[batch_idx, : pre_defined_mask[0].sum()],
            causal=False,
            return_lse=False,
        )
        output_text.append(text_out)
    return torch.cat([output_video, torch.stack(output_text, dim=0)], dim=1)


def _radial_sage_dense_attention(
    query,
    key,
    value,
    video_len,
    tail_len,
    pre_defined_mask,
    sageattn_fn,
):
    B, N, _H, _D = query.shape
    if tail_len and pre_defined_mask is None:
        pre_defined_mask = torch.ones((N, N), dtype=torch.bool, device=query.device)
    if tail_len and pre_defined_mask is not None and B != 1:
        raise RuntimeError("radial use_sage_attention Hunyuan dense path currently supports batch size 1")

    kv_border = int(pre_defined_mask[0].sum().item()) if pre_defined_mask is not None else key.shape[1]
    output_video = sageattn_fn(
        query[:, :video_len, :, :].contiguous(),
        key[:, :kv_border, :, :].contiguous(),
        value[:, :kv_border, :, :].contiguous(),
        tensor_layout="NHD",
    )

    if not tail_len:
        return output_video.contiguous()

    from ...kernels.flashinfer_block_sparse import _ensure_cuda_home_for_flashinfer_jit, get_flashinfer

    _ensure_cuda_home_for_flashinfer_jit()
    flashinfer = get_flashinfer()

    output_text = []
    for batch_idx in range(B):
        text_out = flashinfer.single_prefill_with_kv_cache(
            q=query[batch_idx, video_len:],
            k=key[batch_idx, :kv_border],
            v=value[batch_idx, :kv_border],
            causal=False,
            return_lse=False,
        )
        output_text.append(text_out)
    return torch.cat([output_video, torch.stack(output_text, dim=0)], dim=1).contiguous()


def _radial_append_tail_blocks(video_mask, video_len, tail_len, block_size):
    if tail_len <= 0:
        return video_mask
    video_blocks = _ceil_div(video_len, block_size)
    total_blocks = _ceil_div(video_len + tail_len, block_size)
    if total_blocks <= video_blocks:
        return video_mask

    full_mask = torch.ones(
        (total_blocks, total_blocks),
        dtype=video_mask.dtype,
        device=video_mask.device,
    )
    full_mask[:video_blocks, :video_blocks] = video_mask[:video_blocks, :video_blocks]
    return full_mask


def _expand_attention_mask(attention_mask, sequence_length, device):
    if attention_mask is None:
        return None
    mask = attention_mask
    while mask.ndim > 2:
        mask = mask[0]
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != sequence_length:
        raise RuntimeError(
            f"radial attention_mask length {mask.shape[-1]} does not match sequence length {sequence_length}"
        )
    return mask[0].to(device=device, dtype=torch.bool).expand(sequence_length, sequence_length)


def _cuda_arch(device: torch.device) -> str:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{major}{minor}"


def _sparge_sage_qk_block_sizes(arch: str) -> tuple[int, int]:
    if arch == "sm90":
        return 64, 128
    return 128, 64


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sparge_mask_convert(mask: torch.Tensor, block_size: int = 128, arch="sm") -> torch.Tensor:
    if block_size not in (64, 128):
        raise ValueError("Radial Attention only supports block size of 128 or 64")
    if mask.shape[0] != mask.shape[1]:
        raise ValueError("Input mask must be square.")

    if block_size == 128:
        if arch == "sm90":
            return torch.repeat_interleave(mask, 2, dim=0)
        return torch.repeat_interleave(mask, 2, dim=1)

    if arch == "sm90":
        num_row, num_col = mask.shape
        if num_col % 2:
            mask = torch.cat(
                [mask, torch.zeros((num_row, 1), dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
            num_row, num_col = mask.shape
        return torch.max(mask.view(num_row, num_col // 2, 2), dim=2).values

    num_row, num_col = mask.shape
    if num_row % 2:
        mask = torch.cat(
            [mask, torch.zeros((1, num_col), dtype=mask.dtype, device=mask.device)],
            dim=0,
        )
        num_row, num_col = mask.shape
    return torch.max(mask.view(num_row // 2, 2, num_col), dim=1).values


def _radial_bsr_mask(
    vid_len: int,
    block_size: int,
    frame_size: int,
    num_frames: int,
    decay_factor: float,
    model_type: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the same shrinked BSR mask as upstream radial_attn.attn_mask."""
    if block_size not in (64, 128):
        raise ValueError("Radial Attention only supports block size of 128 or 64")
    if device is None:
        device = torch.device("cpu")
    num_blocks = _ceil_div(vid_len, block_size)
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)
    radial_model_type = _radial_model_type(model_type)

    col_indices = torch.arange(0, frame_size, device=device).view(1, -1)
    row_indices = torch.arange(0, frame_size, device=device).view(-1, 1)

    for i in range(num_frames):
        for j in range(num_frames):
            if j == 0 and radial_model_type == "wan":
                local_mask = torch.ones((frame_size, frame_size), dtype=torch.bool, device=device)
            else:
                window_width = _radial_window_width(
                    i, j, frame_size, decay_factor, block_size, radial_model_type,
                )
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                if not _radial_diagonal_split(i, j, frame_size):
                    local_mask = torch.zeros_like(local_mask)

            remainder_row = (i * frame_size) % block_size
            remainder_col = (j * frame_size) % block_size
            all_length_row = _padded_block_extent(remainder_row, frame_size, block_size)
            all_length_col = _padded_block_extent(remainder_col, frame_size, block_size)

            padded_local_mask = torch.zeros((all_length_row, all_length_col), dtype=torch.bool, device=device)
            padded_local_mask[
                remainder_row:remainder_row + frame_size,
                remainder_col:remainder_col + frame_size,
            ] = local_mask

            block_mask = _shrink_mask_strict(padded_local_mask, block_size)
            if vid_len % block_size:
                block_mask = torch.logical_or(
                    block_mask,
                    _shrink_mask_variable_rows(
                        padded_local_mask,
                        block_size,
                        valid_row_start=remainder_row,
                        valid_row_length=frame_size,
                    ),
                )
            block_row_start = (i * frame_size) // block_size
            block_col_start = (j * frame_size) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                mask[block_row_start:block_row_end, block_col_start:block_col_end],
                block_mask,
            )

    return mask


def _radial_rectangular_bsr_mask(
    q_vid_len: int,
    kv_vid_len: int,
    block_size: int,
    frame_size: int,
    q_num_frames: int,
    kv_num_frames: int,
    q_kv_offset: int,
    decay_factor: float,
    model_type: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    if block_size not in (64, 128):
        raise ValueError("Radial Attention only supports block size of 128 or 64")
    if device is None:
        device = torch.device("cpu")
    if q_kv_offset % frame_size != 0:
        raise ValueError("q_kv_offset must align to whole frames")

    q_blocks = _ceil_div(q_vid_len, block_size)
    kv_blocks = _ceil_div(kv_vid_len, block_size)
    mask = torch.zeros(q_blocks, kv_blocks, dtype=torch.bool, device=device)
    radial_model_type = _radial_model_type(model_type)
    prefix_frames = int(q_kv_offset) // int(frame_size)

    col_indices = torch.arange(0, frame_size, device=device).view(1, -1)
    row_indices = torch.arange(0, frame_size, device=device).view(-1, 1)

    for i in range(q_num_frames):
        q_frame = prefix_frames + i
        for j in range(kv_num_frames):
            if j < prefix_frames or (j == 0 and radial_model_type == "wan"):
                local_mask = torch.ones((frame_size, frame_size), dtype=torch.bool, device=device)
            else:
                window_width = _radial_window_width(
                    q_frame, j, frame_size, decay_factor, block_size, radial_model_type,
                )
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                if not _radial_diagonal_split(q_frame, j, frame_size):
                    local_mask = torch.zeros_like(local_mask)

            q_start = i * frame_size
            kv_start = j * frame_size
            remainder_row = q_start % block_size
            remainder_col = kv_start % block_size
            all_length_row = _padded_block_extent(remainder_row, frame_size, block_size)
            all_length_col = _padded_block_extent(remainder_col, frame_size, block_size)

            padded_local_mask = torch.zeros((all_length_row, all_length_col), dtype=torch.bool, device=device)
            padded_local_mask[
                remainder_row:remainder_row + frame_size,
                remainder_col:remainder_col + frame_size,
            ] = local_mask

            block_mask = _shrink_mask_strict(padded_local_mask, block_size)
            if q_vid_len % block_size or kv_vid_len % block_size:
                block_mask = torch.logical_or(
                    block_mask,
                    _shrink_mask_variable_rows(
                        padded_local_mask,
                        block_size,
                        valid_row_start=remainder_row,
                        valid_row_length=frame_size,
                    ),
                )
            block_row_start = q_start // block_size
            block_col_start = kv_start // block_size
            block_row_end = min(block_row_start + block_mask.shape[0], q_blocks)
            block_col_end = min(block_col_start + block_mask.shape[1], kv_blocks)
            block_mask = block_mask[:block_row_end - block_row_start, :block_col_end - block_col_start]
            mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                mask[block_row_start:block_row_end, block_col_start:block_col_end],
                block_mask,
            )

    return mask


def _padded_block_extent(remainder: int, length: int, block_size: int) -> int:
    return _ceil_div(int(remainder) + int(length), int(block_size)) * int(block_size)


def _radial_model_type(model_type: str) -> str:
    if model_type == "hunyuan_video":
        return "hunyuan"
    if model_type == "wan":
        return "wan"
    if model_type in ("cogvideox", "ltx_video", "allegro", "mochi", "easyanimate"):
        return "hunyuan"
    raise ValueError(f"Unknown model type: {model_type}")


def _radial_window_width(i, j, frame_size, decay_factor, block_size, radial_model_type):
    dist = abs(i - j)
    if radial_model_type == "wan":
        if dist < 1:
            return frame_size
        if dist == 1:
            return frame_size
    elif radial_model_type == "hunyuan":
        if dist <= 1:
            return frame_size
    else:
        raise ValueError(f"Unknown model type: {radial_model_type}")
    group = dist.bit_length()
    decay_length = 2 ** frame_size.bit_length() / 2 ** group * decay_factor
    threshold = block_size
    if decay_length >= threshold:
        return decay_length
    return threshold


def _radial_diagonal_split(i, j, frame_size):
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128
    decay_length = 2 ** frame_size.bit_length() / 2 ** group
    if decay_length >= threshold:
        return True
    split_factor = int(threshold / decay_length)
    return dist % split_factor == 0


def _shrink_mask_strict(mask, block_size=128):
    row_blocks = mask.shape[0] // block_size
    col_blocks = mask.shape[1] // block_size
    mask = mask[:row_blocks * block_size, :col_blocks * block_size].view(
        row_blocks, block_size, col_blocks, block_size,
    )
    col_densities = mask.sum(dim=1) / block_size
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (
        non_zero_densities.sum(dim=-1) + 1e-9
    )
    return frac_high_density_cols > 0.6


def _shrink_mask_variable_rows(mask, block_size=128, *, valid_row_start: int, valid_row_length: int):
    row_blocks = mask.shape[0] // block_size
    col_blocks = mask.shape[1] // block_size
    mask = mask[:row_blocks * block_size, :col_blocks * block_size].view(
        row_blocks, block_size, col_blocks, block_size,
    )
    row_offsets = torch.arange(row_blocks * block_size, device=mask.device).view(row_blocks, block_size)
    valid_rows = (row_offsets >= int(valid_row_start)) & (
        row_offsets < int(valid_row_start) + int(valid_row_length)
    )
    row_counts = valid_rows.sum(dim=1).clamp_min(1).view(row_blocks, 1, 1)
    col_densities = mask.sum(dim=1) / row_counts
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1 / 3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (
        non_zero_densities.sum(dim=-1) + 1e-9
    )
    return frac_high_density_cols > 0.6


def _estimate_frame_size(video_len):
    # Kept as a compatibility helper for older tests; runtime uses
    # infer_video_frame_shape so upstream 18/30-frame layouts are not guessed.
    for nf, frame_size in (
        (18, 48 * 80),
        (30, 48 * 80),
        (21, 45 * 80),
        (33, 45 * 80),
    ):
        if video_len == nf * frame_size:
            return frame_size
    for nf in (33, 30, 25, 21, 18, 17, 13, 9, 5):
        if video_len % nf == 0:
            return video_len // nf
    return video_len // 13 if video_len >= 13 else video_len
