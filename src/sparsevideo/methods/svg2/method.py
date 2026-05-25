from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from .._schedule import configured_dense_warmup_layer_count, configured_dense_warmup_requires_dense, runtime_num_inference_steps, scheduler_timestep_from_tracker
from ...kernels.dynamic_map import identify_dynamic_map
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from . import config as method_config


class SVG2Method(SparseMethod):
    """SVG2: k-means clustering + block-sparse attention.

    Port of the second Sparse-VideoGen method.
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
            raise NotImplementedError(f"svg2 not yet supported for {self.model_info.model_type}")

        cfg = self.config
        first_layer_count = configured_dense_warmup_layer_count(cfg, total_layers)
        state = _new_runtime_state()

        def attn_fn(query, key, value, attention_mask, **kwargs):
            scheduler_timestep = scheduler_timestep_from_tracker(step_tracker, kwargs)
            runtime_state = _state_for_cache_suffix(state, kwargs.get("cache_key_suffix"))
            prompt_length = kwargs.get("prompt_length")
            if prompt_length is None:
                prompt_length = cfg.get("prompt_length")
            full_attention = (
                layer_idx < first_layer_count
                or configured_dense_warmup_requires_dense(
                    cfg,
                    runtime_num_inference_steps(step_tracker),
                    step_tracker.step,
                    scheduler_timestep,
                    notifier=self.warmup_notifier,
                )
            )
            if full_attention:
                if (
                    cfg["zero_step_kmeans_init"]
                    and query.is_cuda
                    and (
                        attention_mask is None
                        or self.model_info.model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate")
                    )
                ):
                    _svg2_attention(
                        query, key, value,
                        top_p_kmeans=cfg["top_p_kmeans"],
                        min_kc_ratio=cfg["min_kc_ratio"],
                        num_q_centroids=cfg["num_q_centroids"],
                        num_k_centroids=cfg["num_k_centroids"],
                        kmeans_iter_init=cfg["kmeans_iter_init"],
                        kmeans_iter_step=cfg["kmeans_iter_step"],
                        state=runtime_state,
                        initialize_only=True,
                        model_type=self.model_info.model_type,
                        text_len=kwargs.get("text_len", 0),
                        prompt_length=prompt_length,
                        context_length=cfg.get("context_length"),
                    )
                    self.record_runtime_dispatch(
                        "initialize_only",
                        backend="triton_kmeans",
                        layer_idx=layer_idx,
                        step=getattr(step_tracker, "step", None),
                    )
                out = _svg2_dense_attention(
                    query, key, value, attention_mask,
                    model_type=self.model_info.model_type,
                )
                self.record_runtime_dispatch(
                    "dense",
                    backend=_svg2_dense_backend_name(query, attention_mask, self.model_info.model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("svg2 sparse path requires CUDA self-attention without an attention mask")
            if (
                attention_mask is not None
                and self.model_info.model_type not in ("hunyuan_video", "cogvideox", "mochi", "easyanimate")
            ):
                raise RuntimeError("svg2 sparse path requires CUDA self-attention without an attention mask")
            out = _svg2_attention(
                query, key, value,
                top_p_kmeans=cfg["top_p_kmeans"],
                min_kc_ratio=cfg["min_kc_ratio"],
                num_q_centroids=cfg["num_q_centroids"],
                num_k_centroids=cfg["num_k_centroids"],
                kmeans_iter_init=cfg["kmeans_iter_init"],
                kmeans_iter_step=cfg["kmeans_iter_step"],
                state=runtime_state,
                model_type=self.model_info.model_type,
                text_len=kwargs.get("text_len", 0),
                prompt_length=prompt_length,
                context_length=cfg.get("context_length"),
            )
            self.record_runtime_dispatch(
                "sparse",
                backend="flashinfer",
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


def _new_runtime_state():
    return {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
    }


def _state_for_cache_suffix(state, cache_key_suffix):
    if cache_key_suffix is None:
        return state
    suffix_states = state.setdefault("cache_suffix_states", {})
    return suffix_states.setdefault(cache_key_suffix, _new_runtime_state())


def _svg2_dense_backend_name(query, attention_mask, model_type):
    if (
        model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate")
        and attention_mask is not None
        and query.is_cuda
    ):
        return "svg2_flashinfer_varlen"
    if attention_mask is not None:
        return "diffusers_dispatch"
    return "torch_sdpa"


def _svg2_attention(query, key, value, top_p_kmeans, min_kc_ratio,
                    num_q_centroids, num_k_centroids, kmeans_iter_init,
                    kmeans_iter_step, state, initialize_only=False,
                    model_type="wan",
                    text_len=0, prompt_length=None, context_length=None):
    """SVG2: k-means clustering + block-sparse attention.

    query/key/value: [B, N, H, D]
    Sparse attention backend: flashinfer VariableBlockSparseAttentionWrapper.
    """
    from ...kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn
    from .kmeans import triton_kmeans

    B, N, H, D = query.shape

    # CogVideoX classifier-free guidance arrives as a batch of negative and
    # positive prompts. The owned kernels operate on folded batch-head slots,
    # so each batch item remains independent without serializing the sparse path.
    q_full = query.permute(0, 2, 1, 3).contiguous().reshape(B * H, N, D)
    k_full = key.permute(0, 2, 1, 3).contiguous().reshape(B * H, N, D)
    v_full = value.permute(0, 2, 1, 3).contiguous().reshape(B * H, N, D)

    text_len = int(text_len or 0)
    if model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate") and text_len > 0:
        if context_length is not None and int(context_length) != text_len:
            raise RuntimeError(
                "svg2 context_length must match the text token tail length "
                f"seen by the processor; got context_length={int(context_length)}, text_len={text_len}"
            )
        video_len = N - text_len
        if video_len <= 0:
            raise RuntimeError("svg2 hunyuan sparse path could not find video tokens")
        prompt_length = _resolve_svg2_prompt_length(prompt_length, text_len)
        q_flat = q_full[:, :video_len, :].contiguous()
        k_flat = k_full[:, :video_len, :].contiguous()
        v_flat = v_full[:, :video_len, :].contiguous()
    else:
        video_len = N
        prompt_length = 0
        q_flat = q_full
        k_flat = k_full
        v_flat = v_full

    nqc = min(num_q_centroids, video_len)
    nkc = min(num_k_centroids, video_len)

    kmeans_iters = kmeans_iter_step if state["centroids_init"] else kmeans_iter_init
    q_labels, q_centroids, q_sizes = triton_kmeans(
        q_flat,
        nqc,
        kmeans_iters,
        init_centroids=state.get("prev_q_centroids"),
        final_reassign=False,
    )
    k_labels, k_centroids, k_sizes = triton_kmeans(
        k_flat,
        nkc,
        kmeans_iters,
        init_centroids=state.get("prev_k_centroids"),
        final_reassign=False,
    )
    state["centroids_init"] = True
    state["prev_q_centroids"] = q_centroids.detach()
    state["prev_k_centroids"] = k_centroids.detach()

    if initialize_only:
        return None

    dynamic_map = identify_dynamic_map(
        q_centroids,
        k_centroids,
        q_sizes,
        k_sizes,
        top_p_kmeans,
        min_kc_ratio,
    )

    # Sort tokens by cluster. CUDA dispatches the package-owned Triton
    # permutation kernels used by Sparse-VideoGen/SVOO instead of silently using
    # the slower PyTorch gather path.
    q_sorted, q_sorted_idx = _svg2_permute_by_labels(q_flat, q_labels)
    k_sorted, k_sorted_idx = _svg2_permute_by_labels(k_flat, k_labels)
    v_sorted, _ = _svg2_permute_by_sorted_indices(v_flat, k_sorted_idx)

    if model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate") and text_len > 0:
        q_sorted, k_sorted, v_sorted, dynamic_map, q_sizes, k_sizes, q_sorted_idx = _svg2_append_hunyuan_text_clusters(
            q_sorted,
            k_sorted,
            v_sorted,
            q_full,
            k_full,
            v_full,
            dynamic_map,
            q_sizes,
            k_sizes,
            q_sorted_idx,
            video_len=video_len,
            text_len=text_len,
            prompt_length=prompt_length,
        )

    # Block-sparse attention: FlashInfer parity path.
    q_sizes_i32 = q_sizes.to(torch.int32)
    k_sizes_i32 = k_sizes.to(torch.int32)
    if not HAS_FLASHINFER:
        raise RuntimeError("svg2 sparse path requires flashinfer.sparse")
    out_sorted = variable_block_sparse_attn(
        q_sorted, k_sorted, v_sorted,
        dynamic_map, q_sizes_i32, k_sizes_i32,
    )

    # Unsort
    out_flat = _svg2_inverse_permutation(out_sorted, q_sorted_idx)

    return out_flat.reshape(B, H, N, D).permute(0, 2, 1, 3)


def _svg2_dense_attention(query, key, value, attention_mask, *, model_type):
    if (
        model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate")
        and attention_mask is not None
        and query.is_cuda
    ):
        return _svg2_hunyuan_flashinfer_varlen(query, key, value, attention_mask)
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


def _svg2_hunyuan_flashinfer_varlen(query, key, value, attention_mask):
    if query.shape[0] != 1:
        raise RuntimeError("SVG2 Hunyuan FlashInfer varlen path follows upstream batch size 1")

    batch, seq_len, heads, dim = query.shape
    valid_len = int(attention_mask.sum().item())
    total_len = int(attention_mask.numel())
    if total_len != seq_len:
        raise RuntimeError(
            "SVG2 Hunyuan FlashInfer varlen path requires a total-sequence attention_mask; "
            f"got attention_mask length {total_len} for sequence length {seq_len}."
        )

    from ...kernels.flashinfer_block_sparse import hunyuan_flashinfer_varlen_attn

    q = query.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    k = key.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    v = value.permute(0, 2, 1, 3).reshape(batch * heads, seq_len, dim).contiguous()
    hidden_states = hunyuan_flashinfer_varlen_attn(
        q, k, v,
        valid_len=valid_len,
    )
    return hidden_states.reshape(batch, heads, seq_len, dim).permute(0, 2, 1, 3).contiguous()


def _svg2_permute_by_labels(tensor, labels):
    if tensor.is_cuda:
        from ...kernels.permute import permute_tensor_by_labels_triton

        out, sorted_indices = permute_tensor_by_labels_triton(
            tensor.unsqueeze(0),
            labels,
            dim=2,
        )
        return out.squeeze(0), sorted_indices

    sorted_indices = labels.long().argsort(dim=-1)
    return _svg2_gather_by_sorted_indices(tensor, sorted_indices), sorted_indices


def _svg2_permute_by_sorted_indices(tensor, sorted_indices):
    if tensor.is_cuda:
        from ...kernels.permute import permute_tensor_by_labels_triton

        out, sorted_indices = permute_tensor_by_labels_triton(
            tensor.unsqueeze(0),
            None,
            dim=2,
            sorted_indices=sorted_indices,
        )
        return out.squeeze(0), sorted_indices

    return _svg2_gather_by_sorted_indices(tensor, sorted_indices), sorted_indices


def _svg2_inverse_permutation(tensor, sorted_indices):
    if tensor.is_cuda:
        from ...kernels.permute import apply_inverse_permutation_triton

        out = apply_inverse_permutation_triton(
            tensor.unsqueeze(0),
            sorted_indices,
            dim=2,
        )
        return out.squeeze(0)

    inverse_indices = sorted_indices.long().argsort(dim=-1)
    return _svg2_gather_by_sorted_indices(tensor, inverse_indices)


def _svg2_gather_by_sorted_indices(tensor, sorted_indices):
    return torch.gather(
        tensor,
        1,
        sorted_indices.long().unsqueeze(-1).expand(-1, -1, tensor.shape[-1]),
    )


def _resolve_svg2_prompt_length(prompt_length, text_len):
    if prompt_length is None:
        return int(text_len)
    if isinstance(prompt_length, torch.Tensor):
        return int(prompt_length.detach().flatten()[0].item())
    return int(prompt_length)


def _svg2_append_hunyuan_text_clusters(
    q_sorted,
    k_sorted,
    v_sorted,
    q_full,
    k_full,
    v_full,
    dynamic_map,
    q_sizes,
    k_sizes,
    q_sorted_idx,
    *,
    video_len,
    text_len,
    prompt_length,
):
    """Match upstream Hunyuan SVG2/SAP post-processing for prompt/fake text."""
    prompt_length = max(0, min(int(prompt_length), int(text_len)))
    unprompt_length = int(text_len) - prompt_length

    q_sorted = torch.cat([q_sorted, q_full[:, video_len:, :]], dim=1)
    k_sorted = torch.cat([k_sorted, k_full[:, video_len:, :]], dim=1)
    v_sorted = torch.cat([v_sorted, v_full[:, video_len:, :]], dim=1)

    dynamic_map = F.pad(dynamic_map, (0, 2, 0, 2), value=False)
    dynamic_map[:, -2, :-1] = True
    dynamic_map[:, :-1, -2] = True
    dynamic_map[:, -1, -1] = True

    q_sizes = F.pad(q_sizes, (0, 2), value=0)
    q_sizes[:, -2] = prompt_length
    q_sizes[:, -1] = unprompt_length
    k_sizes = F.pad(k_sizes, (0, 2), value=0)
    k_sizes[:, -2] = prompt_length
    k_sizes[:, -1] = unprompt_length

    tail_indices = torch.arange(video_len, video_len + text_len, device=q_sorted_idx.device)
    tail_indices = tail_indices.unsqueeze(0).expand(q_sorted_idx.shape[0], -1)
    q_sorted_idx = torch.cat([q_sorted_idx, tail_indices], dim=1)
    return q_sorted, k_sorted, v_sorted, dynamic_map, q_sizes, k_sizes, q_sorted_idx
