from __future__ import annotations

import torch
import torch.nn.functional as F

from .._base import SparseMethod
from ...processors.allegro import SparseAllegroAttnProcessor
from ...processors.cogvideox import SparseCogVideoXAttnProcessor
from ...processors.easyanimate import SparseEasyAnimateAttnProcessor
from ...processors.wan import SparseWanAttnProcessor
from ...processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from ...processors.ltx_video import SparseLTXVideoAttnProcessor
from ...processors.mochi import SparseMochiAttnProcessor
from . import config as method_config


class AdaClusterMethod(SparseMethod):
    """AdaCluster: Triton k-means clustering + block-sparse attention.

    Uses upstream topk_num/q_kernel_num/kv_kernel_num naming. Hunyuan applies
    the hardcoded upstream dense gates before entering the sparse path.

    Port of:
    - training_free/Adacluster/triton_kernel/fast_kmeans_single.py
    - training_free/Adacluster/triton_kernel/triton_cluster_sparse_attn.py
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
            raise NotImplementedError(f"adacluster not yet supported for {self.model_info.model_type}")

        cfg = self.config
        state = {
            "centroids_init": False,
            "prev_q_centroids": None,
            "prev_k_centroids": None,
            "q_kernel_num": None,
            "kv_kernel_num": None,
            "use_full_attention": False,
        }
        model_type = self.model_info.model_type

        def attn_fn(query, key, value, attention_mask, **kwargs):
            if (
                model_type == "hunyuan_video"
                and _adacluster_hunyuan_uses_dense(layer_idx, step_tracker.step)
            ):
                if attention_mask is not None:
                    key, value = _adacluster_trim_hunyuan_kv(key, value, attention_mask)
                    attention_mask = None
                out = _adacluster_dense_attention(query, key, value, model_type=model_type)
                self.record_runtime_dispatch(
                    "dense",
                    backend=_adacluster_dense_backend_name(query, model_type),
                    layer_idx=layer_idx,
                    step=getattr(step_tracker, "step", None),
                )
                return out
            if not query.is_cuda:
                raise RuntimeError("adacluster sparse path requires CUDA self-attention without an attention mask")
            if attention_mask is not None:
                if model_type != "hunyuan_video":
                    raise RuntimeError("adacluster sparse path requires CUDA self-attention without an attention mask")
                key, value = _adacluster_trim_hunyuan_kv(key, value, attention_mask)
            topk_num = cfg["topk_num"]
            q_kernel_num = cfg["q_kernel_num"]
            kv_kernel_num = cfg["kv_kernel_num"]
            backend_trace = []
            out = _adacluster_attention(
                query, key, value,
                topk_num=topk_num,
                q_kernel_num=q_kernel_num,
                kv_kernel_num=kv_kernel_num,
                kmeans_iter_init=cfg["kmeans_iter_init"],
                kmeans_iter_step=cfg["kmeans_iter_step"],
                state=state,
                topk_policy="minmax" if model_type == "hunyuan_video" else "cluster_attn",
                reuse_prev_centroids=False if model_type == "hunyuan_video" else "both",
                model_type=model_type,
                thresholded_kmeans_config=_adacluster_thresholded_config(cfg)
                if model_type == "wan" and cfg["use_thresholded_kmeans_loop"]
                else None,
                backend_trace=backend_trace,
            )
            dispatch = "dense" if backend_trace and backend_trace[-1] in ("torch_sdpa", "flash_attn") else "sparse"
            self.record_runtime_dispatch(
                dispatch,
                backend=backend_trace[-1] if backend_trace else None,
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


def _adacluster_hunyuan_uses_dense(layer_idx: int, step: int) -> bool:
    """Upstream Hunyuan AdaCluster dense gates.

    training_free/Adacluster/runhunyuan uses full attention for the first
    eight denoising steps, layers <= 17, and layers 34/38/39.
    StepTracker is one-based after the first transformer forward.
    """
    return step <= 8 or layer_idx <= 17 or layer_idx in (34, 38, 39)


def _adacluster_topk_from_qkv_minmax(query: torch.Tensor, key: torch.Tensor, topk: int) -> torch.Tensor:
    q_pos = torch.clamp(query, min=0.0)
    q_neg = torch.clamp(query, max=0.0)
    k_pos = torch.clamp(key, min=0.0)
    k_neg = torch.clamp(key, max=0.0)
    score = torch.matmul(q_pos, k_pos.transpose(-2, -1)) + torch.matmul(q_neg, k_neg.transpose(-2, -1))
    return score.topk(k=topk, dim=-1).indices


def _adacluster_trim_hunyuan_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_length = _adacluster_hunyuan_kv_length(attention_mask, key.shape[1])
    return key[:, :kv_length], value[:, :kv_length]


def _adacluster_hunyuan_kv_length(attention_mask: torch.Tensor, max_length: int) -> int:
    # Upstream uses torch.sum(attention_mask) as the real KV length.
    kv_length = int(attention_mask.sum().item())
    return max(1, min(max_length, kv_length))


def _shared_random_centroids(x: torch.Tensor, n_clusters: int) -> torch.Tensor:
    _, N, D = x.shape
    idx = torch.randperm(N, device=x.device)[:n_clusters]
    return torch.gather(x, 1, idx.view(1, -1, 1).expand(x.shape[0], -1, D)).contiguous()


def _shared_random_centroids_bhsd(x: torch.Tensor, n_clusters: int) -> torch.Tensor:
    idx = torch.randperm(x.shape[2], device=x.device)[:n_clusters]
    return x[:, :, idx, :].contiguous()


def _adacluster_flash_kmeans_single(kernel: torch.Tensor, data: torch.Tensor, iter_time: int):
    from ...kernels.native.adacluster.fast_kmeans_single import flash_kmeans_single

    return flash_kmeans_single(kernel, data, iter_time)


def _adacluster_cluster_sparse_attn(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compressed_attn_mask: torch.Tensor,
    q_counts: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    from ...kernels.native.adacluster.triton_cluster_sparse_attn import triton_cluster_sparse_attn

    return triton_cluster_sparse_attn(
        query=query,
        key=key,
        value=value,
        compressed_attn_mask=compressed_attn_mask,
        q_counts=q_counts,
        kv_counts=kv_counts,
        sm_scale=sm_scale,
    )


def _adacluster_thresholded_config(cfg):
    return {
        "initial_q_kernel_num": cfg["initial_q_kernel_num"],
        "initial_kv_kernel_num": cfg["initial_kv_kernel_num"],
        "q_distance_threshold": cfg["q_distance_threshold"],
        "kv_distance_threshold": cfg["kv_distance_threshold"],
        "thresholded_kmeans_iter_time": cfg["thresholded_kmeans_iter_time"],
        "thresholded_kmeans_max_iterations": cfg["thresholded_kmeans_max_iterations"],
    }


def _adacluster_dense_attention(query, key, value, *, model_type="wan"):
    if model_type == "hunyuan_video" and query.is_cuda:
        return _adacluster_hunyuan_flash_attn(query, key, value)

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd,
        dropout_p=0.0, is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def _adacluster_dense_backend_name(query, model_type):
    if model_type == "hunyuan_video" and query.is_cuda:
        return "flash_attn"
    return "torch_sdpa"


def _adacluster_hunyuan_flash_attn(query, key, value):
    flash_attn_func = _load_flash_attn_func()
    return flash_attn_func(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        causal=False,
        softmax_scale=1.0 / (query.shape[-1] ** 0.5),
    )


def _load_flash_attn_func():
    from flash_attn import flash_attn_func

    return flash_attn_func


def _adacluster_attention(query, key, value, topk_num, q_kernel_num, kv_kernel_num,
                          kmeans_iter_init, kmeans_iter_step, state,
                          topk_policy="cluster_attn", reuse_prev_centroids=True,
                          thresholded_kmeans_config=None, model_type="wan", backend_trace=None):
    """AdaCluster: upstream fast_kmeans_single + triton_cluster_sparse_attn.

    query/key/value: [B, N, H, D]
    """
    B, NQ, H, D = query.shape
    original_head_dim = D
    NK = key.shape[1]
    scale = original_head_dim ** -0.5

    if state.get("use_full_attention"):
        if backend_trace is not None:
            backend_trace.append(_adacluster_dense_backend_name(query, model_type))
        return _adacluster_dense_attention(query, key, value, model_type=model_type)

    kernel_head_dim = _adacluster_kernel_head_dim(original_head_dim)
    if kernel_head_dim != original_head_dim:
        query, key, value = _adacluster_pad_head_dim(query, key, value, kernel_head_dim)
        D = kernel_head_dim

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()

    nqc = min(q_kernel_num, NQ)
    nkc = min(kv_kernel_num, NK)
    if thresholded_kmeans_config is not None:
        if not state["centroids_init"]:
            nkc = _adacluster_thresholded_kmeans_count(
                k_bhsd.reshape(B * H, NK, D),
                initial_clusters=thresholded_kmeans_config["initial_kv_kernel_num"],
                iter_time=thresholded_kmeans_config["thresholded_kmeans_iter_time"],
                distance_threshold=thresholded_kmeans_config["kv_distance_threshold"],
                max_iterations=thresholded_kmeans_config["thresholded_kmeans_max_iterations"],
                num_heads=H,
            )
            if nkc == -1:
                state["use_full_attention"] = True
                if backend_trace is not None:
                    backend_trace.append(_adacluster_dense_backend_name(query, model_type))
                return _adacluster_dense_attention(query, key, value, model_type=model_type)

            nqc = _adacluster_thresholded_kmeans_count(
                q_bhsd.reshape(B * H, NQ, D),
                initial_clusters=thresholded_kmeans_config["initial_q_kernel_num"],
                iter_time=thresholded_kmeans_config["thresholded_kmeans_iter_time"],
                distance_threshold=thresholded_kmeans_config["q_distance_threshold"],
                max_iterations=thresholded_kmeans_config["thresholded_kmeans_max_iterations"],
                num_heads=H,
            )
            if nqc == -1:
                state["use_full_attention"] = True
                if backend_trace is not None:
                    backend_trace.append(_adacluster_dense_backend_name(query, model_type))
                return _adacluster_dense_attention(query, key, value, model_type=model_type)
            state["q_kernel_num"] = int(nqc)
            state["kv_kernel_num"] = int(nkc)
        else:
            nqc = int(state["q_kernel_num"])
            nkc = int(state["kv_kernel_num"])
    nqc = max(1, min(int(nqc), NQ))
    nkc = max(1, min(int(nkc), NK))
    kmeans_iters = kmeans_iter_step if state["centroids_init"] else kmeans_iter_init
    reuse_q, reuse_k = _adacluster_reuse_policy(reuse_prev_centroids)
    if state["centroids_init"] and reuse_q:
        q_init = state.get("prev_q_centroids")
    else:
        q_init = _shared_random_centroids_bhsd(q_bhsd, nqc)
    if state["centroids_init"] and reuse_k:
        k_init = state.get("prev_k_centroids")
    else:
        k_init = _shared_random_centroids_bhsd(k_bhsd, nkc)

    q_centroids, q_sizes, q_labels = _adacluster_flash_kmeans_single(
        q_init,
        q_bhsd,
        kmeans_iters,
    )
    k_centroids, k_sizes, k_labels = _adacluster_flash_kmeans_single(
        k_init,
        k_bhsd,
        kmeans_iters,
    )
    state["centroids_init"] = True
    if reuse_q or reuse_k:
        state["prev_q_centroids"] = q_centroids.detach()
        state["prev_k_centroids"] = k_centroids.detach()

    cluster_scores = torch.matmul(k_centroids, q_centroids.transpose(2, 3)) * scale
    cluster_bias = torch.where(
        k_sizes > 0,
        torch.log(k_sizes),
        torch.finfo(cluster_scores.dtype).min,
    )
    cluster_attn = F.softmax(
        (cluster_scores + cluster_bias).transpose(2, 3),
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    k_keep = min(cluster_attn.size(-1), max(1, int(topk_num)))
    if topk_policy == "minmax":
        topk_idx = _adacluster_topk_from_qkv_minmax(q_centroids, k_centroids, k_keep)
    else:
        _, topk_idx = torch.topk(cluster_attn, k=k_keep, dim=-1)
    compressed_mask = torch.zeros_like(cluster_attn, dtype=torch.bool)
    compressed_mask.scatter_(dim=-1, index=topk_idx, value=True)

    k_counts = k_sizes.squeeze(-1).to(torch.int32)
    q_counts = q_sizes.squeeze(-1).to(torch.int32)
    k_counts = torch.cumsum(k_counts, dim=-1).to(torch.int32).contiguous()
    q_counts = torch.cumsum(q_counts, dim=-1).to(torch.int32).contiguous()

    # Sort by cluster and compute block-sparse attention
    q_sorted_idx = q_labels.long().argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)

    q_sorted = torch.gather(q_bhsd, 2, q_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, D)).contiguous()
    k_sorted = torch.gather(k_bhsd, 2, k_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, D)).contiguous()
    v_sorted = torch.gather(v_bhsd, 2, k_sorted_idx.unsqueeze(-1).expand(-1, -1, -1, D)).contiguous()

    out_sorted = _adacluster_cluster_sparse_attn(
        query=q_sorted,
        key=k_sorted,
        value=v_sorted,
        compressed_attn_mask=compressed_mask.contiguous(),
        q_counts=q_counts,
        kv_counts=k_counts,
        sm_scale=scale,
    )

    # Unsort
    inv_q_idx = q_sorted_idx.argsort(dim=-1)
    out_bhsd = torch.gather(out_sorted, 2, inv_q_idx.unsqueeze(-1).expand(-1, -1, -1, D))

    if backend_trace is not None:
        backend_trace.append("triton_cluster_sparse_attn")
    return out_bhsd[..., :original_head_dim].permute(0, 2, 1, 3).contiguous()


def _adacluster_kernel_head_dim(head_dim: int) -> int:
    for supported in (16, 32, 64, 128, 256):
        if head_dim <= supported:
            return supported
    raise RuntimeError(f"adacluster cannot pad head_dim={head_dim} to a supported kernel width")


def _adacluster_pad_head_dim(query, key, value, kernel_head_dim: int):
    pad = (0, kernel_head_dim - query.shape[-1])
    return F.pad(query, pad), F.pad(key, pad), F.pad(value, pad)


def _adacluster_reuse_policy(reuse_prev_centroids) -> tuple[bool, bool]:
    if reuse_prev_centroids in (True, "both"):
        return True, True
    if reuse_prev_centroids == "key":
        return False, True
    if reuse_prev_centroids in (False, "none", None):
        return False, False
    raise ValueError(f"unknown adacluster centroid reuse policy: {reuse_prev_centroids!r}")


def _adacluster_thresholded_kmeans_count(
    data: torch.Tensor,
    *,
    initial_clusters: int,
    iter_time: int,
    distance_threshold: float,
    max_iterations: int,
    num_heads: int,
) -> int:
    """Port of Adacluster Wan thresholded_kmeans_loop cluster-count selection.

    The upstream Wan path uses this only on the first layer call to pick per-layer
    Q/KV cluster counts. It returns -1 to request full attention when the
    threshold loop would exceed the seq_len // 3 cluster cap.
    """
    batch_heads, seq_len, head_dim = data.shape
    if seq_len <= 0:
        return -1
    if batch_heads % num_heads != 0:
        raise ValueError(
            f"adacluster thresholded_kmeans_loop expects batch_heads divisible by num_heads; "
            f"got batch_heads={batch_heads}, num_heads={num_heads}"
        )

    batch = batch_heads // num_heads
    current_data = data.view(batch, num_heads, seq_len, head_dim).contiguous()
    max_clusters = seq_len // 3
    if max_clusters <= 0:
        return -1

    global_cluster_indices = torch.full(
        (batch, num_heads, seq_len),
        -1,
        dtype=torch.int32,
        device=data.device,
    )
    b_idx = torch.arange(batch, device=data.device).view(-1, 1, 1).expand(batch, num_heads, seq_len)
    h_idx = torch.arange(num_heads, device=data.device).view(1, -1, 1).expand(batch, num_heads, seq_len)
    s_idx = torch.arange(seq_len, device=data.device).view(1, 1, -1).expand(batch, num_heads, seq_len)
    current_global_indices = torch.stack([b_idx, h_idx, s_idx], dim=-1)

    kernel_num = min(max(1, int(initial_clusters)), current_data.shape[2])
    idx = torch.randperm(current_data.shape[2], device=data.device)[:kernel_num]
    kernel = current_data[:, :, idx, :].clone()
    all_kernels = []
    all_counts = []
    current_cluster_offset = 0

    iteration = 0
    while kernel_num > 0 and iteration < int(max_iterations):
        iteration += 1
        current_seq_len = current_data.shape[2]
        if current_seq_len == 0:
            break

        kernel, counts, labels = _adacluster_flash_kmeans_single(
            kernel,
            current_data,
            iter_time,
        )
        if kernel.shape[2] > 0:
            invalid_labels = (labels < 0) | (labels >= kernel.shape[2])
            if torch.any(invalid_labels):
                labels = torch.clamp(labels, 0, kernel.shape[2] - 1)
        assigned_centers = torch.gather(
            kernel,
            dim=2,
            index=labels.long().unsqueeze(-1).expand(-1, -1, -1, head_dim),
        )

        distances = torch.norm(current_data - assigned_centers, dim=-1)
        position_distances = distances.mean(dim=1, keepdim=True).expand(-1, num_heads, -1)
        unassigned_mask = position_distances > distance_threshold
        distance_threshold = max(float(distance_threshold) * 1.1, float(distances.mean().item()) * 1.1)

        assigned_mask = ~unassigned_mask
        if torch.any(assigned_mask):
            assigned_global_indices = current_global_indices[assigned_mask]
            assigned_labels = labels[assigned_mask].to(torch.int32) + int(current_cluster_offset)
            global_cluster_indices[
                assigned_global_indices[:, 0],
                assigned_global_indices[:, 1],
                assigned_global_indices[:, 2],
            ] = assigned_labels
            all_kernels.append(kernel)
            all_counts.append(counts)
            current_cluster_offset += kernel.shape[2]

        if int(unassigned_mask.sum().item()) == 0:
            break

        current_data = current_data[unassigned_mask].view(batch, num_heads, -1, head_dim).contiguous()
        current_global_indices = current_global_indices[unassigned_mask].view(batch, num_heads, -1, 3)

        if current_cluster_offset + kernel_num * 2 >= max_clusters:
            return -1

        kernel_num = min(kernel_num, max_clusters - current_cluster_offset, current_data.shape[2])
        if kernel_num <= 0:
            break
        if current_data.shape[2] > kernel_num:
            idx = torch.randperm(current_data.shape[2], device=data.device)[:kernel_num]
            kernel = current_data[:, :, idx, :].clone()
        else:
            kernel = current_data.clone()
            kernel_num = current_data.shape[2]

    if all_kernels:
        return int(torch.cat(all_kernels, dim=2).shape[2])
    return -1
