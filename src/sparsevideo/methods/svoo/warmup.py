from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterable

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _warmup_mode() -> str:
    return os.environ.get("SVOO_TRITON_WARMUP_MODE", "compile").strip().lower()


def _effective_seq_len(seq_len: int) -> int:
    if _warmup_mode() in ("full", "profile", "benchmark"):
        return int(seq_len)
    return min(int(seq_len), max(1, _env_int("SVOO_TRITON_WARMUP_SEQ_LEN", 4096)))


def _as_cuda_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda":
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@contextmanager
def _preserve_rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _valid_centroids(num_q_centroids, num_k_centroids) -> bool:
    return (
        num_q_centroids is not None
        and num_k_centroids is not None
        and int(num_q_centroids) > 0
        and int(num_k_centroids) > 0
    )


def _first_attr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def warmup_dimensions(pipe, *, model_type: str, height: int, width: int, num_frames: int, config: dict) -> dict:
    transformer = pipe.transformer
    tconfig = transformer.config

    num_heads = int(_first_attr(tconfig, ("num_attention_heads", "num_heads"), 0))
    head_dim = int(_first_attr(tconfig, ("attention_head_dim", "head_dim"), 0))
    if num_heads <= 0 or head_dim <= 0:
        raise RuntimeError("SVOO warmup could not infer attention head shape from transformer config")

    block_hidden_dim = num_heads * head_dim
    blocks = getattr(transformer, "blocks", None)
    if blocks:
        norm1 = getattr(blocks[0], "norm1", None)
        weight = getattr(norm1, "weight", None)
        if weight is not None:
            block_hidden_dim = int(weight.shape[-1])

    if model_type == "wan":
        patch_t, patch_h, patch_w = transformer.config.patch_size
        temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        spatial_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        num_frame_patches = 1 + int(num_frames) // (temporal_scale * int(patch_t))
        frame_size = (int(height) // (spatial_scale * int(patch_h))) * (
            int(width) // (spatial_scale * int(patch_w))
        )
        return {
            "model_name": "Wan",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": num_frame_patches * frame_size,
            "inverse_seq_len": None,
            "include_wan_block_kernels": True,
            "include_rmsnorm": True,
        }

    if model_type == "hunyuan_video":
        context_length = int(config.get("context_length") or 256)
        num_frame_patches = 1 + int(num_frames) // 4
        frame_size = int(height) * int(width) // 256
        seq_len = num_frame_patches * frame_size
        return {
            "model_name": "HunyuanVideo 1.0",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": context_length + seq_len,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    if model_type == "cogvideox":
        patch_size = int(transformer.config.patch_size)
        patch_size_t = getattr(transformer.config, "patch_size_t", None)
        temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        spatial_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        latent_frames = (int(num_frames) - 1) // temporal_scale + 1
        if patch_size_t is not None:
            patch_size_t = int(patch_size_t)
            latent_frames = (latent_frames + patch_size_t - 1) // patch_size_t
        frame_size = (int(height) // (spatial_scale * patch_size)) * (
            int(width) // (spatial_scale * patch_size)
        )
        seq_len = latent_frames * frame_size
        context_length = int(
            config.get("context_length")
            or getattr(transformer.config, "max_text_seq_length", 226)
        )
        return {
            "model_name": "CogVideoX",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": context_length + seq_len,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    if model_type == "ltx_video":
        patch_size = int(getattr(transformer.config, "patch_size", 1))
        patch_size_t = int(getattr(transformer.config, "patch_size_t", 1))
        temporal_scale = int(
            _first_attr(pipe, ("vae_temporal_compression_ratio", "vae_scale_factor_temporal"), 8)
        )
        spatial_scale = int(
            _first_attr(pipe, ("vae_spatial_compression_ratio", "vae_scale_factor_spatial"), 32)
        )
        latent_frames = (int(num_frames) - 1) // temporal_scale + 1
        latent_frames = latent_frames // patch_size_t
        frame_size = (int(height) // (spatial_scale * patch_size)) * (
            int(width) // (spatial_scale * patch_size)
        )
        seq_len = latent_frames * frame_size
        return {
            "model_name": "LTX Video",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": None,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    if model_type == "allegro":
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        patch_size_t = int(getattr(transformer.config, "patch_size_t", 1))
        vae_spatial_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        vae_temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        latent_frames = (int(num_frames) - 1) // vae_temporal_scale + 1
        latent_frames = latent_frames // patch_size_t
        frame_size = (int(height) // (vae_spatial_scale * patch_size)) * (
            int(width) // (vae_spatial_scale * patch_size)
        )
        seq_len = latent_frames * frame_size
        return {
            "model_name": "Allegro",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": None,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    if model_type == "mochi":
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        vae_spatial_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        vae_temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 6))
        latent_frames = (int(num_frames) - 1) // vae_temporal_scale + 1
        frame_size = (int(height) // (vae_spatial_scale * patch_size)) * (
            int(width) // (vae_spatial_scale * patch_size)
        )
        seq_len = latent_frames * frame_size
        context_length = int(config.get("context_length") or getattr(transformer.config, "max_sequence_length", 256))
        return {
            "model_name": "Mochi",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": context_length + seq_len,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    if model_type == "easyanimate":
        patch_size = int(getattr(transformer.config, "patch_size", 2))
        vae_spatial_scale = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        vae_temporal_scale = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        latent_frames = (int(num_frames) - 1) // vae_temporal_scale + 1
        frame_size = (int(height) // (vae_spatial_scale * patch_size)) * (
            int(width) // (vae_spatial_scale * patch_size)
        )
        seq_len = latent_frames * frame_size
        context_length = int(config.get("context_length") or getattr(transformer.config, "max_text_seq_length", 256))
        return {
            "model_name": "EasyAnimate",
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_hidden_dim": block_hidden_dim,
            "norm_hidden_dim": head_dim,
            "seq_len": seq_len,
            "inverse_seq_len": context_length + seq_len,
            "include_wan_block_kernels": False,
            "include_rmsnorm": False,
        }

    raise RuntimeError(f"SVOO warmup does not support model_type={model_type!r}")


def _warmup_wan_block_kernels(hidden_dim: int, dtype: torch.dtype, device: torch.device) -> None:
    from ...kernels.layernorm import triton_layernorm_forward
    from ...kernels.modulate import triton_modulate_gate_residual_forward, triton_modulate_shift_forward

    x = torch.zeros(1, 1, hidden_dim, dtype=dtype, device=device).contiguous()
    w = torch.ones(hidden_dim, dtype=dtype, device=device)
    b = torch.zeros(hidden_dim, dtype=dtype, device=device)
    y = triton_layernorm_forward(x, w, b, 1e-6, elementwise_affine=True)

    scale_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    shift_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    gate_1d = torch.zeros(1, 1, hidden_dim, dtype=torch.float32, device=device)
    y = triton_modulate_shift_forward(y, scale_1d, shift_1d, output_dtype=dtype)
    _ = triton_modulate_gate_residual_forward(x, y, gate_1d, output_dtype=dtype)

    scale_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    shift_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    gate_b = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    y = triton_modulate_shift_forward(x, scale_b, shift_b, output_dtype=dtype)
    _ = triton_modulate_gate_residual_forward(x, y, gate_b, output_dtype=dtype)
    _sync(device)


def _warmup_rmsnorm(hidden_dim: int, seq_len: int, dtype: torch.dtype, device: torch.device) -> None:
    from ...kernels.fused_norm_rope import triton_rmsnorm_inplace

    x = torch.zeros(1, seq_len, 1, hidden_dim, dtype=dtype, device=device).contiguous()
    w = torch.ones(hidden_dim, dtype=dtype, device=device)
    _ = triton_rmsnorm_inplace(x, w, 1e-6)
    _sync(device)


def _warmup_permute(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    inverse_seq_len: int | None = None,
) -> None:
    from ...kernels.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton

    x = torch.zeros(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    labels = torch.zeros(num_heads, seq_len, dtype=torch.int64, device=device)
    x_perm, sorted_idx = permute_tensor_by_labels_triton(x, labels, dim=2)
    _ = apply_inverse_permutation_triton(x_perm, sorted_idx, dim=2)

    if inverse_seq_len is not None and inverse_seq_len != seq_len:
        x_full = torch.zeros(1, num_heads, inverse_seq_len, head_dim, dtype=dtype, device=device)
        sorted_full = torch.arange(inverse_seq_len, dtype=torch.int32, device=device)
        sorted_full = sorted_full.expand(num_heads, -1).contiguous()
        _ = apply_inverse_permutation_triton(x_full, sorted_full, dim=2)
    _sync(device)


def _warmup_cocluster(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    num_q_centroids: int,
    num_k_centroids: int,
    dtype: torch.dtype,
    device: torch.device,
    cfg_values: Iterable[int],
) -> None:
    from ...kernels.co_cluster import co_cluster_tokens

    for cfg in cfg_values:
        batch_heads = int(cfg) * num_heads
        q = torch.zeros(batch_heads, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.zeros(batch_heads, seq_len, head_dim, dtype=dtype, device=device)
        _ = co_cluster_tokens(q, k, int(num_q_centroids), int(num_k_centroids), max_iters=1)
        _sync(device)


def _split_sizes(total: int, blocks: int, batch_heads: int, device: torch.device) -> torch.Tensor:
    base = total // blocks
    remainder = total % blocks
    sizes = torch.full((batch_heads, blocks), base, dtype=torch.long, device=device)
    if remainder:
        sizes[:, :remainder] += 1
    return sizes


def _warmup_flashinfer_sparse(num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device) -> None:
    if not _env_flag("SVOO_FLASHINFER_WARMUP", default=True):
        return

    from ...kernels.flashinfer_block_sparse import variable_block_sparse_attn

    seq_len = max(16, _env_int("SVOO_FLASHINFER_WARMUP_SEQ_LEN", 128))
    num_blocks = max(1, min(seq_len, _env_int("SVOO_FLASHINFER_WARMUP_BLOCKS", 4)))
    q = torch.zeros(num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    block_mask = torch.ones(num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
    block_sizes = _split_sizes(seq_len, num_blocks, num_heads, device)
    _ = variable_block_sparse_attn(q, k, v, block_mask, block_sizes, block_sizes)
    _sync(device)


def warmup_svoo_kernels_from_pipeline(
    pipe,
    *,
    model_type: str,
    height: int,
    width: int,
    num_frames: int,
    config: dict,
    dtype: torch.dtype,
    device,
) -> dict:
    status = {
        "enabled": _env_flag("SVOO_TRITON_WARMUP", default=True),
        "ran": False,
        "mode": _warmup_mode(),
        "error": None,
    }
    if not status["enabled"]:
        status["reason"] = "disabled"
        return status

    device = _as_cuda_device(device)
    if device.type != "cuda":
        status["reason"] = "non_cuda"
        return status

    strict = _env_flag("SVOO_TRITON_WARMUP_STRICT", default=False)
    dims = warmup_dimensions(
        pipe,
        model_type=model_type,
        height=height,
        width=width,
        num_frames=num_frames,
        config=config,
    )
    warmup_seq_len = _effective_seq_len(dims["seq_len"])
    warmup_inverse_seq_len = dims["inverse_seq_len"]
    if warmup_inverse_seq_len is not None and _warmup_mode() not in ("full", "profile", "benchmark"):
        warmup_inverse_seq_len = min(
            int(warmup_inverse_seq_len),
            warmup_seq_len + max(0, int(dims["inverse_seq_len"]) - int(dims["seq_len"])),
        )

    status.update(
        {
            "model_name": dims["model_name"],
            "seq_len": dims["seq_len"],
            "warmup_seq_len": warmup_seq_len,
            "num_heads": dims["num_heads"],
            "head_dim": dims["head_dim"],
            "block_hidden_dim": dims["block_hidden_dim"],
            "sparse_backend": config.get("sparse_backend", "flashinfer"),
        }
    )

    start = time.perf_counter()
    try:
        with _preserve_rng_state(device):
            if dims["include_wan_block_kernels"]:
                _warmup_wan_block_kernels(dims["block_hidden_dim"], dtype, device)
            if dims["include_rmsnorm"]:
                _warmup_rmsnorm(dims["norm_hidden_dim"], warmup_seq_len, dtype, device)

            _warmup_permute(
                dims["num_heads"],
                dims["head_dim"],
                warmup_seq_len,
                dtype,
                device,
                inverse_seq_len=warmup_inverse_seq_len,
            )
            if _valid_centroids(config.get("num_q_centroids"), config.get("num_k_centroids")):
                _warmup_cocluster(
                    dims["num_heads"],
                    dims["head_dim"],
                    warmup_seq_len,
                    int(config["num_q_centroids"]),
                    int(config["num_k_centroids"]),
                    dtype,
                    device,
                    cfg_values=(1,),
                )
            if config.get("sparse_backend", "flashinfer") == "flashinfer":
                _warmup_flashinfer_sparse(dims["num_heads"], dims["head_dim"], dtype, device)
        _cleanup(device)
        status["ran"] = True
        status["elapsed_sec"] = time.perf_counter() - start
        return status
    except Exception as exc:
        _cleanup(device)
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["elapsed_sec"] = time.perf_counter() - start
        if strict:
            raise RuntimeError(f"SVOO kernel warmup failed: {status['error']}") from exc
        return status
