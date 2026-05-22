from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


STA_TILE_SIZE = (6, 8, 8)
STA_TRITON_HEAD_DIMS = (16, 32, 64, 128, 256)
STA_SUPPORTED_SEQ_SHAPES = {
    "18x48x80": (18, 48, 80),
    "30x48x80": (30, 48, 80),
    "36x48x48": (36, 48, 48),
}
_OWNED_FASTVIDEO_STA_TRITON = None

try:
    from ...kernels.native.sta_h100 import sta_fwd  # type: ignore
except Exception:
    sta_fwd = None


def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    """SparseVideo-owned FastVideo STA wrapper.

    The public signature mirrors `fastvideo_kernel.sliding_tile_attention`,
    but runtime dispatch stays inside `src/sparsevideo`.
    """
    seq_shape = _validate_fastvideo_sta_inputs(q, k, v, window_size, has_text, seq_shape)
    if not q.is_cuda:
        raise RuntimeError("STA sparse path requires CUDA")
    if not _can_use_h100_sta(q) or seq_shape not in STA_SUPPORTED_SEQ_SHAPES:
        return _sliding_tile_attention_triton(q, k, v, window_size, text_length, has_text, seq_shape)
    return _sliding_tile_attention_h100(q, k, v, window_size, text_length, has_text, seq_shape)


def _sliding_tile_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool,
    seq_shape: str,
) -> torch.Tensor:
    """Call the SparseVideo-owned copy of FastVideo's Triton STA fallback."""
    if q.ndim != 4:
        raise ValueError(f"STA expects q/k/v in [B,H,S,D] layout, got q.shape={tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"STA expects matching q/k/v shapes, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[1] != len(window_size):
        raise ValueError(f"Number of heads must match window_size entries, got {q.shape[1]} and {len(window_size)}")

    if _triton_full_window_dense_equivalent(q, window_size, text_length, has_text, seq_shape):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    original_head_dim = int(q.shape[-1])
    kernel_head_dim = _sta_triton_head_dim(original_head_dim)
    if kernel_head_dim != original_head_dim:
        q = F.pad(q, (0, kernel_head_dim - original_head_dim))
        k = F.pad(k, (0, kernel_head_dim - original_head_dim))
        v = F.pad(v, (0, kernel_head_dim - original_head_dim))

    sta_triton = _owned_fastvideo_sta_triton()
    out = sta_triton(
        q,
        k,
        v,
        [_triple(item, "window_size") for item in window_size],
        text_length,
        has_text,
        seq_shape,
        sm_scale=original_head_dim ** -0.5,
    )
    if kernel_head_dim != original_head_dim:
        out = out[..., :original_head_dim].contiguous()
    return out


def _sliding_tile_attention_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool,
    seq_shape: str,
) -> torch.Tensor:
    seq_length = q.shape[2]
    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)

    output = torch.empty_like(q)
    seq_shape = seq_shape.lower()
    flag = {"30x48x80": 1, "36x48x48": 2, "18x48x80": 3}[seq_shape]

    for head_idx, (t, h, w) in enumerate(window_size):
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        sta_fwd(q_h, k_h, v_h, o_h, t, h, w, text_length, False, has_text, flag)
        output[:, head_idx:head_idx + 1] = o_h

    if has_text:
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output, 3, 3, 3, text_length, True, True, flag)

    return output[:, :, :seq_length]


def _canvas_shape(seq_shape: str) -> tuple[int, int, int]:
    normalized = seq_shape.lower()
    if normalized not in STA_SUPPORTED_SEQ_SHAPES:
        parts = normalized.split("x")
        if len(parts) != 3:
            raise ValueError(f"Unsupported seq_shape={seq_shape!r}; expected TxHxW")
        try:
            shape = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Unsupported seq_shape={seq_shape!r}; expected integer TxHxW") from exc
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"Unsupported seq_shape={seq_shape!r}; dimensions must be positive")
        return shape  # type: ignore[return-value]
    return STA_SUPPORTED_SEQ_SHAPES[normalized]


def _validate_fastvideo_sta_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    has_text: bool,
    seq_shape: str,
) -> str:
    if q.ndim != 4:
        raise ValueError(f"STA expects q/k/v in [B,H,S,D] layout, got q.shape={tuple(q.shape)}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"STA expects matching q/k/v shapes, got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if q.shape[1] != len(window_size):
        raise ValueError(f"Number of heads must match window_size entries, got {q.shape[1]} and {len(window_size)}")

    normalized = seq_shape.lower()
    canvas_shape = _canvas_shape(normalized)
    image_tokens = math.prod(canvas_shape)
    seq_length = q.shape[2]
    native_shape = normalized in STA_SUPPORTED_SEQ_SHAPES
    total_tile_size = math.prod(STA_TILE_SIZE)
    if any(dim % tile != 0 for dim, tile in zip(canvas_shape, STA_TILE_SIZE)):
        raise ValueError(
            f"Unsupported {normalized}, canvas dimensions must be tile-aligned to {STA_TILE_SIZE}"
        )

    if has_text and native_shape:
        if normalized != "30x48x80":
            raise ValueError("FastVideo STA text path is only defined for seq_shape='30x48x80'")
        if seq_length < image_tokens or seq_length > image_tokens + 256:
            raise ValueError(
                f"Unsupported {normalized}, current shape is {tuple(q.shape)}, "
                "only support image tokens plus up to 256 text tokens"
            )
    elif has_text:
        if seq_length < image_tokens or seq_length > image_tokens + total_tile_size:
            raise ValueError(
                f"Unsupported {normalized}, current shape is {tuple(q.shape)}, "
                f"only support image tokens plus up to one STA tile ({total_tile_size}) text tokens"
            )
    elif native_shape and normalized not in ("18x48x80", "36x48x48"):
        raise ValueError(
            f"Unsupported {normalized}, current shape is {tuple(q.shape)}, "
            "only support '36x48x48' for Stepvideo and '18x48x80' for Wan when has_text=False"
        )
    elif seq_length != image_tokens:
        raise ValueError(
            f"Unsupported {normalized}, current shape is {tuple(q.shape)}, "
            f"expected exactly {image_tokens} image tokens when has_text=False"
        )

    return normalized


def _triton_full_window_dense_equivalent(
    q: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool,
    seq_shape: str,
) -> bool:
    canvas_t, canvas_h, canvas_w = _canvas_shape(seq_shape)
    image_tokens = canvas_t * canvas_h * canvas_w
    if has_text:
        if int(text_length) != q.shape[2] - image_tokens:
            return False
    elif q.shape[2] != image_tokens or int(text_length) != 0:
        return False

    tile_t, tile_h, tile_w = STA_TILE_SIZE
    full_window = (canvas_t // tile_t, canvas_h // tile_h, canvas_w // tile_w)
    for item in window_size:
        t, h, w = _triple(item, "window_size")
        if t < full_window[0] or h < full_window[1] or w < full_window[2]:
            return False
    return True


def _sta_triton_head_dim(head_dim: int) -> int:
    head_dim = int(head_dim)
    if head_dim < STA_TRITON_HEAD_DIMS[0] or head_dim in STA_TRITON_HEAD_DIMS:
        return head_dim
    for candidate in STA_TRITON_HEAD_DIMS:
        if head_dim <= candidate:
            return candidate
    raise RuntimeError(
        f"STA Triton supports head_dim up to {STA_TRITON_HEAD_DIMS[-1]}; got {head_dim}."
    )


def _can_use_h100_sta(q: torch.Tensor) -> bool:
    if sta_fwd is None or not q.is_cuda:
        return False
    if q.device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(q.device)
    return major >= 9


def _triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must have three integers")
    return int(value[0]), int(value[1]), int(value[2])


def _owned_fastvideo_sta_triton():
    global _OWNED_FASTVIDEO_STA_TRITON
    if _OWNED_FASTVIDEO_STA_TRITON is not None:
        return _OWNED_FASTVIDEO_STA_TRITON

    module_path = (
        Path(__file__).resolve().parents[2]
        / "kernels"
        / "native"
        / "sta_h100"
        / "python"
        / "fastvideo_kernel"
        / "triton_kernels"
        / "st_attn_triton.py"
    )
    if "training_free" in module_path.parts:
        raise RuntimeError(f"Refusing to load STA runtime from training_free path: {module_path}")
    if not module_path.exists():
        raise RuntimeError(f"Missing SparseVideo-owned FastVideo STA Triton source: {module_path}")

    spec = importlib.util.spec_from_file_location("sparsevideo._owned_fastvideo_sta_triton", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SparseVideo-owned FastVideo STA Triton source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch_a100_triton_autotune(module)
    _OWNED_FASTVIDEO_STA_TRITON = module.sliding_tile_attention_triton
    return _OWNED_FASTVIDEO_STA_TRITON


def _patch_a100_triton_autotune(module) -> None:
    mode = os.environ.get("SPARSEVIDEO_STA_TRITON_AUTOTUNE", "a100").lower()
    if mode in {"full", "1", "true", "yes"}:
        return

    triton = getattr(module, "triton", None)
    kernel = getattr(module, "triton_sta_kernel", None)
    if triton is None or kernel is None or not hasattr(kernel, "configs"):
        return
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        return
    if target.backend != "cuda" or str(target.arch) not in {"80", "sm80"}:
        return

    kernel.configs = [
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_stages=3, num_warps=4)
    ]
