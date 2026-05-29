from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F


STA_TILE_SIZE = (6, 8, 8)
STA_SUPPORTED_SEQ_SHAPES = {
    "18x48x80": (18, 48, 80),
    "30x48x80": (30, 48, 80),
    "36x48x48": (36, 48, 48),
}
_STA_A100_BLOCK_SPARSE_FUNC = None
_STA_A100_BLOCK_SPARSE_IMPORT_ERROR = None
_STA_A100_RUNTIME_CACHE_MAX_SIZE = 16
_STA_A100_RUNTIME_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_STA_A100_VALID_MASK_CACHE: dict[tuple, torch.Tensor] = {}

sta_fwd = None

def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
    source_seq_shape: str | None = None,
) -> torch.Tensor:
    """SparseVideo-owned FastVideo STA wrapper.

    The public signature mirrors `fastvideo_kernel.sliding_tile_attention`,
    but runtime dispatch stays inside `src/sparsevideo`.
    """
    seq_shape = _validate_fastvideo_sta_inputs(q, k, v, window_size, has_text, seq_shape)
    if not q.is_cuda:
        raise RuntimeError("STA sparse path requires CUDA")
    if _can_use_h100_sta(q):
        if seq_shape not in STA_SUPPORTED_SEQ_SHAPES:
            raise RuntimeError(
                f"STA H100/TK native path only supports FastVideo seq_shape values "
                f"{sorted(STA_SUPPORTED_SEQ_SHAPES)}; got {seq_shape!r}."
            )
        return _sliding_tile_attention_h100(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape)
    if _can_use_a100_sta(q):
        return _sliding_tile_attention_a100(q, k, v, window_size, text_length, has_text, seq_shape, source_seq_shape)
    raise RuntimeError(
        "STA native path requires the SparseVideo-owned SM80 block-sparse CUDA backend on A100."
    )


def _sliding_tile_attention_h100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool,
    seq_shape: str,
    source_seq_shape: str | None = None,
) -> torch.Tensor:
    if source_seq_shape is not None and source_seq_shape.lower() != seq_shape.lower():
        raise RuntimeError("STA H100/TK native path does not support partial-border source_seq_shape masking")

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


def _sliding_tile_attention_a100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Sequence[Sequence[int]],
    text_length: int,
    has_text: bool,
    seq_shape: str,
    source_seq_shape: str | None = None,
) -> torch.Tensor:
    block_sparse_attn = _load_sta_a100_block_sparse_func()
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"STA A100 block-sparse CUDA path requires fp16/bf16 q/k/v, got {q.dtype}")

    batch, heads, seq_len, head_dim = q.shape
    canvas_t, canvas_h, canvas_w = _canvas_shape(seq_shape)
    image_tokens = canvas_t * canvas_h * canvas_w
    text_tail = max(0, seq_len - image_tokens)
    valid_text = max(0, min(int(text_length), text_tail)) if has_text else 0
    kv_seq_len = image_tokens + valid_text

    original_head_dim = int(head_dim)
    kernel_head_dim = 128
    if original_head_dim > kernel_head_dim:
        raise RuntimeError(
            f"STA A100 block-sparse CUDA path supports head_dim <= {kernel_head_dim}; got {original_head_dim}"
        )
    if original_head_dim < kernel_head_dim:
        q = F.pad(q, (0, kernel_head_dim - original_head_dim))
        k = F.pad(k, (0, kernel_head_dim - original_head_dim))
        v = F.pad(v, (0, kernel_head_dim - original_head_dim))

    if kv_seq_len < seq_len:
        k = torch.cat([k[:, :, :image_tokens, :], k[:, :, image_tokens:kv_seq_len, :]], dim=2)
        v = torch.cat([v[:, :, :image_tokens, :], v[:, :, image_tokens:kv_seq_len, :]], dim=2)

    q_shd = q.permute(0, 2, 1, 3).reshape(batch * seq_len, heads, kernel_head_dim).contiguous()
    k_shd = k.permute(0, 2, 1, 3).reshape(batch * kv_seq_len, heads, kernel_head_dim).contiguous()
    v_shd = v.permute(0, 2, 1, 3).reshape(batch * kv_seq_len, heads, kernel_head_dim).contiguous()

    normalized_windows = tuple(_triple(item, "window_size") for item in window_size)
    key_valid_mask = _sta_a100_key_valid_mask(
        batch,
        kv_seq_len,
        q.device,
        str(seq_shape).lower(),
        None if source_seq_shape is None else str(source_seq_shape).lower(),
        valid_text,
    )
    cu_q, cu_k, head_mask_type, base_blockmask = _sta_a100_runtime_tensors(
        batch,
        heads,
        seq_len,
        kv_seq_len,
        q.device,
        str(seq_shape).lower(),
        normalized_windows,
    )

    out = block_sparse_attn(
        q_shd,
        k_shd,
        v_shd,
        cu_q,
        cu_k,
        head_mask_type,
        None,
        base_blockmask,
        int(seq_len),
        int(kv_seq_len),
        0.0,
        softmax_scale=original_head_dim ** -0.5,
        is_causal=False,
        key_valid_mask=key_valid_mask,
    )
    out = out.reshape(batch, seq_len, heads, kernel_head_dim).permute(0, 2, 1, 3).contiguous()
    if original_head_dim < kernel_head_dim:
        out = out[..., :original_head_dim].contiguous()
    return out


def _sta_a100_key_valid_mask(
    batch: int,
    kv_seq_len: int,
    device: torch.device,
    seq_shape: str,
    source_seq_shape: str | None,
    valid_text: int,
) -> torch.Tensor | None:
    if source_seq_shape is None or source_seq_shape == seq_shape:
        return None

    image_mask_cpu = _sta_a100_image_valid_mask_cpu(seq_shape, source_seq_shape)
    if int(valid_text) > 0:
        text_mask_cpu = torch.ones((int(valid_text),), dtype=torch.uint8)
        base_cpu = torch.cat([image_mask_cpu, text_mask_cpu], dim=0)
    else:
        base_cpu = image_mask_cpu
    if int(base_cpu.numel()) != int(kv_seq_len):
        raise ValueError(
            f"STA A100 valid mask length {base_cpu.numel()} does not match kv_seq_len={kv_seq_len}"
        )
    if bool(torch.all(base_cpu)):
        return None

    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        device.type,
        device_index,
        int(batch),
        int(kv_seq_len),
        seq_shape,
        source_seq_shape,
        int(valid_text),
    )
    cached = _STA_A100_VALID_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    mask = base_cpu.unsqueeze(0).expand(int(batch), -1).contiguous().to(device=device, non_blocking=True)
    if len(_STA_A100_VALID_MASK_CACHE) >= _STA_A100_RUNTIME_CACHE_MAX_SIZE:
        _STA_A100_VALID_MASK_CACHE.pop(next(iter(_STA_A100_VALID_MASK_CACHE)))
    _STA_A100_VALID_MASK_CACHE[key] = mask
    return mask


@lru_cache(maxsize=128)
def _sta_a100_image_valid_mask_cpu(seq_shape: str, source_seq_shape: str) -> torch.Tensor:
    canvas_t, canvas_h, canvas_w = _canvas_shape(seq_shape)
    source_t, source_h, source_w = _canvas_shape(source_seq_shape)
    if source_t > canvas_t or source_h > canvas_h or source_w > canvas_w:
        raise ValueError(
            f"STA A100 source_seq_shape={source_seq_shape!r} exceeds padded seq_shape={seq_shape!r}"
        )

    tile_t, tile_h, tile_w = STA_TILE_SIZE
    mask = torch.zeros((canvas_t, canvas_h, canvas_w), dtype=torch.uint8)
    mask[:source_t, :source_h, :source_w] = 1
    return (
        mask.view(
            canvas_t // tile_t,
            tile_t,
            canvas_h // tile_h,
            tile_h,
            canvas_w // tile_w,
            tile_w,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
        .contiguous()
    )


def _sta_a100_runtime_tensors(
    batch: int,
    heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    device: torch.device,
    seq_shape: str,
    window_size: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        device.type,
        device_index,
        int(batch),
        int(heads),
        int(q_seq_len),
        int(kv_seq_len),
        seq_shape,
        window_size,
    )
    cached = _STA_A100_RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached

    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device) * int(q_seq_len)
    cu_k = torch.arange(batch + 1, dtype=torch.int32, device=device) * int(kv_seq_len)
    head_mask_type = torch.ones((heads,), dtype=torch.int32, device=device)
    mask_cpu = _sta_a100_block_mask_cpu(
        int(heads),
        int(q_seq_len),
        int(kv_seq_len),
        seq_shape,
        window_size,
    )
    base_blockmask = mask_cpu.to(device=device, non_blocking=True).expand(batch, -1, -1, -1).contiguous()
    cached = (cu_q, cu_k, head_mask_type, base_blockmask)
    if len(_STA_A100_RUNTIME_CACHE) >= _STA_A100_RUNTIME_CACHE_MAX_SIZE:
        _STA_A100_RUNTIME_CACHE.pop(next(iter(_STA_A100_RUNTIME_CACHE)))
    _STA_A100_RUNTIME_CACHE[key] = cached
    return cached


def _load_sta_a100_block_sparse_func():
    global _STA_A100_BLOCK_SPARSE_FUNC, _STA_A100_BLOCK_SPARSE_IMPORT_ERROR
    if _STA_A100_BLOCK_SPARSE_FUNC is not None:
        return _STA_A100_BLOCK_SPARSE_FUNC
    try:
        from ...kernels.draft_block_sparse_runtime import load_block_sparse_attn_func

        _STA_A100_BLOCK_SPARSE_FUNC = load_block_sparse_attn_func()
        _STA_A100_BLOCK_SPARSE_IMPORT_ERROR = None
        return _STA_A100_BLOCK_SPARSE_FUNC
    except Exception as exc:
        _STA_A100_BLOCK_SPARSE_IMPORT_ERROR = exc
        raise RuntimeError(
            "STA A100 requires the SparseVideo-owned block-sparse CUDA backend under "
            "src/sparsevideo/kernels/native/draft_block_sparse."
        ) from exc


@lru_cache(maxsize=128)
def _sta_a100_block_mask_cpu(
    heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    seq_shape: str,
    window_size: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    canvas_t, canvas_h, canvas_w = _canvas_shape(seq_shape)
    image_tokens = canvas_t * canvas_h * canvas_w
    if q_seq_len < image_tokens or kv_seq_len < image_tokens:
        raise ValueError("STA A100 block mask sequence length is shorter than the image canvas")
    if heads != len(window_size):
        raise ValueError("STA A100 block mask head count must match window_size")

    tile_t, tile_h, tile_w = STA_TILE_SIZE
    tile_t_count = canvas_t // tile_t
    tile_h_count = canvas_h // tile_h
    tile_w_count = canvas_w // tile_w
    tile_count = tile_t_count * tile_h_count * tile_w_count
    image_blocks = image_tokens // 128
    q_blocks = math.ceil(q_seq_len / 128)
    kv_blocks = math.ceil(kv_seq_len / 128)
    mask = torch.zeros((heads, q_blocks, kv_blocks), dtype=torch.bool)

    for head_idx, item in enumerate(window_size):
        kernel_t, kernel_h, kernel_w = item
        for tile_id in range(tile_count):
            query_block_start = tile_id * 3
            query_block_stop = query_block_start + 3
            query_tile_t = tile_id // (tile_h_count * tile_w_count)
            rem = tile_id % (tile_h_count * tile_w_count)
            query_tile_h = rem // tile_w_count
            query_tile_w = rem % tile_w_count

            center_t = _clamp_int(query_tile_t, kernel_t // 2, (tile_t_count - 1) - kernel_t // 2)
            center_h = _clamp_int(query_tile_h, kernel_h // 2, (tile_h_count - 1) - kernel_h // 2)
            center_w = _clamp_int(query_tile_w, kernel_w // 2, (tile_w_count - 1) - kernel_w // 2)
            start_t = max(0, center_t - kernel_t // 2)
            stop_t = min(tile_t_count, center_t + kernel_t // 2 + 1)
            start_h = max(0, center_h - kernel_h // 2)
            stop_h = min(tile_h_count, center_h + kernel_h // 2 + 1)
            start_w = max(0, center_w - kernel_w // 2)
            stop_w = min(tile_w_count, center_w + kernel_w // 2 + 1)

            kv_block_indices = []
            for kv_t in range(start_t, stop_t):
                for kv_h in range(start_h, stop_h):
                    for kv_w in range(start_w, stop_w):
                        kv_tile_id = (kv_t * tile_h_count + kv_h) * tile_w_count + kv_w
                        base = kv_tile_id * 3
                        kv_block_indices.extend((base, base + 1, base + 2))
            if kv_blocks > image_blocks:
                kv_block_indices.extend(range(image_blocks, kv_blocks))
            mask[head_idx, query_block_start:query_block_stop, kv_block_indices] = True

        if q_blocks > image_blocks:
            mask[head_idx, image_blocks:q_blocks, :kv_blocks] = True

    return mask.unsqueeze(0).contiguous()


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


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
    elif seq_length != image_tokens:
        raise ValueError(
            f"Unsupported {normalized}, current shape is {tuple(q.shape)}, "
            f"expected exactly {image_tokens} image tokens when has_text=False"
        )

    return normalized


def _can_use_h100_sta(q: torch.Tensor) -> bool:
    if sta_fwd is None or not q.is_cuda:
        return False
    if q.device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(q.device)
    return major >= 9


def _can_use_a100_sta(q: torch.Tensor) -> bool:
    if not q.is_cuda:
        return False
    if q.device.type != "cuda":
        return False
    major, _minor = torch.cuda.get_device_capability(q.device)
    if major != 8:
        return False
    try:
        _load_sta_a100_block_sparse_func()
    except RuntimeError:
        return False
    return True


def _triple(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must have three integers")
    return int(value[0]), int(value[1]), int(value[2])
