from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _hunyuan_mean_tokens_triton_kernel(
        x_ptr,
        mean_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        d_block = tl.program_id(2)
        num_heads = tl.num_programs(1)

        dim_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        token_start = 0
        while token_start < N:
            token_offsets = token_start + tl.arange(0, BLOCK_S)
            x_offsets = ((b * num_heads + h) * N + token_offsets[:, None]) * D + dim_offsets[None, :]
            mask = (token_offsets[:, None] < N) & dim_mask[None, :]
            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(x, axis=0)
            token_start += BLOCK_S

        mean = acc / N
        mean_offsets = (b * num_heads + h) * D + dim_offsets
        tl.store(mean_ptr + mean_offsets, mean, mask=dim_mask)


    @triton.jit
    def _hunyuan_pool_blocks_triton_kernel(
        x_ptr,
        x_mean_ptr,
        pool_ptr,
        sim_ptr,
        sim_threshold_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        USE_MEAN: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        block_idx = tl.program_id(2)
        num_heads = tl.num_programs(1)
        num_blocks = tl.num_programs(2)

        token_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        dim_offsets = tl.arange(0, D)
        valid = token_offsets[:, None] < N
        x_offsets = ((b * num_heads + h) * N + token_offsets[:, None]) * D + dim_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)
        block_count = tl.minimum(BLOCK_SIZE, N - block_idx * BLOCK_SIZE)

        if USE_MEAN:
            mean_offsets = (b * num_heads + h) * D + dim_offsets
            x_mean = tl.load(x_mean_ptr + mean_offsets)
            x = x - x_mean[None, :]
            x = tl.where(valid, x, 0.0)

        x_fp32 = x.to(tl.float32)
        pool = tl.sum(x_fp32, axis=0) / block_count

        norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
        norm = tl.maximum(norm, 1.0e-12)
        x_normed = tl.where(valid, x_fp32 / norm, 0.0)
        grams = tl.dot(x_normed, tl.trans(x_normed))
        sim_score = tl.sum(grams).to(tl.float32) / (block_count * block_count)
        threshold = tl.load(sim_threshold_ptr + h)
        sim = sim_score > threshold

        pool_offsets = ((b * num_heads + h) * num_blocks + block_idx) * D + dim_offsets
        tl.store(pool_ptr + pool_offsets, pool)
        sim_offset = (b * num_heads + h) * num_blocks + block_idx
        tl.store(sim_ptr + sim_offset, sim)


def flashomni_paper_sparse_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    sparse_block_size_for_q: int,
    sparse_block_size_for_kv: int,
    tau_q: float,
    tau_kv: float,
    S_q: float = 0.0,
    text_len: int = 0,
    kv_text_len: int | None = None,
    text_position: str = "tail",
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper/benchmark-derived FlashOmni sparse-info policy.

    This follows the public paper description and its parameter names:
    ``tau_q`` for feature-cached query blocks, ``tau_kv`` for block-sparse KV
    pairs, and ``S_q`` for the full-feature-cache degradation threshold. The
    feature-cache mask uses both paper metrics, vision-to-text contribution and
    text-to-vision guidance, and the KV mask follows the public
    ``benchmark/test_attn_score.py`` sparse-info fill logic. It is an owned
    implementation derived from public artifacts, not hidden upstream
    video-pipeline code parity.

    query/key are [B, N, H, D]. Returns unpacked sparse symbols as:
    - sparse_q: [B, H, n_q_blocks]
    - sparse_kv: [B, H, n_q_blocks, n_kv_blocks]
    with 1 meaning compute and 0 meaning skip/cache.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise RuntimeError("flashomni paper policy expects query/key with shape [B, N, H, D]")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2] or query.shape[3] != key.shape[3]:
        raise RuntimeError("flashomni paper policy requires matching batch, head, and head_dim")

    q_block_size = int(sparse_block_size_for_q)
    kv_block_size = int(sparse_block_size_for_kv)
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("flashomni sparse block sizes must be positive")
    if text_position not in ("prefix", "tail", "none"):
        raise ValueError("flashomni text_position must be 'prefix', 'tail', or 'none'")

    batch_size, q_len, num_heads, head_dim = query.shape
    kv_len = key.shape[1]
    q_blocks = _mean_pool_blocks(query, q_block_size)
    kv_blocks = _mean_pool_blocks(key, kv_block_size)

    scale = float(sm_scale) if sm_scale is not None else (float(head_dim) ** -0.5)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_blocks.float(), kv_blocks.float()) * scale
    sparse_q = torch.ones(
        (batch_size, num_heads, q_blocks.shape[1]),
        device=query.device,
        dtype=torch.uint8,
    )
    sparse_kv = torch.ones(
        (batch_size, num_heads, q_blocks.shape[1], kv_blocks.shape[1]),
        device=query.device,
        dtype=torch.uint8,
    )

    kv_text_len = int(text_len) if kv_text_len is None else int(kv_text_len)

    q_text_idx, q_vision_idx = _text_and_vision_block_indices(
        q_blocks.shape[1],
        text_len=int(text_len),
        block_size=q_block_size,
        text_position=text_position,
        device=query.device,
    )
    kv_text_idx, kv_vision_idx = _text_and_vision_block_indices(
        kv_blocks.shape[1],
        text_len=kv_text_len,
        block_size=kv_block_size,
        text_position=text_position,
        device=query.device,
    )

    if (
        float(tau_q) > 0.0
        and q_text_idx.numel()
        and q_vision_idx.numel()
        and kv_text_idx.numel()
        and kv_vision_idx.numel()
    ):
        _apply_score_cdf_feature_cache_symbols(
            sparse_q,
            scores,
            q_text_idx=q_text_idx,
            q_vision_idx=q_vision_idx,
            kv_text_idx=kv_text_idx,
            kv_vision_idx=kv_vision_idx,
            tau_q=float(tau_q),
        )

    if float(tau_kv) > 0.0 and q_vision_idx.numel():
        _apply_score_cdf_kv_symbols(
            sparse_kv,
            scores,
            q_vision_idx=q_vision_idx,
            tau_kv=float(tau_kv),
        )

    if float(S_q) > 0.0 and q_vision_idx.numel():
        _apply_full_feature_cache_threshold(
            sparse_q,
            q_vision_idx=q_vision_idx,
            S_q=float(S_q),
        )

    sparse_kv = sparse_kv * sparse_q.unsqueeze(-1)

    return sparse_q, sparse_kv


def flashomni_hunyuan_sparse_blocks(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    sparse_block_size_for_q: int,
    sparse_block_size_for_kv: int,
    threshold_q: float,
    threshold_kv: float,
    current_iter: int,
    max_sequence_length: int,
    num_inference_steps: int = 50,
    simthreshd1: float = 0.1,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Owned port of FlashOmni anonymous Hunyuan sparse-symbol policy.

    Mirrors ``example/hunyuan/models/flashomni_attn_processor/attention_processor.py``
    from the public anonymous FlashOmni artifact. Inputs are [B, N, H, D];
    outputs use the same unpacked sparse symbols as ``flashomni_paper_sparse_blocks``.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise RuntimeError("flashomni Hunyuan policy expects query/key with shape [B, N, H, D]")
    if query.shape != key.shape:
        raise RuntimeError(
            "flashomni Hunyuan policy follows upstream Hunyuan and requires query/key "
            "with matching [B, N, H, D] shapes"
        )

    q_block_size = int(sparse_block_size_for_q)
    kv_block_size = int(sparse_block_size_for_kv)
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("flashomni sparse block sizes must be positive")
    if q_block_size < kv_block_size or q_block_size % kv_block_size != 0:
        raise ValueError("flashomni Hunyuan policy requires sparse_block_size_for_q % sparse_block_size_for_kv == 0")

    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    batch_size, num_heads, seq_len, head_dim = q.shape
    nblock_q = math.ceil(seq_len / q_block_size)
    nblock_k = math.ceil(seq_len / kv_block_size)
    if nblock_q != nblock_k:
        raise ValueError(
            "flashomni Hunyuan policy follows the anonymous fill kernel and requires "
            "equal query/KV block counts; use matching sparse_block_size_for_q and "
            "sparse_block_size_for_kv for Hunyuan parity"
        )
    sparse_q = torch.ones((batch_size, num_heads, nblock_q), dtype=torch.uint8, device=query.device)
    sparse_kv = torch.ones((batch_size, num_heads, nblock_q, nblock_k), dtype=torch.uint8, device=query.device)

    text_len = int(max_sequence_length)
    if text_len < 0:
        raise ValueError("flashomni Hunyuan policy requires max_sequence_length >= 0")
    vision_len = int(seq_len) - text_len
    if vision_len <= 0 or text_len <= 0:
        return sparse_q, sparse_kv

    ratio = q_block_size // kv_block_size
    i2i = vision_len // q_block_size
    i2i_kv = vision_len // kv_block_size
    if i2i <= 0 or i2i_kv <= 0:
        return sparse_q, sparse_kv
    if i2i_kv // ratio != i2i:
        raise ValueError("flashomni Hunyuan policy requires aligned video q/kv block counts")

    km = _hunyuan_mean_tokens(k)
    sim_threshold = _hunyuan_head_threshold(simthreshd1, num_heads, query.device)
    pooled_qblocks, sim_qblocks = _hunyuan_pool_blocks(q, None, q_block_size, sim_threshold)
    pooled_kblocks, sim_kblocks = _hunyuan_pool_blocks(k, km, kv_block_size, sim_threshold)
    scale = float(sm_scale) if sm_scale is not None else (float(head_dim) ** -0.5)

    factor = (max(0.0, float(current_iter)) / max(1.0, float(num_inference_steps))) ** 1.7
    _flashomni_hunyuan_apply_score_policy(
        sparse_q,
        sparse_kv,
        pooled_qblocks,
        pooled_kblocks,
        sim_qblocks,
        sim_kblocks,
        i2i=i2i,
        i2i_kv=i2i_kv,
        ratio=ratio,
        threshold_q=float(threshold_q) * factor,
        threshold_kv=float(threshold_kv) * factor,
        scale=scale,
    )
    return sparse_q, sparse_kv


def _hunyuan_head_threshold(value: float | torch.Tensor, num_heads: int, device) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return torch.full((num_heads,), float(value.item()), device=device)
        if value.numel() != num_heads:
            raise ValueError(f"flashomni simthreshd1 tensor must have {num_heads} values")
        return value.to(device=device).flatten()
    return torch.full((num_heads,), float(value), device=device)


def _hunyuan_mean_tokens(x: torch.Tensor) -> torch.Tensor:
    if _hunyuan_can_use_mean_tokens_triton(x):
        return _hunyuan_mean_tokens_triton(x)
    return x.mean(dim=-2, keepdim=True)


def _hunyuan_can_use_mean_tokens_triton(x: torch.Tensor) -> bool:
    head_dim = int(x.shape[-1])
    return (
        _HAS_TRITON
        and x.is_cuda
        and x.device.type == "cuda"
        and x.ndim == 4
        and head_dim >= 16
        and head_dim <= 256
        and _is_power_of_two(head_dim)
    )


def _hunyuan_mean_tokens_triton(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.contiguous()
    mean = torch.empty((batch_size, num_heads, 1, head_dim), device=x.device, dtype=x.dtype)
    block_s = 1024
    block_d = min(64, triton.next_power_of_2(head_dim))
    grid = (batch_size, num_heads, triton.cdiv(head_dim, block_d))
    _hunyuan_mean_tokens_triton_kernel[grid](
        x,
        mean,
        seq_len,
        head_dim,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return mean


def _hunyuan_pool_blocks(
    x: torch.Tensor,
    x_mean: torch.Tensor | None,
    block_size: int,
    simthreshd1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _hunyuan_can_use_pool_blocks_triton(x, block_size):
        return _hunyuan_pool_blocks_triton(x, x_mean, block_size, simthreshd1)
    return _hunyuan_pool_blocks_pytorch(x, x_mean, block_size, simthreshd1)


def _hunyuan_can_use_pool_blocks_triton(x: torch.Tensor, block_size: int) -> bool:
    head_dim = int(x.shape[-1])
    block_size = int(block_size)
    return (
        _HAS_TRITON
        and x.is_cuda
        and x.device.type == "cuda"
        and x.ndim == 4
        and block_size >= 16
        and block_size <= 128
        and _is_power_of_two(block_size)
        and head_dim >= 16
        and head_dim <= 256
        and _is_power_of_two(head_dim)
    )


def _hunyuan_pool_blocks_triton(
    x: torch.Tensor,
    x_mean: torch.Tensor | None,
    block_size: int,
    simthreshd1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = x.shape
    nblock = math.ceil(seq_len / int(block_size))
    x = x.contiguous()
    pool = torch.empty((batch_size, num_heads, nblock, head_dim), device=x.device, dtype=x.dtype)
    sim = torch.empty((batch_size, num_heads, nblock), device=x.device, dtype=torch.bool)
    threshold = simthreshd1.to(device=x.device, dtype=torch.float32).contiguous()
    if x_mean is not None:
        x_mean_ptr = x_mean.to(device=x.device, dtype=x.dtype).contiguous().view(batch_size, num_heads, head_dim)
    else:
        x_mean_ptr = x
    _hunyuan_pool_blocks_triton_kernel[(batch_size, num_heads, nblock)](
        x,
        x_mean_ptr,
        pool,
        sim,
        threshold,
        seq_len,
        head_dim,
        int(block_size),
        x_mean is not None,
        num_warps=4,
    )
    return pool, sim


def _hunyuan_pool_blocks_pytorch(
    x: torch.Tensor,
    x_mean: torch.Tensor | None,
    block_size: int,
    simthreshd1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, seq_len, head_dim = x.shape
    nblock = math.ceil(seq_len / int(block_size))
    pad_len = nblock * int(block_size) - seq_len
    if pad_len:
        x = F.pad(x, (0, 0, 0, pad_len))
    blocks = x.view(batch_size, num_heads, nblock, int(block_size), head_dim)
    if x_mean is not None:
        blocks = blocks - x_mean.unsqueeze(2)
        if pad_len:
            valid = torch.ones((seq_len,), device=x.device, dtype=torch.bool)
            valid = F.pad(valid, (0, pad_len), value=False).view(1, 1, nblock, int(block_size), 1)
            blocks = blocks.masked_fill(~valid, 0)
    counts = torch.full((nblock,), int(block_size), device=x.device, dtype=torch.float32)
    if pad_len:
        counts[-1] = int(block_size) - int(pad_len)
    pool = blocks.sum(dim=3) / counts.view(1, 1, nblock, 1).to(dtype=blocks.dtype)

    normed = blocks.float()
    norm = normed.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(torch.finfo(normed.dtype).eps)
    normed = normed / norm
    grams = torch.matmul(normed, normed.transpose(-1, -2))
    sim_denominator = counts.square().view(1, 1, nblock)
    sim = (grams.sum(dim=(-1, -2)) / sim_denominator) > simthreshd1.view(1, num_heads, 1)
    return pool, sim


def _flashomni_hunyuan_apply_score_policy(
    sparse_q: torch.Tensor,
    sparse_kv: torch.Tensor,
    pooled_qblocks: torch.Tensor,
    pooled_kblocks: torch.Tensor,
    sim_qblocks: torch.Tensor,
    sim_kblocks: torch.Tensor,
    *,
    i2i: int,
    i2i_kv: int,
    ratio: int,
    threshold_q: float,
    threshold_kv: float,
    scale: float,
) -> None:
    batch_size, num_heads, nblock_q, _ = pooled_qblocks.shape
    nblock_k = int(pooled_kblocks.shape[2])
    q_rank = torch.arange(int(i2i), device=sparse_q.device).view(1, 1, int(i2i))
    kv_rank = torch.arange(nblock_k, device=sparse_q.device).view(1, 1, nblock_k)

    head0_score = _flashomni_hunyuan_pooled_score_for_heads(
        pooled_qblocks,
        pooled_kblocks,
        0,
        1,
        scale,
    )
    head0_q_score = _flashomni_hunyuan_q_score(head0_score, i2i=i2i, i2i_kv=i2i_kv, ratio=ratio)
    head0_sorted_q_indices = torch.sort(head0_q_score, dim=-1, descending=False).indices
    del head0_score, head0_q_score

    threshold_q_tensor = torch.full(
        (batch_size, 1),
        float(threshold_q),
        device=sparse_q.device,
    )
    threshold_kv_tensor = torch.full(
        (batch_size, nblock_q, 1),
        float(threshold_kv),
        device=sparse_q.device,
    )

    for head_idx in range(num_heads):
        score = _flashomni_hunyuan_pooled_score_for_heads(
            pooled_qblocks,
            pooled_kblocks,
            head_idx,
            head_idx + 1,
            scale,
        ).squeeze(1)

        q_score = _flashomni_hunyuan_q_score(score.unsqueeze(1), i2i=i2i, i2i_kv=i2i_kv, ratio=ratio).squeeze(1)
        sorted_q_values = torch.sort(q_score, dim=-1, descending=False).values
        q_cdf = torch.cumsum(sorted_q_values, dim=-1)
        num_to_select_q = torch.searchsorted(q_cdf, threshold_q_tensor, right=True)
        keep_sorted_q = (q_rank.squeeze(1) >= num_to_select_q).to(torch.uint8)
        sparse_q_head = sparse_q[:, head_idx: head_idx + 1, :int(i2i)]
        source_q_indices = head0_sorted_q_indices.expand(batch_size, 1, int(i2i))
        sparse_q_head.scatter_(dim=-1, index=source_q_indices, src=keep_sorted_q.unsqueeze(1))

        score.masked_fill_(~sim_kblocks[:, head_idx, :].unsqueeze(1), -torch.inf)
        kv_score = score.softmax(-1)
        sorted_kv_score = torch.sort(kv_score, dim=-1, descending=False)
        kv_cdf = torch.cumsum(sorted_kv_score.values, dim=-1)
        num_to_select_kv = torch.searchsorted(kv_cdf, threshold_kv_tensor, right=True).squeeze(-1)
        keep_sorted_kv = (kv_rank >= num_to_select_kv.unsqueeze(-1)).to(torch.uint8)
        sparse_kv_head = sparse_kv[:, head_idx, :, :]
        sparse_kv_head.scatter_(dim=-1, index=sorted_kv_score.indices, src=keep_sorted_kv)

        sparse_kv_head.masked_fill_(~sim_kblocks[:, head_idx, :].unsqueeze(1), 1)
        sparse_kv_head.masked_fill_(~sim_qblocks[:, head_idx, :].unsqueeze(-1), 1)
        sparse_q[:, head_idx, :].masked_fill_(~sim_qblocks[:, head_idx, :], 1)
        sparse_kv_head.mul_(sparse_q[:, head_idx, :].unsqueeze(-1))


def _flashomni_hunyuan_pooled_score_for_heads(
    pooled_qblocks: torch.Tensor,
    pooled_kblocks: torch.Tensor,
    head_start: int,
    head_end: int,
    scale: float,
) -> torch.Tensor:
    return (
        torch.matmul(
            pooled_qblocks[:, head_start:head_end].float(),
            pooled_kblocks[:, head_start:head_end].float().transpose(-1, -2),
        )
        * float(scale)
    )


def _flashomni_hunyuan_q_score(
    pooled_score: torch.Tensor,
    *,
    i2i: int,
    i2i_kv: int,
    ratio: int,
) -> torch.Tensor:
    batch_size, num_heads = pooled_score.shape[:2]
    t2i = pooled_score[:, :, int(i2i):, :int(i2i_kv)].contiguous()
    t2i = t2i.reshape(batch_size, num_heads, -1, int(i2i_kv) // int(ratio), int(ratio))
    t2i = torch.sum(t2i, dim=(2, 4))
    i2t = pooled_score[:, :, :int(i2i), int(i2i_kv):].contiguous().transpose(2, 3)
    i2t = torch.sum(i2t, dim=2)
    return (t2i + i2t).softmax(-1)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _flashomni_hunyuan_fill_q_sparse_info(
    sparse_q: torch.Tensor,
    num_to_select_q: torch.Tensor,
    sorted_q_indices: torch.Tensor,
) -> None:
    # The public anonymous Triton fill kernel indexes sorted_indices without the
    # head stride for sparse_info; reproducing that behavior keeps code parity.
    batch_size, num_heads, nblock_q = sparse_q.shape
    num_vision_q_blocks = int(sorted_q_indices.shape[-1])
    if num_vision_q_blocks > nblock_q:
        raise RuntimeError("flashomni Hunyuan sorted_q_indices exceeds sparse_q block count")
    rank = torch.arange(num_vision_q_blocks, device=sparse_q.device).view(1, 1, num_vision_q_blocks)
    source_indices = sorted_q_indices[:, :1, :].expand(batch_size, num_heads, num_vision_q_blocks)
    keep_sorted = (rank >= num_to_select_q.unsqueeze(-1)).to(torch.uint8)
    vision_sparse_q = sparse_q[:, :, :num_vision_q_blocks]
    vision_sparse_q.scatter_(dim=-1, index=source_indices, src=keep_sorted)


def _mean_pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = x.shape
    num_blocks = math.ceil(seq_len / block_size)
    pad_len = num_blocks * block_size - seq_len
    if pad_len:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
    return x.view(batch_size, num_blocks, block_size, num_heads, head_dim).mean(dim=2)


def _cdf_keep_mask(attn: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0:
        return torch.ones_like(attn, dtype=torch.bool)

    sorted_values, sorted_indices = torch.sort(attn.float(), dim=-1, descending=False)
    cdf = torch.cumsum(sorted_values, dim=-1)
    num_kv_blocks = attn.shape[-1]
    drop_counts = (cdf <= float(threshold)).sum(dim=-1).clamp(max=max(0, num_kv_blocks - 1))
    rank = torch.arange(num_kv_blocks, device=attn.device)
    sorted_keep = (rank.view(*([1] * (attn.ndim - 1)), num_kv_blocks) >= drop_counts.unsqueeze(-1))
    keep = torch.ones_like(attn, dtype=torch.bool)
    keep.scatter_(dim=-1, index=sorted_indices, src=sorted_keep)
    return keep


def _text_and_vision_block_indices(
    num_blocks: int,
    *,
    text_len: int,
    block_size: int,
    text_position: str,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_blocks = torch.arange(int(num_blocks), device=device)
    if text_position == "none" or int(text_len) <= 0:
        return all_blocks[:0], all_blocks

    text_blocks = min(int(num_blocks), math.ceil(int(text_len) / int(block_size)))
    if text_blocks <= 0:
        return all_blocks[:0], all_blocks
    if text_position == "prefix":
        return all_blocks[:text_blocks], all_blocks[text_blocks:]
    return all_blocks[int(num_blocks) - text_blocks :], all_blocks[: int(num_blocks) - text_blocks]


def _apply_score_cdf_feature_cache_symbols(
    sparse_q: torch.Tensor,
    scores: torch.Tensor,
    *,
    q_text_idx: torch.Tensor,
    q_vision_idx: torch.Tensor,
    kv_text_idx: torch.Tensor,
    kv_vision_idx: torch.Tensor,
    tau_q: float,
) -> None:
    # The public paper keeps vision blocks only when both multimodal metrics
    # are small: vision-to-text contribution and text-to-vision guidance.
    attention = scores.float().softmax(dim=-1)
    text_to_vision = attention.index_select(-2, q_text_idx).index_select(-1, kv_vision_idx)
    vision_to_text = attention.index_select(-2, q_vision_idx).index_select(-1, kv_text_idx)
    if text_to_vision.numel() == 0 or vision_to_text.numel() == 0:
        return

    contribution = text_to_vision.sum(dim=(1, 2))
    guidance = vision_to_text.transpose(-2, -1).softmax(dim=-1).sum(dim=(1, 2))
    num_vision = min(q_vision_idx.numel(), contribution.shape[-1], guidance.shape[-1])
    if num_vision <= 0:
        return
    cache_mask = (
        _low_cdf_mask(contribution[..., :num_vision], tau_q)
        & _low_cdf_mask(guidance[..., :num_vision], tau_q)
    )
    mapped = q_vision_idx[:num_vision]
    if mapped.numel() == 0:
        return
    sparse_q[:, :, mapped] = torch.where(
        cache_mask[:, None, :].expand(-1, sparse_q.shape[1], -1),
        torch.zeros((), dtype=sparse_q.dtype, device=sparse_q.device),
        sparse_q[:, :, mapped],
    )


def _apply_score_cdf_kv_symbols(
    sparse_kv: torch.Tensor,
    scores: torch.Tensor,
    *,
    q_vision_idx: torch.Tensor,
    tau_kv: float,
) -> None:
    # Mirrors training_free/FlashOmni/benchmark/test_attn_score.py:
    # img2_ = pooled_score[:, :, vision_q, :].softmax(-1)
    vision_scores = scores.index_select(-2, q_vision_idx).softmax(dim=-1)
    keep = _cdf_keep_mask(vision_scores, tau_kv).to(sparse_kv.dtype)
    sparse_kv[:, :, q_vision_idx, :] = keep


def _apply_full_feature_cache_threshold(
    sparse_q: torch.Tensor,
    *,
    q_vision_idx: torch.Tensor,
    S_q: float,
) -> None:
    if S_q <= 0.0:
        return
    if q_vision_idx.numel() <= 0:
        return

    vision_compute_ratio = sparse_q.index_select(-1, q_vision_idx).float().mean(dim=-1, keepdim=True)
    full_cache = vision_compute_ratio < float(S_q)
    sparse_q[:, :, q_vision_idx] = torch.where(
        full_cache,
        torch.zeros((), dtype=sparse_q.dtype, device=sparse_q.device),
        sparse_q[:, :, q_vision_idx],
    )


def _low_cdf_mask(values: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0:
        return torch.zeros_like(values, dtype=torch.bool)

    sorted_values, sorted_indices = torch.sort(values.float().clamp_min(0.0), dim=-1, descending=False)
    cdf = torch.cumsum(sorted_values, dim=-1)
    totals = cdf[..., -1:].clamp_min(torch.finfo(cdf.dtype).eps)
    cutoff = totals * float(threshold)
    counts = (cdf <= cutoff).sum(dim=-1).clamp(max=max(0, values.shape[-1] - 1))
    rank = torch.arange(values.shape[-1], device=values.device)
    sorted_mask = rank.view(*([1] * (values.ndim - 1)), values.shape[-1]) < counts.unsqueeze(-1)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return mask
