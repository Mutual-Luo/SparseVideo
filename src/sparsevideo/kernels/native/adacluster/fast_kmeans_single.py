import torch
import triton
import triton.language as tl

from typing import Tuple

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["HEAD_DIM"])
@triton.jit
def _compute_norm_squal_impl(
    Kernel_k,
    Normal_k,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    KERNEL_CTX: tl.constexpr,
    stride_kernel_k_z: tl.constexpr,
    stride_kernel_k_h: tl.constexpr,
    stride_kernel_k_q: tl.constexpr,
    stride_kernel_k_d: tl.constexpr,
    stride_norm_z: tl.constexpr,
    stride_norm_h: tl.constexpr,
    stride_norm_q: tl.constexpr,
    stride_norm_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    # BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // HEADS  # which batch
    off_h = off_hz % HEADS  # which heads

    kernel_offset = (
        off_z.to(tl.int64) * stride_kernel_k_z + off_h.to(tl.int64) * stride_kernel_k_h
    )
    norm_offset = (
        off_z.to(tl.int64) * stride_norm_z + off_h.to(tl.int64) * stride_norm_h
    )

    Kernel_k_block_ptr = tl.make_block_ptr(
        base=Kernel_k + kernel_offset,
        shape=(KERNEL_CTX, HEAD_DIM),
        strides=(stride_kernel_k_q, stride_kernel_k_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Normal_k_block_ptr = tl.make_block_ptr(
        base=Normal_k + norm_offset,
        shape=(KERNEL_CTX, 1),
        strides=(stride_norm_q, stride_norm_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    value = tl.load(Kernel_k_block_ptr, boundary_check=(1, 0), padding_option="zero")
    value = value * value
    value = tl.sum(value, axis=1)
    value = tl.reshape(value, (BLOCK_M, 1))

    tl.store(Normal_k_block_ptr, value.to(Normal_k.type.element_ty), boundary_check=(1, 0))


# @triton.autotune(list(filter(keep, configs)), key=["HEAD_DIM"])
@triton.jit
def _compute_cluster_indices_impl(
    Kernel_k,
    Normal_k,
    K,
    Out_cluster_indices,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    KERNEL_CTX: tl.constexpr,
    stride_kernel_k_z: tl.constexpr,
    stride_kernel_k_h: tl.constexpr,
    stride_kernel_k_q: tl.constexpr,
    stride_kernel_k_d: tl.constexpr,
    stride_norm_z: tl.constexpr,
    stride_norm_h: tl.constexpr,
    stride_norm_q: tl.constexpr,
    stride_norm_d: tl.constexpr,
    stride_k_z: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_k_q: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_out_z: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_q: tl.constexpr,
    stride_out_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // HEADS  # which batch
    off_h = off_hz % HEADS  # which heads

    # offsets to the same head and same batch
    k_offset = off_z.to(tl.int64) * stride_k_z + off_h.to(tl.int64) * stride_k_h
    out_offset = off_z.to(tl.int64) * stride_out_z + off_h.to(tl.int64) * stride_out_h

    kernel_offset = (
        off_z.to(tl.int64) * stride_kernel_k_z + off_h.to(tl.int64) * stride_kernel_k_h
    )
    norm_offset = (
        off_z.to(tl.int64) * stride_norm_z + off_h.to(tl.int64) * stride_norm_h
    )

    # input block ptr (no need advance for them)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_k_q, stride_k_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # output block ptr (no need advance for them)
    Out_cluster_indices_block_ptr = tl.make_block_ptr(
        base=Out_cluster_indices + out_offset,
        shape=(N_CTX, 1),
        strides=(stride_out_q, stride_out_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    # kernel block ptr (in same head and batch of input block ptr)
    ## transpose for kernel_k
    Kernel_k_block_ptr = tl.make_block_ptr(
        base=Kernel_k + kernel_offset,
        shape=(HEAD_DIM, KERNEL_CTX),
        strides=(stride_kernel_k_d, stride_kernel_k_q),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    Normal_k_block_ptr = tl.make_block_ptr(
        base=Normal_k + norm_offset,
        shape=(KERNEL_CTX, 1),
        strides=(stride_norm_q, stride_norm_d),
        offsets=(0, 0),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )

    # tmp memory
    tmp_min_distance = tl.zeros([BLOCK_M], tl.float32) + float("inf")
    tmp_min_index = tl.zeros([BLOCK_M], tl.int32)

    k = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")

    for start_n in range(0, KERNEL_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # load kernel_k and it's norm ** 2
        kernel_k_t = tl.load(
            Kernel_k_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )
        norm_k = tl.load(
            Normal_k_block_ptr, boundary_check=(1, 0), padding_option="zero"
        )  # shape [BLOCK_N, 1]

        # compute current distance
        distance = tl.dot(
            k, kernel_k_t
        )  # shape [BLOCK_M, BLOCK_N], but it may contain unvalid value for distance
        norm_k = norm_k.reshape(BLOCK_N)
        distance = norm_k - 2 * distance

        # try mask unvalid value in distance for the valid of shape of kernel_k_t may not be BLOCK_N
        invalid_kernel_mask = tl.arange(0, BLOCK_N) < (KERNEL_CTX - start_n)
        invalid_kernel_mask = invalid_kernel_mask.reshape(BLOCK_N)
        distance = tl.where(invalid_kernel_mask, distance, float("inf"))

        # find min
        block_min_dist, block_min_index = tl.min(distance, axis=1, return_indices=True)
        block_min_index = block_min_index + start_n  # real min index

        # try mask unvalid value in block_min_dist
        # for valid shape of k may not be BLOCK_M
        update_mask = tl.arange(0, BLOCK_M) < (N_CTX - start_m * BLOCK_M)
        update_mask = update_mask & (block_min_dist < tmp_min_distance)

        tmp_min_distance = tl.where(update_mask, block_min_dist, tmp_min_distance)
        tmp_min_index = tl.where(update_mask, block_min_index, tmp_min_index)

        # advance kernel_k and norm_k
        Kernel_k_block_ptr = tl.advance(Kernel_k_block_ptr, (0, BLOCK_N))
        Normal_k_block_ptr = tl.advance(Normal_k_block_ptr, (BLOCK_N, 0))

    # write results
    tmp_min_index = tmp_min_index.reshape(BLOCK_M, 1)
    tl.store(Out_cluster_indices_block_ptr, tmp_min_index, boundary_check=(1, 0))

# @triton.autotune(list(filter(keep, configs)), key=["HEAD_DIM"])
@triton.jit
def _compute_new_kernel_impl(
    Indices,
    Kernel_k,
    K,
    Kernel_count,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    KERNEL_CTX: tl.constexpr,
    stride_indices_z: tl.constexpr,
    stride_indices_h: tl.constexpr,
    stride_indices_q: tl.constexpr,
    stride_indices_d: tl.constexpr,
    stride_kernel_kv_z: tl.constexpr,
    stride_kernel_kv_h: tl.constexpr,
    stride_kernel_kv_q: tl.constexpr,
    stride_kernel_kv_d: tl.constexpr,
    stride_kv_z: tl.constexpr,
    stride_kv_h: tl.constexpr,
    stride_kv_q: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_kernel_count_z: tl.constexpr,
    stride_kernel_count_h: tl.constexpr,
    stride_kernel_count_q: tl.constexpr,
    stride_kernel_count_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    # BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // HEADS
    off_h = off_hz % HEADS

    kernel_kv_offset = (
        off_z.to(tl.int64) * stride_kernel_kv_z
        + off_h.to(tl.int64) * stride_kernel_kv_h
    )
    kv_offset = off_z.to(tl.int64) * stride_kv_z + off_h.to(tl.int64) * stride_kv_h
    kernel_count_offset = (
        off_z.to(tl.int64) * stride_kernel_count_z
        + off_h.to(tl.int64) * stride_kernel_count_h
    )
    indices_count_offset = (
        off_z.to(tl.int64) * stride_indices_z + off_h.to(tl.int64) * stride_indices_h
    )

    # make block pointers for k,v,count
    Indices_block_ptr = tl.make_block_ptr(
        base=Indices + indices_count_offset,
        shape=(N_CTX, 1),
        strides=(stride_indices_q, stride_indices_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kv_q, stride_kv_d),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    indices = tl.load(
        Indices_block_ptr, boundary_check=(1, 0), padding_option="zero"
    ).reshape(BLOCK_M, 1)

    update_mask = tl.arange(0, BLOCK_M) < (N_CTX - start_m * BLOCK_M)
    update_mask = update_mask.reshape(BLOCK_M, 1)

    k = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero").reshape(
        BLOCK_M, HEAD_DIM
    )

    kernel_count_offsets = indices * stride_kernel_count_q
    tl.atomic_add(
        Kernel_count + kernel_count_offset + kernel_count_offsets,
        1,
        mask=update_mask,
    )

    kv_address_offset = indices * stride_kernel_kv_q + tl.arange(0, HEAD_DIM)
    tl.atomic_add(Kernel_k + kernel_kv_offset + kv_address_offset, k, mask=update_mask)

def compute_cluster_indices(kernel_k, k):
    # shape constraints
    BATCH, H, N_CTX, HEAD_DIM = k.shape
    KERNEL_CTX = kernel_k.shape[2]
    assert HEAD_DIM in {16, 32, 64, 128, 256}

    norm_kernel_k = torch.empty(
        (BATCH, H, KERNEL_CTX, 1), device=kernel_k.device, dtype=kernel_k.dtype
    )
    extra_kern_args = {}
    BLOCK_M = 64
    BLOCK_N = 64
    grid_for_norm = lambda args: (
        # triton.cdiv(kernel_k.shape[2], args["BLOCK_M"]),
        triton.cdiv(kernel_k.shape[2], BLOCK_M),
        kernel_k.shape[0] * kernel_k.shape[1],
        1,
    )
    _compute_norm_squal_impl[grid_for_norm](
        kernel_k,
        norm_kernel_k,
        BATCH,
        H,
        KERNEL_CTX,
        kernel_k.stride(0),
        kernel_k.stride(1),
        kernel_k.stride(2),
        kernel_k.stride(3),
        norm_kernel_k.stride(0),
        norm_kernel_k.stride(1),
        norm_kernel_k.stride(2),
        norm_kernel_k.stride(3),
        HEAD_DIM,
        BLOCK_M,
        **extra_kern_args,
    )

    cluster_indices = torch.empty(
        (BATCH, H, N_CTX, 1), dtype=torch.int32, device=kernel_k.device
    )
    extra_kern_args = {}
    grid_for_compute_indices = lambda args: (
        # triton.cdiv(k.shape[2], args["BLOCK_M"]),
        triton.cdiv(k.shape[2], BLOCK_M),
        k.shape[0] * k.shape[1],
        1,
    )
    _compute_cluster_indices_impl[grid_for_compute_indices](
        kernel_k,
        norm_kernel_k,
        k,
        cluster_indices,
        BATCH,
        H,
        N_CTX,
        KERNEL_CTX,
        kernel_k.stride(0),
        kernel_k.stride(1),
        kernel_k.stride(2),
        kernel_k.stride(3),
        norm_kernel_k.stride(0),
        norm_kernel_k.stride(1),
        norm_kernel_k.stride(2),
        norm_kernel_k.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        cluster_indices.stride(0),
        cluster_indices.stride(1),
        cluster_indices.stride(2),
        cluster_indices.stride(3),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        **extra_kern_args,
    )

    # cluster_indices = cluster_indices.squeeze(-1)
    return cluster_indices


def compute_new_kernel(min_indices, kernel_k, k):
    # kernel_k *= kernel_count
    # kernel_v *= kernel_count

    # shape constraints
    BATCH, H, N_CTX, HEAD_DIM = k.shape
    KERNEL_CTX = kernel_k.shape[2]
    assert HEAD_DIM in {16, 32, 64, 128, 256}

    BLOCK_M = 64

    grid_for_reset = lambda args: (
        triton.cdiv(kernel_k.shape[2], BLOCK_M),
        kernel_k.shape[0] * kernel_k.shape[1],
        1,
    )
    kernel_count = torch.zeros(BATCH, H, KERNEL_CTX, 1, dtype=torch.int32, device=kernel_k.device)
    kernel_k = torch.zeros(BATCH, H, KERNEL_CTX, HEAD_DIM, dtype=kernel_k.dtype, device=kernel_k.device)

    extra_kern_args = {}
    grid = lambda args: (
        # triton.cdiv(k.shape[2], args["BLOCK_M"]),
        triton.cdiv(k.shape[2], BLOCK_M),
        k.shape[0] * k.shape[1],
        1,
    )
    kernel_k = kernel_k.to(torch.float32)
    _compute_new_kernel_impl[grid](
        min_indices,
        kernel_k,
        k,
        kernel_count,
        BATCH,
        H,
        N_CTX,
        KERNEL_CTX,
        min_indices.stride(0),
        min_indices.stride(1),
        min_indices.stride(2),
        min_indices.stride(3),
        kernel_k.stride(0),
        kernel_k.stride(1),
        kernel_k.stride(2),
        kernel_k.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        kernel_count.stride(0),
        kernel_count.stride(1),
        kernel_count.stride(2),
        kernel_count.stride(3),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        **extra_kern_args,
    )

    tmp_kernel_count = torch.where(kernel_count > 0, kernel_count, 1)

    kernel_k /= tmp_kernel_count

    kernel_k = kernel_k.to(k.dtype)

    return kernel_k, kernel_count


@torch.jit.ignore
def flash_kmeans_single(
    key_kernel: torch.Tensor,
    key: torch.Tensor,
    iter_time: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_deivce = key_kernel.device

    with torch.cuda.device(current_deivce):
        for _ in range(iter_time):
            cluster_indices = compute_cluster_indices(key_kernel, key)
            key_kernel, count_kernel = compute_new_kernel(
                cluster_indices,
                key_kernel,
                key,
            )

    return key_kernel, count_kernel, cluster_indices.squeeze(-1)
