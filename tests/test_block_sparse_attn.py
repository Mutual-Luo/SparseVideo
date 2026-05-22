from __future__ import annotations

import pytest
import torch


def _reference_block_sparse_attention(q, k, v, q_sizes, k_sizes, dynamic_map, scale):
    out = torch.zeros_like(q)
    q_cum = torch.cumsum(
        torch.cat([torch.zeros(q_sizes.shape[0], 1, dtype=q_sizes.dtype, device=q_sizes.device), q_sizes], dim=1),
        dim=1,
    )
    k_cum = torch.cumsum(
        torch.cat([torch.zeros(k_sizes.shape[0], 1, dtype=k_sizes.dtype, device=k_sizes.device), k_sizes], dim=1),
        dim=1,
    )

    for b in range(q.shape[0]):
        for qi in range(q_sizes.shape[1]):
            q_start = int(q_cum[b, qi].item())
            q_end = int(q_cum[b, qi + 1].item())
            if q_start == q_end:
                continue

            k_parts = []
            v_parts = []
            for ki in range(k_sizes.shape[1]):
                if not bool(dynamic_map[b, qi, ki].item()):
                    continue
                k_start = int(k_cum[b, ki].item())
                k_end = int(k_cum[b, ki + 1].item())
                if k_start == k_end:
                    continue
                k_parts.append(k[b, k_start:k_end])
                v_parts.append(v[b, k_start:k_end])

            if not k_parts:
                continue

            k_active = torch.cat(k_parts, dim=0)
            v_active = torch.cat(v_parts, dim=0)
            scores = torch.matmul(q[b, q_start:q_end].float(), k_active.float().T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b, q_start:q_end] = torch.matmul(probs, v_active.float()).to(out.dtype)

    return out


def test_variable_block_sparse_plan_chunks_respect_kv_budget():
    from sparsevideo.kernels.flashinfer_block_sparse import _variable_block_sparse_plan_chunks

    dynamic_map = torch.tensor(
        [
            [[True, True], [False, True]],
            [[True, False], [True, False]],
            [[True, True], [True, True]],
        ],
        dtype=torch.bool,
    )
    k_sizes = torch.tensor([[5, 7], [11, 13], [17, 19]], dtype=torch.int32)

    assert _variable_block_sparse_plan_chunks(dynamic_map, k_sizes, max_kv_indices=30) == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert _variable_block_sparse_plan_chunks(dynamic_map, k_sizes, max_kv_indices=10_000) == [
        (0, 3),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_block_sparse_attention_matches_pytorch_reference_cuda():
    from sparsevideo.kernels.block_sparse_attn import block_sparse_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(2, 10, 16, device=device, dtype=torch.float16)
    k = torch.randn(2, 10, 16, device=device, dtype=torch.float16)
    v = torch.randn(2, 10, 16, device=device, dtype=torch.float16)
    q_sizes = torch.tensor([[4, 3, 3], [2, 5, 3]], device=device, dtype=torch.long)
    k_sizes = torch.tensor([[5, 3, 2], [4, 4, 2]], device=device, dtype=torch.long)
    dynamic_map = torch.tensor(
        [
            [[True, False, True], [False, True, False], [True, True, True]],
            [[True, False, False], [False, True, True], [True, False, True]],
        ],
        device=device,
        dtype=torch.bool,
    )
    scale = 16 ** -0.5

    actual = block_sparse_attention(q, k, v, q_sizes, k_sizes, dynamic_map, scale)
    expected = _reference_block_sparse_attention(q, k, v, q_sizes, k_sizes, dynamic_map, scale)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_block_sparse_attention_matches_reference_for_non_power_of_two_head_dim_cuda():
    from sparsevideo.kernels.block_sparse_attn import block_sparse_attention

    torch.manual_seed(2)
    device = torch.device("cuda")
    q = torch.randn(2, 12, 96, device=device, dtype=torch.float16)
    k = torch.randn(2, 12, 96, device=device, dtype=torch.float16)
    v = torch.randn(2, 12, 96, device=device, dtype=torch.float16)
    q_sizes = torch.tensor([[5, 4, 3], [3, 5, 4]], device=device, dtype=torch.long)
    k_sizes = torch.tensor([[4, 5, 3], [6, 2, 4]], device=device, dtype=torch.long)
    dynamic_map = torch.tensor(
        [
            [[True, False, True], [True, True, False], [False, True, True]],
            [[True, False, False], [False, True, True], [True, True, False]],
        ],
        device=device,
        dtype=torch.bool,
    )
    scale = 96 ** -0.5

    actual = block_sparse_attention(q, k, v, q_sizes, k_sizes, dynamic_map, scale)
    expected = _reference_block_sparse_attention(q, k, v, q_sizes, k_sizes, dynamic_map, scale)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_variable_block_sparse_flashinfer_matches_pytorch_reference_cuda():
    from sparsevideo.kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    if not HAS_FLASHINFER:
        pytest.skip("flashinfer.sparse is not available")

    torch.manual_seed(1)
    device = torch.device("cuda")
    q = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
    k = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
    v = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
    q_sizes = torch.tensor([[10, 11, 11], [8, 12, 12]], device=device, dtype=torch.int32)
    k_sizes = torch.tensor([[9, 13, 10], [7, 14, 11]], device=device, dtype=torch.int32)
    dynamic_map = torch.tensor(
        [
            [[True, False, True], [False, True, False], [True, True, True]],
            [[True, False, False], [False, True, True], [True, False, True]],
        ],
        device=device,
        dtype=torch.bool,
    )
    scale = 64 ** -0.5

    actual = variable_block_sparse_attn(q, k, v, dynamic_map, q_sizes, k_sizes)
    expected = _reference_block_sparse_attention(q, k, v, q_sizes.long(), k_sizes.long(), dynamic_map, scale)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_variable_block_sparse_flashinfer_pads_head_dim_96_cuda(dtype):
    from sparsevideo.kernels.flashinfer_block_sparse import HAS_FLASHINFER, variable_block_sparse_attn

    if not HAS_FLASHINFER:
        pytest.skip("flashinfer.sparse is not available")

    torch.manual_seed(3)
    device = torch.device("cuda")
    q = torch.randn(2, 32, 96, device=device, dtype=dtype)
    k = torch.randn(2, 32, 96, device=device, dtype=dtype)
    v = torch.randn(2, 32, 96, device=device, dtype=dtype)
    q_sizes = torch.tensor([[10, 11, 11], [8, 12, 12]], device=device, dtype=torch.int32)
    k_sizes = torch.tensor([[9, 13, 10], [7, 14, 11]], device=device, dtype=torch.int32)
    dynamic_map = torch.tensor(
        [
            [[True, False, True], [False, True, False], [True, True, True]],
            [[True, False, False], [False, True, True], [True, False, True]],
        ],
        device=device,
        dtype=torch.bool,
    )
    scale = 96 ** -0.5

    actual = variable_block_sparse_attn(q, k, v, dynamic_map, q_sizes, k_sizes)
    expected = _reference_block_sparse_attention(q, k, v, q_sizes.long(), k_sizes.long(), dynamic_map, scale)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
