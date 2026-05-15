from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _reference_identify_dynamic_map(query_centroids, key_centroids, k_cluster_sizes, p, min_kc_ratio=0):
    batch_heads, q_clusters, head_dim = query_centroids.shape
    k_clusters = key_centroids.shape[1]
    scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (head_dim**0.5)
    weights = k_cluster_sizes.unsqueeze(-2).float()
    max_score = torch.max(scores.float(), dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores.float() - max_score)
    probs = weights * exp_scores / torch.sum(weights * exp_scores, dim=-1, keepdim=True).clamp(min=1e-12)
    sorted_probs, sorted_indices = torch.sort(probs.to(scores.dtype), dim=-1, descending=True)
    remove_indices = torch.cumsum(sorted_probs, dim=-1) > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if isinstance(min_kc_ratio, torch.Tensor):
        ratios = min_kc_ratio.flatten().to(dtype=torch.float32)
        for bh in range(batch_heads):
            ratio = float(ratios[bh if ratios.numel() > 1 else 0].item())
            if ratio > 0:
                remove_indices[bh, :, : int(ratio * k_clusters)] = False
    elif float(min_kc_ratio) > 0:
        remove_indices[..., : int(float(min_kc_ratio) * k_clusters)] = False

    keep = ~remove_indices
    dynamic_map = torch.zeros(batch_heads, q_clusters, k_clusters, dtype=torch.bool)
    dynamic_map.scatter_(-1, sorted_indices, keep)
    return dynamic_map


def test_svoo_dynamic_map_matches_upstream_top_p_shift_semantics():
    from sparsevideo.methods.svoo.ops import identify_dynamic_map

    query_centroids = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 0.5]],
        ]
    )
    key_centroids = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8], [-1.0, 0.0]],
            [[0.5, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    q_sizes = torch.tensor([[2, 3], [4, 1]])
    k_sizes = torch.tensor([[3, 2, 1], [1, 4, 2]])
    min_ratio = torch.tensor([0.34, 0.0])

    actual = identify_dynamic_map(query_centroids, key_centroids, q_sizes, k_sizes, 0.7, min_ratio)
    expected = _reference_identify_dynamic_map(query_centroids, key_centroids, k_sizes, 0.7, min_ratio)

    assert actual.equal(expected)


def test_svoo_co_cluster_tokens_rejects_zero_iters_before_work():
    from sparsevideo.kernels.co_cluster import co_cluster_tokens

    q = torch.empty(1, 4, 16)
    k = torch.empty(1, 4, 16)
    with pytest.raises(ValueError, match="max_iters > 0"):
        co_cluster_tokens(q, k, 2, 2, max_iters=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_co_cluster_tokens_smoke_shapes_on_cuda():
    from sparsevideo.kernels.co_cluster import co_cluster_tokens

    torch.manual_seed(0)
    q = torch.randn(2, 32, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 32, 16, device="cuda", dtype=torch.float16)

    q_labels, q_centroids, q_sizes, k_labels, k_centroids, k_sizes = co_cluster_tokens(
        q, k, 4, 8, max_iters=1,
    )

    assert q_labels.shape == (2, 32)
    assert k_labels.shape == (2, 32)
    assert q_centroids.shape == (2, 4, 16)
    assert k_centroids.shape == (2, 8, 16)
    assert q_sizes.shape == (2, 4)
    assert k_sizes.shape == (2, 8)
    assert q_sizes.sum(dim=1).tolist() == [32, 32]
    assert k_sizes.sum(dim=1).tolist() == [32, 32]
