"""Parity / correctness tests for the SVG-EAR (Error-Aware Reduction) method.

EAR reuses the SVG2/SAP k-means + permutation scaffolding but swaps:
  * block selection -> identify_dynamic_map_estimated (error-aware)
  * block-sparse attention -> dynamic_block_sparse_prune_fwd_flashinfer
    (exact attention on selected blocks, centroid approximation on pruned blocks).

The kernels need CUDA + the package's FlashInfer build, so these tests are
GPU-gated.
"""

import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _dense(q, k, v):
    qb, kb, vb = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    return F.scaled_dot_product_attention(qb, kb, vb).permute(0, 2, 1, 3)


def _ref_prune(q, k, v, k_centroids, v_centroids, dyn, q_sizes, k_sizes, prune_mask, scale):
    """Pure-torch reference of dynamic_block_sparse_prune_fwd_flashinfer.

    Each query token softmaxes over the union of:
      * key tokens in clusters selected by ``dyn`` (exact attention), and
      * centroid pseudo-tokens of clusters where ``prune_mask`` == 0, with a
        log(cluster_size) prior, valued by the value-centroid.
    All tensors are [B, H, ...]; computed in fp32.
    """
    B, H, S, D = q.shape
    QC = q_sizes.shape[-1]
    KC = k_sizes.shape[-1]
    out = torch.empty((B, H, S, D), device=q.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            qcl = torch.repeat_interleave(torch.arange(QC, device=q.device), q_sizes[b, h].long())
            kcl = torch.repeat_interleave(torch.arange(KC, device=q.device), k_sizes[b, h].long())
            ql, kl, vl = q[b, h].float(), k[b, h].float(), v[b, h].float()
            exact = (ql @ kl.t()) * scale
            sel = dyn[b, h][qcl][:, kcl]
            exact = exact.masked_fill(~sel, float("-inf"))
            cent = (ql @ k_centroids[b, h].float().t()) * scale
            cent = cent + torch.log(k_sizes[b, h].float() + 1e-6)[None, :]
            keepc = (prune_mask[b, h] == 0)[qcl]
            cent = cent.masked_fill(~keepc, float("-inf"))
            logits = torch.cat([exact, cent], dim=1)
            probs = torch.softmax(logits, dim=1)
            vals = torch.cat([vl, v_centroids[b, h].float()], dim=0)
            out[b, h] = probs @ vals
    return out


def _make_clustered(BH, QC, KC, S, D, device, dtype, seed=0):
    """Build cluster-sorted q/k/v plus cluster sizes/centroids for one folded batch."""
    g = torch.Generator(device=device).manual_seed(seed)

    def _sizes(n_clusters):
        base = torch.ones(n_clusters, dtype=torch.long, device=device)
        extra = S - n_clusters
        idx = torch.randint(0, n_clusters, (extra,), generator=g, device=device)
        base += torch.bincount(idx, minlength=n_clusters)
        return base

    q_sizes = torch.stack([_sizes(QC) for _ in range(BH)])  # [BH, QC]
    k_sizes = torch.stack([_sizes(KC) for _ in range(BH)])  # [BH, KC]
    q = torch.randn(BH, S, D, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(BH, S, D, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(BH, S, D, generator=g, device=device, dtype=dtype) * 0.5

    def _centroids(x, sizes):
        outs = []
        for b in range(BH):
            off = 0
            cs = []
            for sz in sizes[b].tolist():
                cs.append(x[b, off:off + sz].float().mean(0))
                off += sz
            outs.append(torch.stack(cs))
        return torch.stack(outs).to(dtype)

    k_centroids = _centroids(k, k_sizes)
    v_centroids = _centroids(v, k_sizes)  # v grouped by key clusters
    return q, k, v, q_sizes, k_sizes, k_centroids, v_centroids


@cuda
def test_svgear_registered():
    from sparsevideo._registry import get_method_class, list_methods

    assert "svgear" in list_methods()
    cls = get_method_class("svgear")
    cfg = cls.default_config(model_key="wan21-t2v-1.3b", num_inference_steps=50)
    assert cfg["gamma"] == 1.0
    assert "min_kc_ratio" in cfg


@cuda
def test_prune_kernel_matches_reference():
    from sparsevideo.kernels.flashinfer_block_sparse import dynamic_block_sparse_prune_fwd_flashinfer

    device, dtype = "cuda", torch.float16
    BH, QC, KC, S, D = 2, 6, 10, 192, 64
    q, k, v, q_sizes, k_sizes, kc, vc = _make_clustered(BH, QC, KC, S, D, device, dtype, seed=1)
    scale = D ** -0.5

    g = torch.Generator(device=device).manual_seed(7)
    dyn = torch.rand(BH, QC, KC, generator=g, device=device) < 0.4
    # Guarantee every query cluster attends to at least one key cluster (avoid empty flash rows).
    dyn[:, :, 0] = True

    # 4D [B,H,...] with H=1 (folded convention used by the method layer).
    out = dynamic_block_sparse_prune_fwd_flashinfer(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        kc.unsqueeze(1), vc.unsqueeze(1),
        dyn.unsqueeze(1), q_sizes.to(torch.int32).unsqueeze(1), k_sizes.to(torch.int32).unsqueeze(1),
    ).squeeze(1)

    ref = _ref_prune(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        kc.unsqueeze(1), vc.unsqueeze(1),
        dyn.unsqueeze(1), q_sizes.unsqueeze(1), k_sizes.unsqueeze(1),
        prune_mask=dyn.unsqueeze(1), scale=scale,
    ).squeeze(1)

    err = (out.float() - ref).abs()
    assert err.max().item() < 2e-2, f"max abs err {err.max().item()}"


@cuda
def test_prune_kernel_distinct_prune_mask_excludes_columns():
    """prune_mask forces certain pruned columns out of the centroid step entirely."""
    from sparsevideo.kernels.flashinfer_block_sparse import dynamic_block_sparse_prune_fwd_flashinfer

    device, dtype = "cuda", torch.float16
    BH, QC, KC, S, D = 1, 4, 8, 128, 64
    q, k, v, q_sizes, k_sizes, kc, vc = _make_clustered(BH, QC, KC, S, D, device, dtype, seed=3)
    scale = D ** -0.5
    dyn = torch.zeros(BH, QC, KC, dtype=torch.bool, device=device)
    dyn[:, :, 0] = True  # only cluster 0 exact-selected
    # Exclude the last 2 key clusters from centroid approximation.
    prune_mask = dyn.clone()
    prune_mask[..., -2:] = True

    out = dynamic_block_sparse_prune_fwd_flashinfer(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        kc.unsqueeze(1), vc.unsqueeze(1),
        dyn.unsqueeze(1), q_sizes.to(torch.int32).unsqueeze(1), k_sizes.to(torch.int32).unsqueeze(1),
        prune_mask=prune_mask.unsqueeze(1),
    ).squeeze(1)
    ref = _ref_prune(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        kc.unsqueeze(1), vc.unsqueeze(1),
        dyn.unsqueeze(1), q_sizes.unsqueeze(1), k_sizes.unsqueeze(1),
        prune_mask=prune_mask.unsqueeze(1), scale=scale,
    ).squeeze(1)
    err = (out.float() - ref).abs()
    assert err.max().item() < 2e-2, f"max abs err {err.max().item()}"


@cuda
def test_identify_estimated_shape_and_floor():
    from sparsevideo.kernels.dynamic_map import identify_dynamic_map_estimated

    device, dtype = "cuda", torch.float16
    BH, QC, KC, S, D = 2, 6, 12, 192, 64
    q, k, v, q_sizes, k_sizes, kc, vc = _make_clustered(BH, QC, KC, S, D, device, dtype, seed=5)
    min_kc_ratio = 0.25
    q_centroids = _q_centroids(q, q_sizes)
    dyn = identify_dynamic_map_estimated(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
        q_sizes.unsqueeze(1), k_sizes.unsqueeze(1),
        q_centroids.unsqueeze(1),
        kc.unsqueeze(1), vc.unsqueeze(1),
        top_p=0.9, gamma=1.0, min_kc_ratio=min_kc_ratio,
    ).squeeze(1)
    assert dyn.shape == (BH, QC, KC)
    assert dyn.dtype == torch.bool
    # Every query cluster keeps at least floor(min_kc_ratio * KC) clusters.
    floor = max(1, int(min_kc_ratio * KC))
    assert (dyn.sum(dim=-1) >= floor).all()


def _q_centroids(q, q_sizes):
    BH = q.shape[0]
    outs = []
    for b in range(BH):
        off = 0
        cs = []
        for sz in q_sizes[b].tolist():
            cs.append(q[b, off:off + sz].float().mean(0))
            off += sz
        outs.append(torch.stack(cs))
    return torch.stack(outs).to(q.dtype)


@cuda
def test_svgear_full_selection_equals_dense():
    from sparsevideo.methods.svgear.method import _svgear_attention

    device, dtype = "cuda", torch.float16
    B, H, D, S = 1, 4, 64, 256
    g = torch.Generator(device=device).manual_seed(11)
    q = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.5
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    out = _svgear_attention(
        q, k, v, top_p_kmeans=1.0, min_kc_ratio=1.0, gamma=1.0,
        num_q_centroids=8, num_k_centroids=16, kmeans_iter_init=3, kmeans_iter_step=2,
        state=state, model_type="wan", text_len=0,
    )
    ref = _dense(q, k, v)
    err = (out.float() - ref.float()).abs()
    assert err.max().item() < 5e-3, f"max abs err {err.max().item()}"


@cuda
def test_svgear_end_to_end_close_to_dense():
    from sparsevideo.methods.svgear.method import _svgear_attention

    device, dtype = "cuda", torch.float16
    B, H, D, S = 1, 8, 64, 512
    g = torch.Generator(device=device).manual_seed(13)
    q = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    k = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    v = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    out = _svgear_attention(
        q, k, v, top_p_kmeans=0.9, min_kc_ratio=0.1, gamma=1.0,
        num_q_centroids=16, num_k_centroids=32, kmeans_iter_init=5, kmeans_iter_step=2,
        state=state, model_type="wan", text_len=0,
    )
    ref = _dense(q, k, v)
    assert torch.isfinite(out).all()
    assert out.shape == ref.shape
    # Sparse approximation should track dense reasonably on random data.
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.3, f"relative error {rel.item()}"


@cuda
def test_svgear_text_tail_runs_and_finite():
    """Hunyuan-style text-tail path: video tokens + text tail. Must run and stay finite."""
    from sparsevideo.methods.svgear.method import _svgear_attention

    device, dtype = "cuda", torch.float16
    B, H, D = 1, 4, 64
    video_len, text_len = 256, 32
    S = video_len + text_len
    g = torch.Generator(device=device).manual_seed(17)
    q = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    k = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    v = torch.randn(B, S, H, D, generator=g, device=device, dtype=dtype) * 0.4
    state = {"centroids_init": False, "prev_q_centroids": None, "prev_k_centroids": None}
    out = _svgear_attention(
        q, k, v, top_p_kmeans=0.9, min_kc_ratio=0.1, gamma=1.0,
        num_q_centroids=16, num_k_centroids=32, kmeans_iter_init=5, kmeans_iter_step=2,
        state=state, model_type="hunyuan_video", text_len=text_len, prompt_length=text_len,
        context_length=text_len,
    )
    assert out.shape == (B, S, H, D)
    assert torch.isfinite(out).all()
