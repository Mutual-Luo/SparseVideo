from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _prepare_upstream_svoo_cuda_env():
    env_root = Path(sys.executable).resolve().parents[1]
    if not (env_root / "bin" / "nvcc").exists():
        return

    os.environ.setdefault("CUDA_HOME", str(env_root))
    os.environ.setdefault("CUDA_PATH", str(env_root))

    bin_dir = str(env_root / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    lib_dirs = [
        str(env_root / "lib"),
        str(env_root / "targets" / "x86_64-linux" / "lib"),
    ]
    existing_libs = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    prepend_libs = [path for path in lib_dirs if Path(path).exists() and path not in existing_libs]
    if prepend_libs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            prepend_libs + [os.environ.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)

    if Path(lib_dirs[0]).exists():
        os.environ.setdefault("FLASHINFER_EXTRA_LDFLAGS", f"-L{lib_dirs[0]}")


def _source_without_trailing_ws(path: Path) -> str:
    normalized = []
    for line in path.read_text().splitlines(keepends=True):
        if line.endswith("\n"):
            normalized.append(line[:-1].rstrip(" \t") + "\n")
        else:
            normalized.append(line.rstrip(" \t"))
    return "".join(normalized)


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


def test_hunyuan_prompt_length_excludes_video_tokens_from_attention_mask():
    from sparsevideo.processors.hunyuan_video import _prompt_length_from_attention_mask

    video_tokens = 12
    text_tokens = 5
    prompt_tokens = 3
    mask = torch.zeros(1, 1, 1, video_tokens + text_tokens, dtype=torch.bool)
    mask[..., : video_tokens + prompt_tokens] = True

    prompt_length = _prompt_length_from_attention_mask(mask, video_tokens + text_tokens, text_tokens)

    assert prompt_length == prompt_tokens


def test_hunyuan_prompt_length_accepts_text_only_attention_mask():
    from sparsevideo.processors.hunyuan_video import _prompt_length_from_attention_mask

    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    prompt_length = _prompt_length_from_attention_mask(mask, total_seq_len=17, text_len=5)

    assert prompt_length == 3


def test_hunyuan_sparse_forward_patch_exposes_upstream_timestep_argument():
    import inspect

    from sparsevideo.processors.hunyuan_sparse_forward import install_hunyuan_sparse_forward_patch
    from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformerBlock

    original_forward = HunyuanVideoTransformerBlock.forward
    restore = install_hunyuan_sparse_forward_patch()
    try:
        signature = inspect.signature(HunyuanVideoTransformerBlock.forward)
        assert HunyuanVideoTransformerBlock.forward is not original_forward
        assert "timestep" in signature.parameters
    finally:
        restore()

    assert HunyuanVideoTransformerBlock.forward is original_forward


def test_hunyuan_sparse_forward_accepts_i2v_token_replace_positional_order():
    from sparsevideo.processors.hunyuan_sparse_forward import _normalize_token_replace_forward_args

    token_replace_emb = torch.randn(1, 4)
    timestep, normalized_token_replace_emb, first_frame_num_tokens = _normalize_token_replace_forward_args(
        token_replace_emb,
        16,
        None,
    )

    assert timestep is None
    assert normalized_token_replace_emb is token_replace_emb
    assert first_frame_num_tokens == 16


def test_svoo_wan_fast_block_patch_passes_timestep_to_attn1(monkeypatch):
    from types import SimpleNamespace

    from sparsevideo.processors import wan_fast_block

    calls = {}

    def fake_attn1(**kwargs):
        calls["timestep"] = kwargs.get("timestep")
        return torch.zeros_like(kwargs["hidden_states"])

    block = SimpleNamespace(
        scale_shift_table=torch.zeros(6, 4),
        norm1=object(),
        norm2=object(),
        norm3=object(),
        attn1=fake_attn1,
        attn2=lambda **kwargs: torch.zeros_like(kwargs["hidden_states"]),
        ffn=lambda hidden_states: torch.zeros_like(hidden_states),
    )

    monkeypatch.setattr(wan_fast_block, "_layernorm_forward", lambda norm, hidden_states: hidden_states)
    monkeypatch.setattr(
        wan_fast_block,
        "_modulate_shift",
        lambda hidden_states, scale, shift, output_dtype: hidden_states.to(output_dtype),
    )
    monkeypatch.setattr(
        wan_fast_block,
        "_modulate_gate_residual",
        lambda residual, x, gate, output_dtype: residual.to(output_dtype),
    )

    hidden_states = torch.randn(1, 2, 4)
    encoder_hidden_states = torch.randn(1, 3, 4)
    temb = torch.zeros(1, 6, 4)
    timestep = torch.tensor([925])

    out = wan_fast_block._svoo_wan_block_forward(
        block,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb=None,
        timestep=timestep,
    )

    assert calls["timestep"] is timestep
    assert out.shape == hidden_states.shape


def test_svoo_co_cluster_tokens_rejects_zero_iters_before_work():
    from sparsevideo.kernels.co_cluster import co_cluster_tokens

    q = torch.empty(1, 4, 16)
    k = torch.empty(1, 4, 16)
    with pytest.raises(ValueError, match="max_iters > 0"):
        co_cluster_tokens(q, k, 2, 2, max_iters=0)


def test_svoo_co_cluster_tokens_uses_owned_triton_l2norm(monkeypatch):
    import torch.nn.functional as F

    from sparsevideo.kernels import co_cluster

    calls = {"l2norm": 0}

    def fake_l2norm(x, eps=1e-8):
        calls["l2norm"] += 1
        return F.normalize(x.float(), p=2, dim=-1, eps=eps)

    def fake_profile_norm(x, kcentroids):
        return torch.ones(x.shape[0], x.shape[1], device=x.device)

    def fake_assign(x, kcentroids, profile_centroids, norms):
        clusters = profile_centroids.shape[1]
        return (torch.arange(x.shape[1], device=x.device) % clusters).expand(x.shape[0], -1).clone()

    def fake_centroid_update(x, labels, old_centroids, *, block_n=256):
        sizes = torch.zeros(x.shape[0], old_centroids.shape[1], device=x.device, dtype=torch.long)
        sizes.scatter_add_(1, labels.long(), torch.ones_like(labels, dtype=torch.long))
        return old_centroids, sizes

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(co_cluster, "triton_l2norm_forward", fake_l2norm)
    monkeypatch.setattr(co_cluster, "profile_norm", fake_profile_norm)
    monkeypatch.setattr(co_cluster, "co_cluster_assign", fake_assign)
    monkeypatch.setattr(co_cluster, "centroid_update_sorted_euclid", fake_centroid_update)

    q = torch.randn(2, 4, 3)
    k = torch.randn(2, 4, 3)

    co_cluster.co_cluster_tokens(q, k, 2, 2, max_iters=1)

    assert calls["l2norm"] == 2


def test_svoo_dynamic_sparsity_lookup_does_not_silently_fallback(tmp_path):
    from sparsevideo.methods.svoo.ops import load_sparsity_lookup

    with pytest.raises(FileNotFoundError, match="dynamic_min_kc_ratio"):
        load_sparsity_lookup(tmp_path / "missing.csv")


def test_svoo_dynamic_sparsity_lookup_rejects_training_free_path():
    from sparsevideo.methods.svoo.ops import load_sparsity_lookup

    repo_root = Path(__file__).resolve().parents[1]
    upstream_profile = repo_root / "training_free/SVOO/sparsity_profiles/sparsity_wan_1.3B_t2v.csv"

    with pytest.raises(RuntimeError, match="inside training_free"):
        load_sparsity_lookup(upstream_profile)


def test_svoo_method_rejects_training_free_dynamic_sparsity_path():
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import SVOOMethod

    repo_root = Path(__file__).resolve().parents[1]
    upstream_profile = repo_root / "training_free/SVOO/sparsity_profiles/sparsity_wan_1.3B_t2v.csv"

    with pytest.raises(RuntimeError, match="inside training_free"):
        SVOOMethod(
            config={
                "use_dynamic_min_kc_ratio": True,
                "sparsity_csv_path": str(upstream_profile),
            },
            model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
        )


def test_svoo_method_stores_resolved_dynamic_sparsity_path(tmp_path):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import SVOOMethod

    sparsity_csv = tmp_path / "sparsity.csv"
    sparsity_csv.write_text("Step,Layer,Head,Sparsity\n", encoding="utf-8")

    method = SVOOMethod(
        config={
            "use_dynamic_min_kc_ratio": True,
            "sparsity_csv_path": str(sparsity_csv),
        },
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
    )

    assert method.config["sparsity_csv_path"] == str(sparsity_csv.resolve())


def test_svoo_exact_sparsity_profiler_matches_full_attention_shape():
    from sparsevideo.methods.svoo.sparsity import compute_exact_attention_sparsity

    query = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4)
    key = query.clone()

    avg_sparsity, per_head, stats = compute_exact_attention_sparsity(
        query, key, batch_size=2, threshold=0.95,
    )

    assert stats["attn_scores_shape"] == [1, 1, 4, 4]
    assert stats["chunk_size"] == 2
    assert len(per_head) == 1
    assert 0.0 < avg_sparsity <= 1.0


def test_svoo_measure_attention_sparsity_logs_with_upstream_format(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import method as svoo_method

    output_file = tmp_path / "attention_sparsity.txt"
    method = svoo_method.SVOOMethod(
        config={
            "measure_attention_sparsity": True,
            "sparsity_output_file": str(output_file),
            "sparsity_batch_size": 2,
            "sparsity_query_samples": 2,
        },
        model_info=SimpleNamespace(model_type="wan", transformers=[object()]),
    )
    monkeypatch.setattr(
        svoo_method,
        "_svoo_dense_attention",
        lambda query, key, value, attention_mask, *, model_type: torch.empty_like(query),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 4, 2, 4)

    processor.attn_fn(query, query, query, None)

    text = output_file.read_text(encoding="utf-8")
    assert "[Sparsity] Layer 3 | Type: self | Step: 1" in text
    assert "Query Samples: 2/4" in text
    assert "Head  0: Sparsity=" in text
    assert "Head  1: Sparsity=" in text


def test_svoo_global_constraint_dynamic_map_is_owned_and_configurable():
    from types import SimpleNamespace

    from sparsevideo.kernels.dynamic_map import compute_lambda_schedule, identify_dynamic_map_global
    from sparsevideo.methods.svoo import SVOOMethod
    from sparsevideo.methods.svoo.ops import _svoo_frame_layout

    method = SVOOMethod(
        config={
            "use_global_constraints": True,
            "lambda_schedule": "cosine",
            "diverse_top_p_k": 0.1,
        },
        model_info=SimpleNamespace(model_type="wan", transformers=[object()]),
    )
    assert method.config["use_global_constraints"] is True
    assert method.config["lambda_schedule"] == "cosine"
    assert method.config["diverse_top_p_k"] == 0.1
    assert compute_lambda_schedule(500, 2, 10, "constant") == 0.5

    torch.manual_seed(0)
    q_centroids = torch.randn(1, 2, 2, 4)
    k_centroids = torch.randn(1, 2, 3, 4)
    q_sizes = torch.tensor([[[3, 3], [2, 4]]])
    k_sizes = torch.tensor([[[2, 2, 2], [1, 3, 2]]])
    key_tokens = torch.randn(1, 2, 6, 4)
    k_labels = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 1, 1, 2, 2]])

    dynamic_map = identify_dynamic_map_global(
        q_centroids,
        k_centroids,
        q_sizes,
        k_sizes,
        0.9,
        0.34,
        key_tokens=key_tokens,
        k_labels=k_labels,
        num_frame=2,
        frame_size=3,
        timestep=500,
        layer_idx=2,
        num_layers=10,
        lambda_schedule="cosine",
        diverse_top_p_k=0.1,
    )

    assert dynamic_map.shape == (1, 2, 2, 3)
    assert dynamic_map.dtype == torch.bool
    assert dynamic_map.any(dim=-1).all()
    assert _svoo_frame_layout(21 * 45 * 80, "wan") == (21, 45 * 80)
    assert _svoo_frame_layout(33 * 45 * 80, "hunyuan_video") == (33, 45 * 80)


def test_svoo_flashinfer_env_names_follow_upstream(monkeypatch):
    from sparsevideo.kernels import flashinfer_block_sparse

    monkeypatch.setenv("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN", "0")
    monkeypatch.setenv("SV_FLASHINFER_MEM_EFFICIENT_PLAN", "1")
    assert not flashinfer_block_sparse._env_flag_first(
        ("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN", "SV_FLASHINFER_MEM_EFFICIENT_PLAN"),
        True,
    )

    monkeypatch.delenv("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN")
    assert flashinfer_block_sparse._env_flag_first(
        ("SVOO_FLASHINFER_MEM_EFFICIENT_PLAN", "SV_FLASHINFER_MEM_EFFICIENT_PLAN"),
        False,
    )


def test_svoo_dense_warmup_step_ratio_routes_to_dense(monkeypatch):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import method as svoo_method

    calls = {"dense": 0, "sparse": 0}

    def fake_dense(query, key, value, **kwargs):
        calls["dense"] += 1
        return torch.empty_like(query)

    def fake_svoo_attention(*args, **kwargs):
        calls["sparse"] += 1
        raise AssertionError("sparse path should not run during dense warmup")

    monkeypatch.setattr(
        svoo_method,
        "_svoo_dense_attention",
        lambda query, key, value, attention_mask, *, model_type: fake_dense(query, key, value),
    )
    monkeypatch.setattr(svoo_method, "svoo_attention", fake_svoo_attention)

    method = svoo_method.SVOOMethod(
        config={
            "dense_warmup_step_ratio": 0.5,
            "dense_warmup_layer_ratio": 0.0,
            "num_inference_steps": 50,
            "use_dynamic_min_kc_ratio": False,
        },
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=20, timestep=926),
    )
    query = torch.randn(1, 4, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == {"dense": 1, "sparse": 0}


def test_svoo_wan_processor_matches_upstream_qk_norm_rope_split():
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import SVOOMethod

    method = SVOOMethod(
        config={"use_dynamic_min_kc_ratio": False},
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
    )

    processor = method.create_processor(
        layer_idx=0,
        total_layers=30,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1, timestep=999),
    )

    assert processor.use_fused_qk_norm is True
    assert processor.use_fused_rope is True


def test_svoo_dense_warmup_step_ratio_allows_sparse_after_boundary(monkeypatch):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import method as svoo_method

    calls = {"dense": 0, "sparse": 0}

    def fake_dense(query, key, value, **kwargs):
        calls["dense"] += 1
        return torch.empty_like(query)

    def fake_svoo_attention(query, key, value, cfg, state, **kwargs):
        calls["sparse"] += 1
        assert kwargs["scheduler_timestep"] == 925
        return torch.empty_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        svoo_method,
        "_svoo_dense_attention",
        lambda query, key, value, attention_mask, *, model_type: fake_dense(query, key, value),
    )
    monkeypatch.setattr(svoo_method, "svoo_attention", fake_svoo_attention)

    method = svoo_method.SVOOMethod(
        config={
            "dense_warmup_step_ratio": 0.5,
            "dense_warmup_layer_ratio": 0.0,
            "num_inference_steps": 50,
            "use_dynamic_min_kc_ratio": False,
        },
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=26, timestep=925),
    )
    query = torch.randn(1, 4, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == {"dense": 0, "sparse": 1}


def test_svoo_dense_warmup_ratio_uses_first_step_count(monkeypatch):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import method as svoo_method

    calls = {"dense": 0}
    monkeypatch.setattr(
        svoo_method,
        "_svoo_dense_attention",
        lambda query, key, value, attention_mask, *, model_type: (
            calls.__setitem__("dense", calls["dense"] + 1) or torch.empty_like(query)
        ),
    )

    method = svoo_method.SVOOMethod(
        config={
            "num_inference_steps": 50,
            "dense_warmup_step_ratio": 0.2,
            "dense_warmup_layer_ratio": 0.0,
            "use_dynamic_min_kc_ratio": False,
        },
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-1.3b", transformers=[object()]),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=10, timestep=100),
    )
    query = torch.randn(1, 4, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == {"dense": 1}


def test_svoo_wan_full_attention_uses_upstream_sdpa_layout(monkeypatch):
    from sparsevideo.methods.svoo import method as svoo_method

    calls = {}

    def fake_sdpa(q, k, v, **kwargs):
        calls["shape"] = q.shape
        calls["kwargs"] = kwargs
        return q

    monkeypatch.setattr(svoo_method.F, "scaled_dot_product_attention", fake_sdpa)

    query = torch.randn(1, 4, 2, 3)
    out = svoo_method._svoo_dense_attention(query, query, query, None, model_type="wan")

    assert calls["shape"] == (1, 2, 4, 3)
    assert calls["kwargs"] == {"dropout_p": 0.0, "is_causal": False}
    assert out.shape == query.shape


def test_svoo_hunyuan_full_attention_uses_upstream_flashinfer_varlen(monkeypatch):
    from sparsevideo.methods.svoo import method as svoo_method

    calls = {}

    def fake_flashinfer(query, key, value, attention_mask):
        calls["q_shape"] = query.shape
        calls["mask_shape"] = attention_mask.shape
        calls["valid_len"] = int(attention_mask.sum().item())
        return torch.empty_like(query)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(svoo_method, "_svoo_hunyuan_flashinfer_varlen", fake_flashinfer)

    query = torch.randn(1, 6, 2, 3)
    attention_mask = torch.tensor([[[[1, 1, 1, 1, 0, 0]]]], dtype=torch.bool)
    out = svoo_method._svoo_dense_attention(
        query, query, query, attention_mask, model_type="hunyuan_video",
    )

    assert calls == {
        "q_shape": (1, 6, 2, 3),
        "mask_shape": (1, 1, 1, 6),
        "valid_len": 4,
    }
    assert out.shape == query.shape


def test_svoo_hunyuan_flashinfer_varlen_uses_upstream_two_segment_mask(monkeypatch):
    from sparsevideo.kernels import flashinfer_block_sparse

    calls = {}

    class FakeWrapper:
        def plan(self, **kwargs):
            calls["plan"] = kwargs

        def run(self, q, k, v):
            calls["run_shape"] = q.shape
            return q

    monkeypatch.setattr(flashinfer_block_sparse, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_block_sparse, "_ensure_cuda_home_for_flashinfer_jit", lambda: None)
    monkeypatch.setattr(
        flashinfer_block_sparse,
        "_make_variable_block_sparse_wrapper",
        lambda f_buffer, backend="auto": FakeWrapper(),
    )

    q = torch.randn(2, 6, 3)
    out = flashinfer_block_sparse.hunyuan_flashinfer_varlen_attn(q, q, q, valid_len=4)

    assert calls["run_shape"] == (2, 6, 3)
    assert calls["plan"]["block_mask_map"].tolist() == [
        [[True, False], [False, True]],
        [[True, False], [False, True]],
    ]
    assert calls["plan"]["block_row_sz"].tolist() == [[4, 2], [4, 2]]
    assert calls["plan"]["block_col_sz"].tolist() == [[4, 2], [4, 2]]
    assert calls["plan"]["num_qo_heads"] == 2
    assert calls["plan"]["num_kv_heads"] == 2
    assert out.shape == q.shape


def test_svoo_rejects_hunyuan_triton_sparse_backend_as_non_upstream():
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import SVOOMethod

    with pytest.raises(ValueError, match="Hunyuan upstream path uses FlashInfer"):
        SVOOMethod(
            config={"sparse_backend": "triton"},
            model_info=SimpleNamespace(model_type="hunyuan_video", transformers=[object()]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_svoo_hunyuan_sparse_path_accepts_attention_mask_and_prompt_length(monkeypatch):
    from types import SimpleNamespace

    from sparsevideo.methods.svoo import method as svoo_method

    calls = {}

    def fake_svoo_attention(query, key, value, cfg, state, **kwargs):
        calls["text_len"] = kwargs["text_len"]
        calls["prompt_length"] = kwargs["prompt_length"]
        return torch.empty_like(query)

    monkeypatch.setattr(svoo_method, "svoo_attention", fake_svoo_attention)

    method = svoo_method.SVOOMethod(
        config={"dense_warmup_step_ratio": 0.0, "dense_warmup_layer_ratio": 0.0},
        model_info=SimpleNamespace(model_type="hunyuan_video"),
    )
    processor = method.create_processor(
        layer_idx=3,
        total_layers=8,
        original_processor=None,
        step_tracker=SimpleNamespace(step=1),
    )
    query = torch.randn(1, 8, 2, 4, device="cuda")
    attention_mask = torch.tensor([1, 1, 1, 0], device="cuda", dtype=torch.bool)

    processor.attn_fn(query, query, query, attention_mask, text_len=4, prompt_length=3)

    assert calls == {"text_len": 4, "prompt_length": 3}


def test_svoo_owned_kernels_use_upstream_env_names(monkeypatch):
    from sparsevideo.kernels.co_cluster import _env_int
    from sparsevideo.kernels.flashinfer_block_sparse import _env_int_first, _env_str_first

    monkeypatch.setenv("SVOO_PROFILE_NORM_BLOCK_N", "48")
    monkeypatch.setenv("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", "123")
    monkeypatch.setenv("SVOO_FLASHINFER_SPARSE_BACKEND", "fa2")
    monkeypatch.setenv("SV_FLASHINFER_WORKSPACE_BYTES", "456")
    monkeypatch.setenv("SV_FLASHINFER_BACKEND", "fa3")

    assert _env_int("SVOO_PROFILE_NORM_BLOCK_N", 32) == 48
    assert _env_int_first(
        ("SVOO_FLASHINFER_SPARSE_WORKSPACE_BYTES", "SV_FLASHINFER_WORKSPACE_BYTES"),
        128,
    ) == 123
    assert _env_str_first(
        ("SVOO_FLASHINFER_SPARSE_BACKEND", "SV_FLASHINFER_BACKEND"),
        "auto",
    ) == "fa2"


def test_svoo_co_cluster_tuning_cache_uses_upstream_env_names(monkeypatch, tmp_path):
    from sparsevideo.kernels import co_cluster

    cache_path = tmp_path / "svoo_tuning.json"
    cache_key = "profile-norm-key"
    cached_meta = {"BLOCK_N": 16, "BLOCK_K": 16, "num_warps": 4, "num_stages": 1}
    cache_path.write_text(json.dumps({cache_key: cached_meta}))

    monkeypatch.setenv("SVOO_TRITON_TUNE", "cache")
    monkeypatch.setenv("SVOO_TRITON_TUNE_CACHE", str(cache_path))
    monkeypatch.setattr(co_cluster, "_TUNE_CACHE", None)

    def fail_if_bench_runs(meta):
        raise AssertionError("cached tuning meta should avoid benchmarking")

    meta = co_cluster._tune_or_load(
        "profile_norm",
        cache_key,
        co_cluster._TUNE_PROFILE_NORM,
        fail_if_bench_runs,
        {"BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 8, "num_stages": 2},
        max_configs_env="SVOO_PROFILE_NORM_TUNE_MAX_CONFIGS",
    )
    assert meta == cached_meta

    monkeypatch.setenv("SVOO_PROFILE_NORM_TUNE_MAX_CONFIGS", "2")
    assert len(co_cluster._config_candidates(
        co_cluster._TUNE_PROFILE_NORM,
        max_configs_env="SVOO_PROFILE_NORM_TUNE_MAX_CONFIGS",
    )) == 2

    monkeypatch.setenv("SVOO_TRITON_TUNE", "fixed")
    assert not co_cluster._tune_enabled()
    assert co_cluster._tune_cache_path() == str(cache_path)


def test_svoo_wan_fast_block_kernel_sources_match_upstream_references():
    repo_root = Path(__file__).resolve().parents[1]
    pairs = [
        (
            repo_root / "src/sparsevideo/kernels/utils.py",
            repo_root / "training_free/SVOO/svoo/kernels/triton/utils.py",
        ),
        (
            repo_root / "src/sparsevideo/kernels/layernorm.py",
            repo_root / "training_free/SVOO/svoo/kernels/triton/layernorm.py",
        ),
        (
            repo_root / "src/sparsevideo/kernels/l2norm.py",
            repo_root / "training_free/SVOO/svoo/kernels/triton/l2norm.py",
        ),
        (
            repo_root / "src/sparsevideo/kernels/modulate.py",
            repo_root / "training_free/SVOO/svoo/kernels/triton/modulate.py",
        ),
        (
            repo_root / "src/sparsevideo/kernels/sparsity.py",
            repo_root / "training_free/SVOO/svoo/kernels/triton/sparsity.py",
        ),
    ]
    for owned, upstream in pairs:
        assert owned.read_bytes() == upstream.read_bytes()

    owned_permute = repo_root / "src/sparsevideo/kernels/permute.py"
    upstream_permute = repo_root / "training_free/SVOO/svoo/kernels/triton/permute.py"
    owned_text = _source_without_trailing_ws(owned_permute)
    upstream_text = _source_without_trailing_ws(upstream_permute)
    assert owned_text != upstream_text
    for token in [
        "def _permute_kernel",
        "def _inverse_permute_kernel",
        "def permute_tensor_by_labels_triton",
        "def apply_inverse_permutation_triton",
    ]:
        assert token in owned_text
    for token in [
        "def _next_pow2",
        "def _block_d",
        "BLOCK_D",
        "d_mask = d_offsets < D",
    ]:
        assert token in owned_text


def test_svoo_fused_native_kernel_sources_match_upstream_references():
    repo_root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "csrc/ops.cu",
        "csrc/ops.h",
        "csrc/pytorch_extension_utils.h",
        "include/norm/device_utils.cuh",
        "include/norm/narrow_layer_norm.cuh",
        "include/norm/narrow_rms_norm.cuh",
        "include/rope/rope_enc.cuh",
        "include/rope/rope_enc_complex.cuh",
        "include/rope/rope_enc_txtlast.cuh",
    ]

    for relative_path in relative_paths:
        owned = repo_root / "src/sparsevideo/kernels/native/svg_svoo_fused" / relative_path
        upstream = repo_root / "training_free/SVOO/svoo/kernels" / relative_path
        assert owned.read_bytes() == upstream.read_bytes()


def test_svoo_sparsity_profiles_match_upstream_references():
    repo_root = Path(__file__).resolve().parents[1]
    profiles = [
        "sparsity_hunyuan10_13B_i2v.csv",
        "sparsity_hunyuan10_13B_t2v.csv",
        "sparsity_wan22_A14B_i2v.csv",
        "sparsity_wan22_A14B_t2v.csv",
        "sparsity_wan_1.3B_t2v.csv",
        "sparsity_wan_14B_i2v.csv",
        "sparsity_wan_14B_t2v.csv",
    ]

    for profile in profiles:
        owned = repo_root / "src/sparsevideo/methods/svoo/sparsity_profiles" / profile
        upstream = repo_root / "training_free/SVOO/sparsity_profiles" / profile
        assert owned.read_bytes() == upstream.read_bytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_owned_triton_l2norm_matches_torch_normalize_on_cuda():
    import torch.nn.functional as F

    from sparsevideo.kernels.l2norm import triton_l2norm_forward

    torch.manual_seed(0)
    x = torch.randn(2, 5, 7, device="cuda", dtype=torch.float16)

    actual = triton_l2norm_forward(x, eps=1e-8)
    expected = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)

    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_owned_triton_layernorm_matches_torch_layer_norm_on_cuda():
    import torch.nn.functional as F

    from sparsevideo.kernels.layernorm import triton_layernorm_forward

    torch.manual_seed(0)
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.float16)
    weight = torch.randn(16, device="cuda", dtype=torch.float32)
    bias = torch.randn(16, device="cuda", dtype=torch.float32)

    actual = triton_layernorm_forward(x, weight, bias, 1e-5, True)
    expected = F.layer_norm(x.float(), (16,), weight, bias, 1e-5)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_owned_triton_modulate_matches_pytorch_on_cuda():
    from sparsevideo.kernels.modulate import (
        triton_modulate_gate_residual_forward,
        triton_modulate_shift_forward,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    scale = torch.randn(2, 16, device="cuda", dtype=torch.float32)
    shift = torch.randn(2, 16, device="cuda", dtype=torch.float32)
    gate = torch.randn(2, 16, device="cuda", dtype=torch.float32)

    actual_shift = triton_modulate_shift_forward(x, scale, shift, output_dtype=torch.float16)
    expected_shift = (x.float() * (1 + scale[:, None, :]) + shift[:, None, :]).to(torch.float16)
    torch.testing.assert_close(actual_shift, expected_shift, rtol=1e-3, atol=1e-3)

    actual_gate = triton_modulate_gate_residual_forward(residual, x, gate, output_dtype=torch.float16)
    expected_gate = (residual.float() + x.float() * gate[:, None, :]).to(torch.float16)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=1e-3, atol=1e-3)


def test_svoo_sparse_path_uses_owned_triton_permutation(monkeypatch):
    from sparsevideo.methods.svoo import ops as svoo_ops
    from sparsevideo.kernels import block_sparse_attn, co_cluster

    calls = {"permute": 0, "inverse": 0}

    def fake_co_cluster_tokens(q_tokens, k_tokens, num_q_centroids, num_k_centroids, max_iters):
        batch_heads, seq_len, head_dim = q_tokens.shape
        q_labels = (torch.arange(seq_len, device=q_tokens.device) % int(num_q_centroids)).expand(batch_heads, -1).clone()
        k_labels = (torch.arange(seq_len, device=q_tokens.device) % int(num_k_centroids)).expand(batch_heads, -1).clone()
        q_centroids = torch.zeros(batch_heads, int(num_q_centroids), head_dim, device=q_tokens.device)
        k_centroids = torch.zeros(batch_heads, int(num_k_centroids), head_dim, device=q_tokens.device)
        q_sizes = torch.zeros(batch_heads, int(num_q_centroids), device=q_tokens.device, dtype=torch.long)
        k_sizes = torch.zeros(batch_heads, int(num_k_centroids), device=q_tokens.device, dtype=torch.long)
        q_sizes.scatter_add_(1, q_labels.long(), torch.ones_like(q_labels, dtype=torch.long))
        k_sizes.scatter_add_(1, k_labels.long(), torch.ones_like(k_labels, dtype=torch.long))
        return q_labels, q_centroids, q_sizes, k_labels, k_centroids, k_sizes

    def fake_identify_dynamic_map(q_centroids, k_centroids, q_sizes, k_sizes, top_p, min_ratio):
        return torch.ones(
            q_centroids.shape[0],
            q_centroids.shape[1],
            k_centroids.shape[1],
            dtype=torch.bool,
            device=q_centroids.device,
        )

    def fake_permute(tensor, labels, dim, *, sorted_indices=None):
        calls["permute"] += 1
        assert dim == 2
        batch, heads, seq_len, head_dim = tensor.shape
        if sorted_indices is None:
            sorted_indices = labels.argsort(dim=-1)
        flat = tensor.reshape(batch * heads, seq_len, head_dim)
        out = torch.gather(flat, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, head_dim))
        return out.reshape(batch, heads, seq_len, head_dim), sorted_indices

    def fake_inverse(tensor, sorted_indices, dim):
        calls["inverse"] += 1
        assert dim == 2
        batch, heads, seq_len, head_dim = tensor.shape
        flat = tensor.reshape(batch * heads, seq_len, head_dim)
        out = torch.empty_like(flat)
        out.scatter_(1, sorted_indices.unsqueeze(-1).expand(-1, -1, head_dim), flat)
        return out.reshape(batch, heads, seq_len, head_dim)

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(co_cluster, "co_cluster_tokens", fake_co_cluster_tokens)
    monkeypatch.setattr(svoo_ops, "identify_dynamic_map", fake_identify_dynamic_map)
    monkeypatch.setattr(svoo_ops, "permute_tensor_by_labels_triton", fake_permute)
    monkeypatch.setattr(svoo_ops, "apply_inverse_permutation_triton", fake_inverse)
    monkeypatch.setattr(block_sparse_attn, "block_sparse_attention", lambda q, k, v, q_sizes, k_sizes, dynamic_map, scale: q)

    cfg = {
        "implementation": "native",
        "sparse_backend": "triton",
        "num_q_centroids": 2,
        "num_k_centroids": 2,
        "top_p_kmeans": 0.9,
        "min_kc_ratio": 0.0,
        "use_dynamic_min_kc_ratio": False,
        "sparsity_csv_path": "",
        "start_reuse_step": None,
        "reuse_interval": 1,
        "kmeans_iter_init": 1,
        "kmeans_iter_step": 1,
        "use_svoo": True,
        "use_global_constraints": False,
    }
    state = {"centroids_init": False, "cached_clustering": None}
    query = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)

    out = svoo_ops.svoo_attention(
        query,
        query,
        query,
        cfg,
        state,
        current_step=1,
        layer_idx=0,
        model_type="wan",
    )

    assert out.shape == query.shape
    assert calls == {"permute": 3, "inverse": 1}


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_co_cluster_tokens_supports_non_power_of_two_head_dim_on_cuda():
    from sparsevideo.kernels.co_cluster import co_cluster_tokens
    from sparsevideo.kernels.kmeans import triton_kmeans

    torch.manual_seed(11)
    q = torch.randn(2, 32, 96, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 32, 96, device="cuda", dtype=torch.float16)

    labels, centroids, sizes = triton_kmeans(q, 4, max_iters=1)
    assert labels.shape == (2, 32)
    assert centroids.shape == (2, 4, 96)
    assert sizes.shape == (2, 4)

    q_labels, q_centroids, q_sizes, k_labels, k_centroids, k_sizes = co_cluster_tokens(
        q, k, 4, 8, max_iters=1,
    )

    assert q_labels.shape == (2, 32)
    assert k_labels.shape == (2, 32)
    assert q_centroids.shape == (2, 4, 96)
    assert k_centroids.shape == (2, 8, 96)
    assert q_sizes.shape == (2, 4)
    assert k_sizes.shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_svoo_permute_round_trips_non_power_of_two_head_dim_on_cuda():
    from sparsevideo.kernels.permute import apply_inverse_permutation_triton, permute_tensor_by_labels_triton

    torch.manual_seed(12)
    tensor = torch.randn(2, 3, 17, 96, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, 5, (6, 17), device="cuda", dtype=torch.int64)

    permuted, sorted_indices = permute_tensor_by_labels_triton(tensor, labels, dim=2)
    expected = torch.gather(
        tensor.reshape(6, 17, 96),
        1,
        sorted_indices.unsqueeze(-1).expand(-1, -1, 96),
    ).reshape_as(tensor)
    restored = apply_inverse_permutation_triton(permuted, sorted_indices, dim=2)

    torch.testing.assert_close(permuted, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_svoo_flashinfer_sparse_backend_smoke_on_cuda():
    pytest.importorskip("flashinfer.sparse")

    from sparsevideo._runtime import _cuda_toolkit_status
    from sparsevideo.methods.svoo.config import default_config
    from sparsevideo.methods.svoo.ops import svoo_attention

    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    torch.manual_seed(0)
    cfg = default_config(model_family="wan", model_key="wan21-t2v-1.3b")
    cfg.update(
        {
            "num_q_centroids": 16,
            "num_k_centroids": 32,
            "top_p_kmeans": 0.9,
            "min_kc_ratio": 0.1,
            "kmeans_iter_init": 1,
            "kmeans_iter_step": 1,
            "use_dynamic_min_kc_ratio": False,
            "start_reuse_step": None,
            "reuse_interval": 1,
            "sparse_backend": "flashinfer",
        }
    )
    state = {
        "centroids_init": False,
        "prev_q_centroids": None,
        "prev_k_centroids": None,
        "prev_q_profile_centroids": None,
        "cached_clustering": None,
        "sparsity_lookup": None,
        "sparsity_lookup_path": None,
        "last_logged_sparsity_step": None,
    }
    query = torch.randn(1, 1024, 4, 64, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = svoo_attention(
        query,
        key,
        value,
        cfg,
        state,
        current_step=1,
        layer_idx=0,
        model_type="wan",
        total_layers=1,
    )

    torch.cuda.synchronize()
    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    assert state["centroids_init"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")
def test_svoo_flashinfer_attention_matches_upstream_manual_wan_cuda():
    pytest.importorskip("flashinfer.sparse")

    from sparsevideo._runtime import _cuda_toolkit_status

    if not _cuda_toolkit_status()["available"]:
        pytest.skip("FlashInfer sparse JIT requires nvcc")

    _prepare_upstream_svoo_cuda_env()
    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = str(repo_root / "training_free" / "SVOO")
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)

    from sparsevideo.methods.svoo.config import default_config
    from sparsevideo.methods.svoo.ops import svoo_attention
    from svoo.co_clustering import (
        co_cluster_tokens as upstream_co_cluster_tokens,
        dynamic_block_sparse_fwd_flashinfer as upstream_flashinfer_attention,
        identify_dynamic_map as upstream_identify_dynamic_map,
    )
    from svoo.kernels.triton.permute import (
        apply_inverse_permutation_triton as upstream_inverse_permute,
        permute_tensor_by_labels_triton as upstream_permute,
    )

    torch.manual_seed(0)
    batch, seq_len, heads, head_dim = 1, 128, 2, 64
    query = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    nqc = 8
    nkc = 12
    max_iters = 2
    top_p = 0.9
    min_kc_ratio = 0.1

    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    q_flat = q_bhsd.reshape(batch * heads, seq_len, head_dim)
    k_flat = k_bhsd.reshape(batch * heads, seq_len, head_dim)

    torch.manual_seed(1234)
    (
        q_labels,
        q_centroids,
        q_sizes,
        k_labels,
        k_centroids,
        k_sizes,
    ) = upstream_co_cluster_tokens(q_flat, k_flat, nqc, nkc, max_iters=max_iters)
    dynamic_map = upstream_identify_dynamic_map(
        q_centroids.view(batch, heads, nqc, head_dim),
        k_centroids.view(batch, heads, nkc, head_dim),
        q_sizes.view(batch, heads, nqc),
        k_sizes.view(batch, heads, nkc),
        top_p,
        min_kc_ratio,
    )
    q_sorted_idx = q_labels.argsort(dim=-1)
    k_sorted_idx = k_labels.long().argsort(dim=-1)
    q_sorted, q_sorted_idx = upstream_permute(
        q_bhsd, None, dim=2, sorted_indices=q_sorted_idx,
    )
    k_sorted, k_sorted_idx = upstream_permute(
        k_bhsd, None, dim=2, sorted_indices=k_sorted_idx,
    )
    v_sorted, _ = upstream_permute(
        v_bhsd, None, dim=2, sorted_indices=k_sorted_idx,
    )
    expected = upstream_flashinfer_attention(
        q_sorted,
        k_sorted,
        v_sorted,
        dynamic_map,
        q_sizes.view(batch, heads, nqc).to(torch.int32),
        k_sizes.view(batch, heads, nkc).to(torch.int32),
        is_cpu=False,
    )
    expected = upstream_inverse_permute(expected, q_sorted_idx, dim=2).permute(0, 2, 1, 3)

    cfg = default_config(model_family="wan", model_key="wan21-t2v-1.3b")
    cfg.update(
        {
            "num_q_centroids": nqc,
            "num_k_centroids": nkc,
            "top_p_kmeans": top_p,
            "min_kc_ratio": min_kc_ratio,
            "kmeans_iter_init": max_iters,
            "kmeans_iter_step": max_iters,
            "use_dynamic_min_kc_ratio": False,
            "start_reuse_step": None,
            "reuse_interval": 1,
            "sparse_backend": "flashinfer",
            "enable_mem_save": False,
        }
    )
    state = {"centroids_init": False, "cached_clustering": None}

    torch.manual_seed(1234)
    actual = svoo_attention(
        query,
        key,
        value,
        cfg,
        state,
        current_step=1,
        layer_idx=0,
        model_type="wan",
        total_layers=1,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
