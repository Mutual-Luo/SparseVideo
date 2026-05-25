from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo.methods.flashomni.method import (
    FlashOmniMethod,
    _FlashOmniPaperMMDiTState,
    _candidate_flashomni_roots,
    _clear_flashomni_modules,
    _flashomni_apply_cached_q_blocks,
    _flashomni_cache_q_blocks,
    _flashomni_custom_mask_from_attention_mask,
    _flashomni_global_random_sparse_blocks,
    _flashomni_pack_sparse_q_info,
    _flashomni_paper_mmdit_attention,
    _flashomni_paper_mmdit_effective_step,
    _flashomni_paper_mmdit_schedule,
    _flashomni_normalize_sparse_bits,
    _flashomni_sparse_o_gemm,
    _flashomni_sparse_o_gemm_cache_bias,
    _flashomni_sparse_q_gemm,
    _flashomni_trim_prefix_key_value_mask,
    _flashomni_upstream_attention,
    _flashomni_import,
    _flashomni_native_head_dim,
    _has_flashomni_extension,
    _is_training_free_runtime,
)
from sparsevideo.methods.flashomni.hunyuan_forward import (
    _flashomni_hunyuan_cache_init,
    cal_type,
    cal_type_sparse,
    derivative_approximation,
    flashomni_hunyuan_forward,
    install_flashomni_hunyuan_forward_patch,
    _maybe_start_taylor_cache,
    _flashomni_trace_memory,
    taylor_formula,
)
from sparsevideo.methods.flashomni.policy import (
    _HAS_TRITON,
    _hunyuan_mean_tokens,
    _hunyuan_pool_blocks,
    _hunyuan_pool_blocks_pytorch,
    flashomni_hunyuan_sparse_blocks,
    flashomni_paper_sparse_blocks,
)


def _load_upstream_flashomni_benchmark_utils():
    path = Path(__file__).resolve().parents[1] / "training_free" / "FlashOmni" / "benchmark" / "utils.py"
    spec = importlib.util.spec_from_file_location("_upstream_flashomni_benchmark_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _expected_low_cdf_mask(values: torch.Tensor, threshold: float) -> torch.Tensor:
    sorted_values, sorted_indices = torch.sort(values.float().clamp_min(0.0), dim=-1, descending=False)
    cdf = torch.cumsum(sorted_values, dim=-1)
    cutoff = cdf[..., -1:] * float(threshold)
    counts = (cdf <= cutoff).sum(dim=-1).clamp(max=max(0, values.shape[-1] - 1))
    rank = torch.arange(values.shape[-1], device=values.device)
    sorted_mask = rank.view(*([1] * (values.ndim - 1)), values.shape[-1]) < counts.unsqueeze(-1)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return mask


def test_flashomni_hunyuan_mean_tokens_matches_torch_when_available():
    if not (_HAS_TRITON and torch.cuda.is_available()):
        pytest.skip("requires CUDA and Triton")
    x = torch.randn(1, 2, 129, 32, device="cuda", dtype=torch.bfloat16)

    actual = _hunyuan_mean_tokens(x)
    expected = x.mean(dim=-2, keepdim=True)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_flashomni_default_requires_upstream_explicit_sparse_info():
    with pytest.raises(NotImplementedError, match="sparse_pattern='explicit'"):
        FlashOmniMethod(config={}, model_info=SimpleNamespace(model_type="wan"))


def test_flashomni_explicit_path_requires_sparse_info_tensors():
    with pytest.raises(NotImplementedError, match="sparse_pattern='explicit'"):
        FlashOmniMethod(config={"sparse_pattern": "explicit"}, model_info=SimpleNamespace(model_type="wan"))


def test_flashomni_explicit_path_accepts_upstream_sparse_info_tensor_names():
    sparse_info = torch.ones(1, dtype=torch.uint8)
    sparse_kv_info = torch.ones(1, dtype=torch.uint8)
    sparse_info_indptr = torch.tensor([0, 1], dtype=torch.int32)
    sparse_kv_info_indptr = torch.tensor([0, 1], dtype=torch.int32)

    method = FlashOmniMethod(
        config={
            "sparse_pattern": "explicit",
            "sparse_info": sparse_info,
            "sparse_kv_info": sparse_kv_info,
            "sparse_info_indptr": sparse_info_indptr,
            "sparse_kv_info_indptr": sparse_kv_info_indptr,
        },
        model_info=SimpleNamespace(model_type="wan"),
    )

    assert method.config["sparse_pattern"] == "explicit"
    assert method.config["sparse_info"] is sparse_info


def test_flashomni_local_qk_topk_path_is_explicit_opt_in():
    with pytest.warns(RuntimeWarning, match="local_qk_topk"):
        method = FlashOmniMethod(
            config={"sparse_pattern": "local_qk_topk"},
            model_info=SimpleNamespace(model_type="wan"),
        )

    assert method.config["sparse_pattern"] == "local_qk_topk"


def test_flashomni_global_random_path_uses_upstream_benchmark_names():
    method = FlashOmniMethod(
        config={"sparse_pattern": "global_random", "spq_Q": 0.25, "spq_KV": 0.5, "sparse_size": 64},
        model_info=SimpleNamespace(model_type="wan"),
    )

    assert method.config["sparse_pattern"] == "global_random"
    assert method.config["spq_Q"] == 0.25
    assert method.config["spq_KV"] == 0.5
    assert method.config["sparse_size"] == 64
    assert method.config["sparse_block_size_for_q"] == 64
    assert method.config["sparse_block_size_for_kv"] == 64


def test_flashomni_paper_mmdit_path_is_explicitly_paper_derived():
    method = FlashOmniMethod(
        config={"sparse_pattern": "paper_mmdit", "tau_q": 0.5, "tau_kv": 0.25, "N": 6, "D": 1, "S_q": 0.3},
        model_info=SimpleNamespace(model_type="hunyuan_video"),
    )

    assert method.config["sparse_pattern"] == "paper_mmdit"
    assert method.config["tau_q"] == 0.5
    assert method.config["tau_kv"] == 0.25
    assert method.config["N"] == 6
    assert method.config["D"] == 1
    assert method.config["S_q"] == 0.3
    assert method.config["threshold_q"] == 0.5
    assert method.config["threshold_kv"] == 0.25
    assert method.config["fresh_threshold"] == 6
    assert method.config["max_order"] == 1
    assert method.config["saving_threshold_q_for_taylor"] == 0.3


def test_flashomni_hunyuan_config_uses_upstream_names_as_primary():
    with pytest.warns(RuntimeWarning, match="threshold_q"):
        method = FlashOmniMethod(
            config={
                "sparse_pattern": "paper_mmdit",
                "threshold_q": 0.4,
                "threshold_kv": 0.02,
                "fresh_threshold": 5,
                "max_order": 0,
                "saving_threshold_q_for_taylor": 0.2,
            },
            model_info=SimpleNamespace(model_type="hunyuan_video"),
        )

    assert method.config["threshold_q"] == method.config["tau_q"] == 0.4
    assert method.config["threshold_kv"] == method.config["tau_kv"] == 0.02
    assert method.config["fresh_threshold"] == method.config["N"] == 5
    assert method.config["max_order"] == method.config["D"] == 0
    assert method.config["saving_threshold_q_for_taylor"] == method.config["S_q"] == 0.2


def test_flashomni_hunyuan_config_rejects_conflicting_legacy_aliases():
    with pytest.raises(ValueError, match="threshold_q"):
        FlashOmniMethod(
            config={"sparse_pattern": "paper_mmdit", "threshold_q": 0.6, "tau_q": 0.4},
            model_info=SimpleNamespace(model_type="hunyuan_video"),
        )


def test_flashomni_paper_sparse_blocks_builds_sparse_q_and_kv_masks():
    query = torch.randn(1, 8, 2, 4)
    key = torch.randn(1, 8, 2, 4)

    sparse_q, sparse_kv = flashomni_paper_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        tau_q=0.8,
        tau_kv=0.3,
        S_q=0.0,
        text_len=2,
    )

    assert sparse_q.shape == (1, 2, 4)
    assert sparse_kv.shape == (1, 2, 4, 4)
    assert sparse_q.dtype == torch.uint8
    assert sparse_kv.dtype == torch.uint8
    active_rows = sparse_q.to(torch.bool)
    assert (sparse_kv.sum(dim=-1)[active_rows] >= 1).all()
    assert torch.equal(sparse_q[:, :, -1], torch.ones_like(sparse_q[:, :, -1]))


def test_flashomni_paper_sparse_blocks_match_paper_contribution_guidance_prefix_logic():
    torch.manual_seed(0)
    query = torch.randn(2, 6, 3, 4)
    key = torch.randn(2, 6, 3, 4)
    tau_q = 0.35
    tau_kv = 0.25

    sparse_q, sparse_kv = flashomni_paper_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=1,
        sparse_block_size_for_kv=1,
        tau_q=tau_q,
        tau_kv=tau_kv,
        S_q=0.0,
        text_len=2,
        text_position="prefix",
        sm_scale=1.0,
    )

    pooled_score = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    attention = pooled_score.softmax(dim=-1)
    expected_q = torch.ones_like(sparse_q)
    expected_kv = torch.ones_like(sparse_kv)

    contribution = attention[:, :, :2, 2:].sum(dim=(1, 2))
    guidance = attention[:, :, 2:, :2].transpose(-2, -1).softmax(dim=-1).sum(dim=(1, 2))
    contribution_mask = _expected_low_cdf_mask(contribution, tau_q)
    guidance_mask = _expected_low_cdf_mask(guidance, tau_q)
    cache_mask = contribution_mask & guidance_mask
    assert not torch.equal(contribution_mask, cache_mask)
    for batch in range(query.shape[0]):
        selected = torch.nonzero(cache_mask[batch], as_tuple=False).flatten()
        expected_q[batch, :, 2 + selected] = 0

    img2 = pooled_score[:, :, 2:, :].softmax(dim=-1)
    sorted_kv = torch.sort(img2, dim=-1, descending=False)
    cdf_kv = torch.cumsum(sorted_kv.values, dim=-1)
    threshold = torch.full((query.shape[0], query.shape[2], 4, 1), tau_kv)
    num_kv = torch.searchsorted(cdf_kv, threshold, right=True).squeeze(-1)
    for batch in range(query.shape[0]):
        for head in range(query.shape[2]):
            for q_idx in range(4):
                expected_kv[batch, head, 2 + q_idx, sorted_kv.indices[batch, head, q_idx, : num_kv[batch, head, q_idx]]] = 0
    expected_kv = expected_kv * expected_q.unsqueeze(-1)

    assert torch.equal(sparse_q, expected_q)
    assert torch.equal(sparse_kv, expected_kv)
    assert torch.equal(sparse_q[:, :, :2], torch.ones_like(sparse_q[:, :, :2]))
    assert torch.equal(sparse_kv[:, :, :2], torch.ones_like(sparse_kv[:, :, :2]))


def test_flashomni_paper_sparse_blocks_match_score_cdf_tail_logic_with_trimmed_kv_text():
    torch.manual_seed(0)
    query = torch.randn(2, 6, 3, 4)
    key = torch.randn(2, 5, 3, 4)
    tau_q = 0.35
    tau_kv = 0.25

    sparse_q, sparse_kv = flashomni_paper_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=1,
        sparse_block_size_for_kv=1,
        tau_q=tau_q,
        tau_kv=tau_kv,
        S_q=0.0,
        text_len=2,
        kv_text_len=1,
        text_position="tail",
        sm_scale=1.0,
    )

    pooled_score = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    attention = pooled_score.softmax(dim=-1)
    expected_q = torch.ones_like(sparse_q)
    expected_kv = torch.ones_like(sparse_kv)

    q_text_idx = torch.arange(4, 6)
    q_vision_idx = torch.arange(0, 4)
    kv_text_idx = torch.arange(4, 5)
    kv_vision_idx = torch.arange(0, 4)
    text_to_vision = attention.index_select(-2, q_text_idx).index_select(-1, kv_vision_idx)
    vision_to_text = attention.index_select(-2, q_vision_idx).index_select(-1, kv_text_idx)
    contribution = text_to_vision.sum(dim=(1, 2))
    guidance = vision_to_text.transpose(-2, -1).softmax(dim=-1).sum(dim=(1, 2))
    cache_mask = _expected_low_cdf_mask(contribution, tau_q) & _expected_low_cdf_mask(guidance, tau_q)
    for batch in range(query.shape[0]):
        selected = torch.nonzero(cache_mask[batch], as_tuple=False).flatten()
        expected_q[batch, :, selected] = 0

    img2 = pooled_score.index_select(-2, q_vision_idx).softmax(dim=-1)
    sorted_kv = torch.sort(img2, dim=-1, descending=False)
    cdf_kv = torch.cumsum(sorted_kv.values, dim=-1)
    threshold = torch.full((query.shape[0], query.shape[2], q_vision_idx.numel(), 1), tau_kv)
    num_kv = torch.searchsorted(cdf_kv, threshold, right=True).squeeze(-1)
    for batch in range(query.shape[0]):
        for head in range(query.shape[2]):
            for q_idx in range(q_vision_idx.numel()):
                expected_kv[
                    batch,
                    head,
                    q_idx,
                    sorted_kv.indices[batch, head, q_idx, : num_kv[batch, head, q_idx]],
                ] = 0
    expected_kv = expected_kv * expected_q.unsqueeze(-1)

    assert torch.equal(sparse_q, expected_q)
    assert torch.equal(sparse_kv, expected_kv)
    assert torch.equal(sparse_q[:, :, 4:], torch.ones_like(sparse_q[:, :, 4:]))
    assert torch.equal(sparse_kv[:, :, 4:], torch.ones_like(sparse_kv[:, :, 4:]))


def test_flashomni_hunyuan_sparse_blocks_follow_anonymous_policy_shapes_and_schedule_factor():
    torch.manual_seed(0)
    query = torch.randn(1, 8, 2, 4)
    key = torch.randn(1, 8, 2, 4)

    warmup_q, warmup_kv = flashomni_hunyuan_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        threshold_q=0.5,
        threshold_kv=0.05,
        current_iter=0,
        max_sequence_length=2,
        num_inference_steps=50,
        simthreshd1=-1.0,
        sm_scale=1.0,
    )
    active_q, active_kv = flashomni_hunyuan_sparse_blocks(
        query,
        key,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        threshold_q=0.5,
        threshold_kv=0.05,
        current_iter=50,
        max_sequence_length=2,
        num_inference_steps=50,
        simthreshd1=-1.0,
        sm_scale=1.0,
    )

    assert warmup_q.shape == (1, 2, 4)
    assert warmup_kv.shape == (1, 2, 4, 4)
    assert torch.equal(warmup_q, torch.ones_like(warmup_q))
    assert torch.equal(warmup_kv, torch.ones_like(warmup_kv))
    assert active_q.dtype == torch.uint8
    assert active_kv.dtype == torch.uint8
    assert int(active_kv.sum()) <= int(warmup_kv.sum())


def test_flashomni_hunyuan_sparse_blocks_require_upstream_equal_block_counts():
    query = torch.randn(1, 8, 2, 4)

    with pytest.raises(ValueError, match="equal query/KV block counts"):
        flashomni_hunyuan_sparse_blocks(
            query,
            query,
            sparse_block_size_for_q=2,
            sparse_block_size_for_kv=1,
            threshold_q=0.5,
            threshold_kv=0.05,
            current_iter=8,
            max_sequence_length=2,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON,
    reason="FlashOmni Hunyuan pooling Triton parity requires CUDA and Triton",
)
def test_flashomni_hunyuan_pool_blocks_triton_matches_pytorch_fallback():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.float16)
    x_mean = x.mean(dim=-2, keepdim=True)
    simthreshd1 = torch.full((2,), -2.0, device="cuda")

    pool, sim = _hunyuan_pool_blocks(x, x_mean, 64, simthreshd1)
    expected_pool, expected_sim = _hunyuan_pool_blocks_pytorch(x, x_mean, 64, simthreshd1)

    torch.testing.assert_close(pool.float(), expected_pool.float(), rtol=2e-3, atol=2e-3)
    assert torch.equal(sim, expected_sim)


def test_flashomni_paper_mmdit_schedule_refreshes_symbols_without_dense_dispatch():
    no_symbols = _flashomni_paper_mmdit_schedule(
        1,
        fresh_threshold=6,
        first_enhance=8,
        num_inference_steps=50,
        has_symbols=False,
    )
    first_symbols = _flashomni_paper_mmdit_schedule(
        8,
        fresh_threshold=6,
        first_enhance=8,
        num_inference_steps=50,
        has_symbols=False,
    )
    sparse = _flashomni_paper_mmdit_schedule(
        9,
        fresh_threshold=6,
        first_enhance=8,
        num_inference_steps=50,
        has_symbols=True,
    )
    refresh = _flashomni_paper_mmdit_schedule(
        14,
        fresh_threshold=6,
        first_enhance=8,
        num_inference_steps=50,
        has_symbols=True,
    )
    last = _flashomni_paper_mmdit_schedule(
        50,
        fresh_threshold=6,
        first_enhance=8,
        num_inference_steps=50,
        has_symbols=True,
    )

    assert no_symbols.full is False and no_symbols.compute_symbols is True
    assert first_symbols.full is False and first_symbols.compute_symbols is True
    assert sparse.full is False and sparse.compute_symbols is False
    assert refresh.full is False and refresh.compute_symbols is True
    assert last.full is False and last.compute_symbols is True


def test_flashomni_paper_mmdit_schedule_uses_hunyuan_zero_based_current_step():
    assert _flashomni_paper_mmdit_effective_step(7, {"step": 7}) == 8
    assert _flashomni_paper_mmdit_effective_step(7, None) == 7


def test_flashomni_hunyuan_forward_cache_schedule_matches_anonymous_defaults():
    model = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_single_layers=1))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {
            "fresh_threshold": 6,
            "max_order": 1,
            "first_enhance": 8,
            "saving_threshold_q_for_taylor": 0.3,
            "num_inference_steps": 50,
            "dense_warmup_layer_ratio": 0.0,
        },
    )

    states = []
    for step in range(15):
        current["step"] = step
        cal_type(cache_dic, current)
        current["stream"] = "double_stream"
        current["layer"] = 0
        cache_dic["cache"][-1]["double_stream"][0]["sparse_ratio"] = [torch.tensor(0.5), torch.tensor(0.8)]
        cal_type_sparse(cache_dic, current)
        states.append((current["type"], current.get("flashomni"), current.get("sparse_type")))

    assert states[0] == ("full", False, None)
    assert states[4] == ("full", True, None)
    assert states[5] == ("Sparse", False, "flashomni")
    assert states[13] == ("Sparse", False, "flashomni")


def test_flashomni_hunyuan_forward_layer_warmup_uses_common_ratio():
    model = SimpleNamespace(config=SimpleNamespace(num_layers=2, num_single_layers=2))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {
            "num_inference_steps": 50,
            "dense_warmup_step_ratio": 0.0,
            "dense_warmup_layer_ratio": 0.5,
        },
    )
    current["step"] = 10
    cal_type(cache_dic, current)

    states = []
    for stream, layer in (("double_stream", 0), ("double_stream", 1), ("single_stream", 0)):
        current["stream"] = stream
        current["layer"] = layer
        cal_type_sparse(cache_dic, current)
        states.append((current["type"], current.get("sparse_type")))

    assert states == [("full", None), ("full", None), ("Sparse", "flashomni")]


def test_flashomni_hunyuan_taylor_formula_matches_upstream_default_and_attention_orders():
    model = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_single_layers=0))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {"fresh_threshold": 6, "max_order": 1, "first_enhance": 8, "num_inference_steps": 50},
    )
    current.update({"stream": "double_stream", "layer": 0, "module": "img_attn", "activated_steps": [6, 7]})
    cache_dic["cache"][-1]["double_stream"][0]["img_attn"] = {0: torch.tensor([1.0])}
    current["step"] = 7
    derivative_approximation(cache_dic, current, torch.tensor([3.0]))

    current["step"] = 8
    assert torch.equal(taylor_formula(cache_dic, current), torch.tensor([3.0]))

    cache_dic["cache"][-1]["double_stream"][0]["img_attn"] = {0: torch.tensor([1.0])}
    current["step"] = 7
    derivative_approximation(cache_dic, current, torch.tensor([3.0]), is_attn=True)

    current["step"] = 8
    assert torch.equal(taylor_formula(cache_dic, current), torch.tensor([5.0]))


def test_flashomni_hunyuan_taylor_cache_can_store_on_cpu_and_materialize_to_device():
    model = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_single_layers=0))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {
            "fresh_threshold": 6,
            "max_order": 1,
            "first_enhance": 8,
            "num_inference_steps": 50,
            "taylor_cache_device": "cpu",
        },
    )
    current.update({"stream": "double_stream", "layer": 0, "module": "img_attn", "activated_steps": [6, 7]})
    current["step"] = 7

    feature = torch.tensor([3.0])
    derivative_approximation(cache_dic, current, feature)
    cached = cache_dic["cache"][-1]["double_stream"][0]["img_attn"][0]

    assert cached.device.type == "cpu"
    assert torch.equal(taylor_formula(cache_dic, current, device=feature.device), feature)


def test_flashomni_hunyuan_taylor_start_requires_saved_sparse_ratio():
    model = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_single_layers=0))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {
            "fresh_threshold": 6,
            "max_order": 1,
            "first_enhance": 8,
            "saving_threshold_q_for_taylor": 0.3,
            "num_inference_steps": 50,
        },
    )
    current.update({"stream": "double_stream", "layer": 0, "module": "attn", "type": "full", "flashomni": True})

    _maybe_start_taylor_cache(cache_dic, current)

    assert cache_dic["cache_index"]["taylor_start"]["double_stream"][0] is False


def test_flashomni_hunyuan_forward_patch_installs_and_restores_diffusers_classes():
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTransformer3DModel,
        HunyuanVideoTransformerBlock,
    )

    model = SimpleNamespace()
    model_info = SimpleNamespace(transformers=[model])
    original_single = HunyuanVideoSingleTransformerBlock.forward
    original_block = HunyuanVideoTransformerBlock.forward
    original_model = HunyuanVideoTransformer3DModel.forward

    restore = install_flashomni_hunyuan_forward_patch(
        model_info,
        {"fresh_threshold": 6, "max_order": 1, "first_enhance": 8},
    )
    try:
        assert HunyuanVideoSingleTransformerBlock.forward is not original_single
        assert HunyuanVideoTransformerBlock.forward is not original_block
        assert HunyuanVideoTransformer3DModel.forward is not original_model
        assert model._sparsevideo_flashomni_config["fresh_threshold"] == 6
    finally:
        restore()

    assert HunyuanVideoSingleTransformerBlock.forward is original_single
    assert HunyuanVideoTransformerBlock.forward is original_block
    assert HunyuanVideoTransformer3DModel.forward is original_model
    assert not hasattr(model, "_sparsevideo_flashomni_config")


def test_flashomni_hunyuan_forward_patch_updates_accelerate_old_forward():
    def original_forward():
        return "original"

    model = SimpleNamespace(_old_forward=original_forward, transformer_blocks=[], single_transformer_blocks=[])
    model_info = SimpleNamespace(transformers=[model])

    restore = install_flashomni_hunyuan_forward_patch(
        model_info,
        {"fresh_threshold": 6, "max_order": 1, "first_enhance": 8},
    )
    try:
        assert model._old_forward.__func__ is flashomni_hunyuan_forward
    finally:
        restore()

    assert model._old_forward is original_forward


def test_flashomni_hunyuan_debug_memory_trace_records_cache_bytes():
    trace = []
    model = SimpleNamespace(config=SimpleNamespace(num_layers=1, num_single_layers=0))
    cache_dic, current = _flashomni_hunyuan_cache_init(
        model,
        {
            "fresh_threshold": 6,
            "max_order": 1,
            "first_enhance": 8,
            "num_inference_steps": 50,
            "debug_memory": True,
            "debug_memory_max_events": 2,
            "_memory_trace": trace,
        },
    )
    current.update(stream="double_stream", layer=0, module="img_mlp", type="full")
    cache_dic["cache"][-1]["double_stream"][0]["img_mlp"] = {0: torch.zeros(2, 3, dtype=torch.float32)}

    _flashomni_trace_memory(cache_dic, current, "before_ff", feature=torch.zeros(1, 2, dtype=torch.float32))
    _flashomni_trace_memory(cache_dic, current, "after_ff")
    _flashomni_trace_memory(cache_dic, current, "after_cache")

    assert [item["event"] for item in trace] == ["after_ff", "after_cache"]
    assert trace[-1]["cache"]["tensor_count"] == 1
    assert trace[-1]["cache"]["total_gb"] > 0
    assert trace[-1]["cache"]["top"][0]["path"].endswith("img_mlp.0")


def test_flashomni_paper_mmdit_processor_generates_sparse_patterns(monkeypatch):
    calls = []

    def fake_upstream(q, k, v, block_mask_pattern, **kwargs):
        calls.append(
            {
                "block_mask_pattern_shape": None if block_mask_pattern is None else tuple(block_mask_pattern.shape),
                "sparse_q_shape": None
                if kwargs.get("sparse_q_block_pattern") is None
                else tuple(kwargs["sparse_q_block_pattern"].shape),
                "text_len": kwargs["text_len"],
                "q_block_size": kwargs["q_block_size"],
                "kv_block_size": kwargs["kv_block_size"],
                "is_full": kwargs.get("is_full", False),
            }
        )
        if kwargs.get("return_layout") == "NHD" and kwargs.get("input_layout") == "NHD":
            return torch.zeros_like(q)
        if kwargs.get("return_layout") == "NHD":
            return torch.zeros(q.shape[0], q.shape[2], q.shape[1], q.shape[3], dtype=q.dtype, device=q.device)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        fake_upstream,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    method = FlashOmniMethod(
        config={
            "sparse_pattern": "paper_mmdit",
            "sparse_block_size_for_q": 2,
            "sparse_block_size_for_kv": 2,
            "tau_q": 0.0,
            "tau_kv": 0.3,
            "N": 3,
            "first_enhance": 0,
            "dense_warmup_step_ratio": 0.0,
            "dense_warmup_layer_ratio": 0.0,
        },
        model_info=SimpleNamespace(model_type="hunyuan_video"),
    )
    step_tracker = SimpleNamespace(step=1)
    processor = method.create_processor(layer_idx=0, total_layers=1, original_processor=None, step_tracker=step_tracker)
    query = torch.randn(1, 8, 2, 4)

    processor.attn_fn(query, query, query, None, text_len=2)
    step_tracker.step = 2
    processor.attn_fn(query, query, query, None, text_len=2)

    assert calls == [
        {
            "block_mask_pattern_shape": (1, 2, 4, 4),
            "sparse_q_shape": (1, 2, 4),
            "text_len": 2,
            "q_block_size": 2,
            "kv_block_size": 2,
            "is_full": False,
        },
        {
            "block_mask_pattern_shape": (1, 2, 4, 4),
            "sparse_q_shape": (1, 2, 4),
            "text_len": 2,
            "q_block_size": 2,
            "kv_block_size": 2,
            "is_full": False,
        },
    ]
    assert method.runtime_summary()["dispatch_counts"] == {"sparse": 2}
    assert method.runtime_summary()["backend_counts"] == {
        "flashomni_explicit_upstream": 2,
    }


def test_flashomni_paper_mmdit_dispatch_reuses_cached_q_blocks(monkeypatch):
    sparse_q = torch.tensor([[[0, 1]]], dtype=torch.uint8)
    sparse_kv = torch.ones(1, 1, 2, 2, dtype=torch.uint8)

    def fake_policy(*args, **kwargs):
        return sparse_q, sparse_kv

    def fake_upstream(q, k, v, block_mask_pattern, **kwargs):
        if kwargs.get("is_full"):
            return torch.full_like(q, 7.0)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method.flashomni_paper_sparse_blocks",
        fake_policy,
    )
    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        fake_upstream,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    state = _FlashOmniPaperMMDiTState()
    query = torch.randn(1, 4, 1, 4)
    update = _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.4,
        tau_kv=0.01,
        N=3,
        D=0,
        S_q=0.0,
        text_len=0,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        implementation="upstream",
        backend="auto",
        workspace_bytes=1,
        state=state,
        step=1,
    )
    dispatch = _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.4,
        tau_kv=0.01,
        N=3,
        D=0,
        S_q=0.0,
        text_len=0,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        implementation="upstream",
        backend="auto",
        workspace_bytes=1,
        state=state,
        step=2,
    )

    assert update.dispatch == "sparse"
    assert dispatch.dispatch == "sparse"
    assert torch.equal(update.output, torch.zeros_like(update.output))
    assert torch.equal(dispatch.output, torch.zeros_like(dispatch.output))


def test_flashomni_paper_mmdit_isolates_mochi_cache_suffixes(monkeypatch):
    calls = []

    def fake_policy(query, key, **kwargs):
        q_block_size = int(kwargs["sparse_block_size_for_q"])
        kv_block_size = int(kwargs["sparse_block_size_for_kv"])
        q_blocks = math.ceil(query.shape[1] / q_block_size)
        kv_blocks = math.ceil(key.shape[1] / kv_block_size)
        sparse_q = torch.zeros(query.shape[0], query.shape[2], q_blocks, dtype=torch.uint8, device=query.device)
        sparse_kv = torch.ones(
            query.shape[0],
            query.shape[2],
            q_blocks,
            kv_blocks,
            dtype=torch.uint8,
            device=query.device,
        )
        return sparse_q, sparse_kv

    def fake_upstream(q, k, v, block_mask_pattern, **kwargs):
        calls.append((q.shape[1], bool(kwargs.get("is_full"))))
        if kwargs.get("is_full"):
            return torch.full_like(q, 7.0)
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method.flashomni_paper_sparse_blocks",
        fake_policy,
    )
    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        fake_upstream,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    method = FlashOmniMethod(
        config={
            "sparse_pattern": "paper_mmdit",
            "sparse_block_size_for_q": 2,
            "sparse_block_size_for_kv": 2,
            "tau_q": 0.0,
            "tau_kv": 0.3,
            "N": 3,
            "first_enhance": 0,
            "dense_warmup_layer_ratio": 0.0,
            "dense_warmup_step_ratio": 0.0,
        },
        model_info=SimpleNamespace(model_type="mochi"),
    )
    step_tracker = SimpleNamespace(step=1)
    processor = method.create_processor(layer_idx=0, total_layers=1, original_processor=None, step_tracker=step_tracker)
    q4 = torch.randn(1, 4, 1, 4)
    q6 = torch.randn(1, 6, 1, 4)

    processor.attn_fn(q4, q4, q4, None, text_len=0, cache_key_suffix=0)
    processor.attn_fn(q6, q6, q6, None, text_len=0, cache_key_suffix=1)
    step_tracker.step = 2
    processor.attn_fn(q4, q4, q4, None, text_len=0, cache_key_suffix=0)
    processor.attn_fn(q6, q6, q6, None, text_len=0, cache_key_suffix=1)

    assert calls == [(4, False), (6, False), (4, False), (6, False)]
    assert method.runtime_summary()["dispatch_counts"] == {"sparse": 4}


def test_flashomni_paper_mmdit_attention_saves_hunyuan_sparse_ratios_for_taylor(monkeypatch):
    sparse_q = torch.tensor([[[0, 1]]], dtype=torch.uint8)
    sparse_kv = torch.tensor([[[[0, 1], [1, 1]]]], dtype=torch.uint8)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method.flashomni_hunyuan_sparse_blocks",
        lambda *args, **kwargs: (sparse_q, sparse_kv),
    )
    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        lambda q, k, v, block_mask_pattern, **kwargs: torch.zeros_like(q),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    query = torch.randn(1, 4, 1, 4)
    cache_dic = {"cache": {-1: {"double_stream": {0: {}}}}}
    current = {"stream": "double_stream", "layer": 0}
    _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.5,
        tau_kv=0.05,
        N=6,
        D=1,
        S_q=0.3,
        text_len=2,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        implementation="upstream",
        backend="auto",
        workspace_bytes=1,
        state=_FlashOmniPaperMMDiTState(),
        step=8,
        cache_dic=cache_dic,
        current=current,
        first_enhance=8,
        model_type="hunyuan_video",
    )

    ratios = cache_dic["cache"][-1]["double_stream"][0]["sparse_ratio"]
    assert torch.equal(ratios[0], torch.tensor(0.5))
    assert torch.equal(ratios[1], torch.tensor(0.75))


def test_flashomni_paper_mmdit_cache_order_predicts_cached_blocks_and_gemm_bias():
    sparse_q = torch.tensor([[[0, 1]]], dtype=torch.uint8)
    state = _FlashOmniPaperMMDiTState()

    state.append_output(torch.full((1, 4, 1, 1), 1.0), sparse_q, q_block_size=2, order=2)
    state.append_gemm_o_bias(torch.tensor([1.0]), order=2)
    state.append_output(torch.full((1, 4, 1, 1), 2.0), sparse_q, q_block_size=2, order=2)
    state.append_gemm_o_bias(torch.tensor([2.0]), order=2)

    first_order = state.predicted_output(order=1)
    out = _flashomni_apply_cached_q_blocks(
        torch.zeros((1, 4, 1, 1)),
        first_order,
        sparse_q,
        q_block_size=2,
    )
    assert torch.equal(out[:, :2], torch.full((1, 2, 1, 1), 3.0))
    assert torch.equal(out[:, 2:], torch.zeros((1, 2, 1, 1)))
    assert torch.equal(state.predicted_gemm_o_bias(order=1), torch.tensor([3.0]))

    state.append_output(torch.full((1, 4, 1, 1), 4.0), sparse_q, q_block_size=2, order=2)
    state.append_gemm_o_bias(torch.tensor([4.0]), order=2)

    second_order = state.predicted_output(order=2)
    out = _flashomni_apply_cached_q_blocks(
        torch.zeros((1, 4, 1, 1)),
        second_order,
        sparse_q,
        q_block_size=2,
    )
    assert torch.equal(out[:, :2], torch.full((1, 2, 1, 1), 6.5))
    assert torch.equal(out[:, 2:], torch.zeros((1, 2, 1, 1)))
    assert torch.equal(state.predicted_gemm_o_bias(order=2), torch.tensor([6.5]))


def test_flashomni_paper_mmdit_trims_hunyuan_prefix_mask_before_native(monkeypatch):
    calls = {}

    def fake_policy(query, key, **kwargs):
        calls["policy_query_len"] = query.shape[1]
        calls["policy_key_len"] = key.shape[1]
        calls["policy_text_len"] = kwargs["text_len"]
        calls["policy_kv_text_len"] = kwargs["kv_text_len"]
        return (
            torch.ones(1, 1, 3, dtype=torch.uint8),
            torch.ones(1, 1, 3, 2, dtype=torch.uint8),
        )

    def fake_upstream(q, k, v, block_mask_pattern, **kwargs):
        # NHD layout: (B, N, H, D) — sequence length is dim 1
        calls["native_q_len"] = q.shape[1]
        calls["native_k_len"] = k.shape[1]
        calls["native_attention_mask"] = kwargs["attention_mask"]
        calls["native_text_len"] = kwargs["text_len"]
        calls["native_is_full"] = kwargs["is_full"]
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method.flashomni_paper_sparse_blocks",
        fake_policy,
    )
    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        fake_upstream,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    query = torch.randn(1, 6, 1, 4)
    attention_mask = torch.tensor([[[[True, True, True, False, False, False]]]])

    _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.5,
        tau_kv=0.05,
        N=6,
        D=1,
        S_q=0.3,
        text_len=4,
        sparse_block_size_for_q=2,
        sparse_block_size_for_kv=2,
        implementation="upstream",
        backend="auto",
        workspace_bytes=1,
        attention_mask=attention_mask,
        state=_FlashOmniPaperMMDiTState(),
        step=1,
    )

    assert calls["policy_query_len"] == 6
    assert calls["policy_key_len"] == 3
    assert calls["policy_text_len"] == 4
    assert calls["policy_kv_text_len"] == 1
    assert calls["native_q_len"] == 6
    assert calls["native_k_len"] == 3
    assert calls["native_attention_mask"] is None
    assert calls["native_text_len"] == 1
    assert calls["native_is_full"] is True


def test_flashomni_cached_q_block_helper_checks_shapes():
    out = torch.zeros(1, 4, 1, 2)
    cached = torch.ones_like(out)
    sparse_q = torch.tensor([[[0, 1]]], dtype=torch.uint8)

    mixed = _flashomni_apply_cached_q_blocks(out, cached, sparse_q, q_block_size=2)

    assert torch.equal(mixed[:, :2], torch.ones_like(mixed[:, :2]))
    assert torch.equal(mixed[:, 2:], torch.zeros_like(mixed[:, 2:]))


def test_flashomni_cached_q_block_helper_can_keep_large_cache_on_cpu(monkeypatch):
    import sparsevideo.methods.flashomni.method as flashomni_method

    monkeypatch.setattr(flashomni_method, "_FLASHOMNI_Q_BLOCK_CACHE_CPU_THRESHOLD_BYTES", 1)
    out = torch.arange(1 * 5 * 2 * 2, dtype=torch.float32).reshape(1, 5, 2, 2)
    sparse_q = torch.tensor([[[0, 1, 0], [1, 0, 0]]], dtype=torch.uint8)

    cached = _flashomni_cache_q_blocks(out, sparse_q, q_block_size=2)

    assert cached.values.device.type == "cpu"
    mixed = _flashomni_apply_cached_q_blocks(torch.zeros_like(out), cached, sparse_q, q_block_size=2)
    assert torch.equal(mixed[:, 0:2, 0], out[:, 0:2, 0])
    assert torch.equal(mixed[:, 2:4, 0], torch.zeros_like(out[:, 2:4, 0]))
    assert torch.equal(mixed[:, 4:5, 0], out[:, 4:5, 0])
    assert torch.equal(mixed[:, 0:2, 1], torch.zeros_like(out[:, 0:2, 1]))
    assert torch.equal(mixed[:, 2:5, 1], out[:, 2:5, 1])


def test_flashomni_sparse_q_info_pack_uses_upstream_gemm_layout():
    calls = {}

    def fake_segment_packbits(values, indptr, bitorder):
        calls["values"] = values.clone()
        calls["indptr"] = indptr.clone()
        calls["bitorder"] = bitorder
        return values.clone(), indptr.clone()

    flashomni = SimpleNamespace(segment_packbits=fake_segment_packbits)
    sparse_q = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.uint8)

    packed, indptr = _flashomni_pack_sparse_q_info(
        flashomni,
        sparse_q,
        q_len=4,
        sparse_q_size=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(calls["values"], torch.tensor([1, 0, 0, 1], dtype=torch.uint8))
    assert torch.equal(calls["indptr"], torch.tensor([0, 4], dtype=torch.int32))
    assert calls["bitorder"] == "little"
    assert torch.equal(packed, calls["values"])
    assert torch.equal(indptr, calls["indptr"])


def test_flashomni_runtime_kernel_counts_are_separate_from_attention_dispatch():
    method = FlashOmniMethod(
        config={"sparse_pattern": "global_random"},
        model_info=SimpleNamespace(model_type="wan"),
    )

    method.record_runtime_kernel("flashomni_sparse_q_gemm", layer_idx=3, step=4)
    method.record_runtime_kernel("flashomni_sparse_q_gemm", layer_idx=3, step=4)

    runtime = method.runtime_summary()
    assert runtime["total_calls"] == 0
    assert runtime["dispatch_counts"] == {}
    assert runtime["backend_counts"] == {}
    assert runtime["kernel_counts"] == {"flashomni_sparse_q_gemm": 2}
    assert runtime["last_kernel"] == {
        "kernel": "flashomni_sparse_q_gemm",
        "layer_idx": 3,
        "step": 4,
    }


def test_flashomni_sparse_q_gemm_calls_owned_runtime(monkeypatch):
    calls = {}

    class FakeFlashOmni:
        def segment_packbits(self, values, indptr, bitorder):
            return values, indptr

        def flashomni_gemm(
            self,
            A,
            B,
            num_qo_heads,
            sparse_info,
            sparse_info_indptr,
            num_text_tokens,
            bias,
            out=None,
            sparse_q_size=128,
            is_full=False,
        ):
            calls["A_shape"] = tuple(A.shape)
            calls["B_shape"] = tuple(B.shape)
            calls["num_qo_heads"] = num_qo_heads
            calls["sparse_info"] = sparse_info.clone()
            calls["sparse_info_indptr"] = sparse_info_indptr.clone()
            calls["num_text_tokens"] = num_text_tokens
            calls["bias_is_linear_bias"] = bias is linear.bias
            calls["out_shape"] = tuple(out.shape)
            calls["sparse_q_size"] = sparse_q_size
            calls["is_full"] = is_full
            out.fill_(3)
            return out

    monkeypatch.setattr("sparsevideo.methods.flashomni.method._flashomni_import", lambda: FakeFlashOmni())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    linear = torch.nn.Linear(32, 256, dtype=torch.float16)
    hidden_states = torch.randn(1, 4, 32, dtype=torch.float16)
    sparse_q = torch.tensor([[[1], [0]]], dtype=torch.uint8)

    out = _flashomni_sparse_q_gemm(linear, hidden_states, sparse_q, num_heads=2, sparse_q_size=128)

    assert out.shape == (1, 4, 256)
    assert torch.equal(out, torch.full_like(out, 3))
    assert calls["A_shape"] == (1, 128, 32)
    assert calls["B_shape"] == (256, 32)
    assert calls["num_qo_heads"] == 2
    assert torch.equal(calls["sparse_info"], torch.tensor([1, 0], dtype=torch.uint8))
    assert torch.equal(calls["sparse_info_indptr"], torch.tensor([0, 2], dtype=torch.int32))
    assert calls["num_text_tokens"] == 0
    assert calls["bias_is_linear_bias"] is True
    assert calls["out_shape"] == (1, 128, 256)
    assert calls["sparse_q_size"] == 128
    assert calls["is_full"] is False


def test_flashomni_sparse_o_gemm_cache_and_active_calls_owned_runtime(monkeypatch):
    calls = []

    class FakeFlashOmni:
        def segment_packbits(self, values, indptr, bitorder):
            return values, indptr

        def flashomni_gemm_reduction(
            self,
            A,
            B,
            num_qo_heads,
            sparse_info,
            sparse_info_indptr,
            num_text_tokens,
            bias=None,
            is_for_cache=False,
            sparse_q_size=128,
        ):
            calls.append(
                {
                    "A_shape": tuple(A.shape),
                    "B_shape": tuple(B.shape),
                    "num_qo_heads": num_qo_heads,
                    "sparse_info": sparse_info.clone(),
                    "sparse_info_indptr": sparse_info_indptr.clone(),
                    "num_text_tokens": num_text_tokens,
                    "bias": bias,
                    "is_for_cache": is_for_cache,
                    "sparse_q_size": sparse_q_size,
                }
            )
            fill = 5 if is_for_cache else 7
            return torch.full((A.shape[0], A.shape[1], B.shape[0]), fill, dtype=A.dtype)

    monkeypatch.setattr("sparsevideo.methods.flashomni.method._flashomni_import", lambda: FakeFlashOmni())

    linear = torch.nn.Linear(256, 128, dtype=torch.float16)
    hidden_states = torch.randn(1, 4, 256, dtype=torch.float16)
    sparse_q = torch.tensor([[[1], [0]]], dtype=torch.uint8)

    cache_bias = _flashomni_sparse_o_gemm_cache_bias(
        linear,
        hidden_states,
        sparse_q,
        num_heads=2,
        sparse_q_size=128,
    )
    out = _flashomni_sparse_o_gemm(
        linear,
        hidden_states,
        sparse_q,
        num_heads=2,
        sparse_q_size=128,
        cache_bias=cache_bias,
    )

    assert cache_bias.shape == (1, 4, 128)
    assert out.shape == (1, 4, 128)
    assert torch.equal(cache_bias, torch.full_like(cache_bias, 5))
    assert torch.equal(out, torch.full_like(out, 7))
    assert calls[0]["is_for_cache"] is True
    assert calls[0]["bias"] is linear.bias
    assert calls[1]["is_for_cache"] is False
    assert calls[1]["bias"].shape == (1, 128, 128)
    assert torch.equal(calls[1]["bias"][:, :4], cache_bias)
    assert torch.equal(calls[1]["bias"][:, 4:], torch.zeros_like(calls[1]["bias"][:, 4:]))
    for call in calls:
        assert call["A_shape"] == (1, 128, 256)
        assert call["B_shape"] == (128, 256)
        assert call["num_qo_heads"] == 2
        assert torch.equal(call["sparse_info"], torch.tensor([1, 0], dtype=torch.uint8))
        assert torch.equal(call["sparse_info_indptr"], torch.tensor([0, 2], dtype=torch.int32))
        assert call["num_text_tokens"] == 0
        assert call["sparse_q_size"] == 128


def test_flashomni_global_random_sparse_blocks_match_upstream_zero_counts():
    torch.manual_seed(0)
    sparse_q, sparse_kv = _flashomni_global_random_sparse_blocks(
        batch_size=2,
        num_heads=3,
        q_len=257,
        kv_len=385,
        spq_Q=0.25,
        spq_KV=0.5,
        sparse_size=128,
        device=torch.device("cpu"),
    )

    assert sparse_q.shape == (2, 3, 3)
    assert sparse_kv.shape == (2, 3, 3, 4)
    assert int((sparse_q == 0).sum().item()) == int(2 * 3 * 3 * 0.25)
    assert int((sparse_kv == 0).sum().item()) == int(2 * 3 * 3 * 4 * 0.5)
    assert sparse_q.dtype == torch.uint8
    assert sparse_kv.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="upstream FlashOmni benchmark helper hardcodes CUDA")
def test_flashomni_global_random_sparse_blocks_match_upstream_benchmark_helper():
    upstream = _load_upstream_flashomni_benchmark_utils()

    torch.manual_seed(1234)
    expected_sparse_info, expected_sparse_q, expected_sparse_kv_info, expected_sparse_kv = upstream.get_qkvo_global_sparse(
        2,
        3,
        257,
        385,
        spq_Q=0.25,
        spq_KV=0.5,
        sparse_size=128,
    )
    torch.manual_seed(1234)
    sparse_q, sparse_kv = _flashomni_global_random_sparse_blocks(
        batch_size=2,
        num_heads=3,
        q_len=257,
        kv_len=385,
        spq_Q=0.25,
        spq_KV=0.5,
        sparse_size=128,
        device=torch.device("cuda"),
    )

    assert torch.equal(sparse_q, expected_sparse_q)
    assert torch.equal(sparse_kv, expected_sparse_kv)
    assert torch.equal(
        sparse_q.transpose(1, 2).contiguous().view(-1, sparse_q.shape[1]),
        expected_sparse_info,
    )
    assert torch.equal(
        sparse_kv.transpose(1, 2).contiguous().view(-1, sparse_kv.shape[1], sparse_kv.shape[-1]),
        expected_sparse_kv_info,
    )


def test_flashomni_global_random_processor_passes_q_and_kv_patterns(monkeypatch):
    calls = {}

    def fake_upstream(q, k, v, block_mask_pattern, **kwargs):
        calls["block_mask_pattern_shape"] = tuple(block_mask_pattern.shape)
        calls["sparse_q_shape"] = tuple(kwargs["sparse_q_block_pattern"].shape)
        calls["q_block_size"] = kwargs["q_block_size"]
        calls["kv_block_size"] = kwargs["kv_block_size"]
        return torch.empty_like(q)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_upstream_attention",
        fake_upstream,
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    method = FlashOmniMethod(
        config={"sparse_pattern": "global_random", "spq_Q": 0.0, "spq_KV": 0.5},
        model_info=SimpleNamespace(model_type="wan"),
    )
    processor = method.create_processor(layer_idx=0, total_layers=1, original_processor=None, step_tracker=None)
    query = torch.randn(1, 257, 2, 4)

    processor.attn_fn(query, query, query, None)

    assert calls == {
        "block_mask_pattern_shape": (1, 2, 3, 3),
        "sparse_q_shape": (1, 2, 3),
        "q_block_size": 128,
        "kv_block_size": 128,
    }


def test_flashomni_flex_is_only_local_diagnostic_path():
    with pytest.raises(NotImplementedError, match="local_qk_topk"):
        FlashOmniMethod(
            config={"implementation": "flex"},
            model_info=SimpleNamespace(model_type="wan"),
        )


def test_flashomni_rejects_public_is_full_dense_switch():
    with pytest.raises(ValueError, match="is_full"):
        FlashOmniMethod(
            config={"is_full": True},
            model_info=SimpleNamespace(model_type="wan"),
        )


def test_flashomni_explicit_path_forwards_attention_mask_as_custom_mask(monkeypatch):
    calls = {}

    def fake_flashomni_attention(query, key, value, **kwargs):
        calls["attention_mask"] = kwargs["attention_mask"]
        calls["text_len"] = kwargs["text_len"]
        return torch.empty_like(query)

    monkeypatch.setattr(
        "sparsevideo.methods.flashomni.method._flashomni_explicit_attention",
        fake_flashomni_attention,
    )
    sparse_info = torch.ones(1, dtype=torch.uint8)
    sparse_kv_info = torch.ones(1, dtype=torch.uint8)
    sparse_info_indptr = torch.tensor([0, 1], dtype=torch.int32)
    sparse_kv_info_indptr = torch.tensor([0, 1], dtype=torch.int32)
    method = FlashOmniMethod(
        config={
            "sparse_pattern": "explicit",
            "sparse_info": sparse_info,
            "sparse_kv_info": sparse_kv_info,
            "sparse_info_indptr": sparse_info_indptr,
            "sparse_kv_info_indptr": sparse_kv_info_indptr,
        },
        model_info=SimpleNamespace(model_type="wan"),
    )
    processor = method.create_processor(layer_idx=0, total_layers=1, original_processor=None, step_tracker=None)
    query = torch.randn(1, 8, 2, 4)
    attention_mask = torch.ones(1, 3, dtype=torch.bool)

    processor.attn_fn(query, query, query, attention_mask, text_len=3)

    assert calls["attention_mask"] is attention_mask
    assert calls["text_len"] == 3


def test_flashomni_custom_mask_accepts_full_sequence_mask():
    attention_mask = torch.tensor([[True, False, True, True]])

    custom_mask = _flashomni_custom_mask_from_attention_mask(
        attention_mask,
        batch_size=1,
        q_len=4,
        kv_len=4,
        device=attention_mask.device,
    )

    assert custom_mask.shape == (16,)
    assert custom_mask.view(4, 4).tolist() == [
        [True, False, True, True],
        [True, False, True, True],
        [True, False, True, True],
        [True, False, True, True],
    ]


def test_flashomni_custom_mask_skips_noop_mask():
    custom_mask = _flashomni_custom_mask_from_attention_mask(
        torch.ones(1, 4, dtype=torch.bool),
        batch_size=1,
        q_len=4,
        kv_len=4,
        device=torch.device("cpu"),
    )

    assert custom_mask is None


def test_flashomni_custom_mask_accepts_hunyuan_text_tail_mask():
    attention_mask = torch.tensor([[True, False]])

    custom_mask = _flashomni_custom_mask_from_attention_mask(
        attention_mask,
        batch_size=1,
        q_len=5,
        kv_len=5,
        text_len=2,
        device=attention_mask.device,
    )

    assert custom_mask.view(5, 5)[0].tolist() == [True, True, True, True, False]


def test_flashomni_prefix_mask_trim_rejects_non_prefix_masks():
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 4)
    attention_mask = torch.tensor([[[[True, False, True, True, False]]]])

    trim = _flashomni_trim_prefix_key_value_mask(key, value, attention_mask, text_len=2)

    assert trim.key is key
    assert trim.value is value
    assert trim.attention_mask is attention_mask
    assert trim.kv_text_len == 2


def test_flashomni_upstream_attention_prepacks_custom_mask(monkeypatch):
    calls = {}

    class _Wrapper:
        def __init__(self, workspace, kv_layout, backend):
            pass

        def plan(self, *args, **kwargs):
            calls["custom_mask"] = kwargs["custom_mask"]
            calls["packed_custom_mask"] = kwargs["packed_custom_mask"]
            qo_indptr, kv_indptr = args[:2]
            q_blocks = torch.ceil(
                (qo_indptr[1:] - qo_indptr[:-1]) / kwargs["sparse_block_size_for_q"]
            ).to(torch.int32)
            kv_blocks = torch.ceil(
                (kv_indptr[1:] - kv_indptr[:-1]) / kwargs["sparse_block_size_for_kv"]
            ).to(torch.int32)
            self._sparse_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_info_indptr_base[1:] = torch.cumsum(q_blocks * kwargs["num_qo_heads"], 0)
            self._sparse_kv_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_kv_info_indptr_base[1:] = torch.cumsum(
                q_blocks * kv_blocks * kwargs["num_qo_heads"], 0
            )

        def run(self, q, k, v, *args):
            return q

    def fake_segment_packbits(values, indptr, bitorder):
        seglen = indptr[1:] - indptr[:-1]
        packed_len = (seglen + 7) // 8
        packed_indptr = torch.zeros_like(indptr)
        packed_indptr[1:] = torch.cumsum(packed_len, 0)
        return torch.ones(int(packed_indptr[-1].item()), dtype=torch.uint8), packed_indptr

    fake_flashomni = SimpleNamespace(
        attention=SimpleNamespace(BatchFlashOmniFAWithRaggedKVWrapper=_Wrapper),
        segment_packbits=fake_segment_packbits,
    )
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._flashomni_import", lambda: fake_flashomni)

    q = torch.randn(1, 2, 4, 8)
    sparse_info = torch.ones(2, dtype=torch.uint8)
    sparse_kv_info = torch.ones(2, dtype=torch.uint8)
    sparse_info_indptr = torch.tensor([0, 2], dtype=torch.int32)
    sparse_kv_info_indptr = torch.tensor([0, 2], dtype=torch.int32)

    _flashomni_upstream_attention(
        q,
        q,
        q,
        None,
        q_len=4,
        q_block_size=128,
        kv_block_size=128,
        backend="auto",
        workspace_bytes=1,
        sparse_info=sparse_info,
        sparse_kv_info=sparse_kv_info,
        sparse_info_indptr=sparse_info_indptr,
        sparse_kv_info_indptr=sparse_kv_info_indptr,
        attention_mask=torch.tensor([[True, False, True, True]], dtype=torch.bool),
    )

    assert calls["custom_mask"] is None
    assert torch.equal(calls["packed_custom_mask"], torch.ones(2, dtype=torch.uint8))


def test_flashomni_upstream_attention_forwards_gqa_and_value_dim_to_plan(monkeypatch):
    calls = {}

    class _Wrapper:
        def __init__(self, workspace, kv_layout, backend):
            calls["workspace_shape"] = tuple(workspace.shape)
            calls["kv_layout"] = kv_layout
            calls["backend"] = backend

        def plan(self, qo_indptr, kv_indptr, **kwargs):
            calls["qo_indptr"] = qo_indptr.clone()
            calls["kv_indptr"] = kv_indptr.clone()
            calls["plan_kwargs"] = dict(kwargs)
            q_blocks = torch.ceil(
                (qo_indptr[1:] - qo_indptr[:-1]) / kwargs["sparse_block_size_for_q"]
            ).to(torch.int32)
            kv_blocks = torch.ceil(
                (kv_indptr[1:] - kv_indptr[:-1]) / kwargs["sparse_block_size_for_kv"]
            ).to(torch.int32)
            self._sparse_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_info_indptr_base[1:] = torch.cumsum(q_blocks * kwargs["num_qo_heads"], 0)
            self._sparse_kv_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_kv_info_indptr_base[1:] = torch.cumsum(
                q_blocks * kv_blocks * kwargs["num_qo_heads"], 0
            )

        def run(
            self,
            q,
            k,
            v,
            sparse_info,
            sparse_kv_info,
            sparse_info_indptr,
            sparse_kv_info_indptr,
            is_full,
        ):
            calls["run_shapes"] = {
                "q": tuple(q.shape),
                "k": tuple(k.shape),
                "v": tuple(v.shape),
            }
            calls["is_full"] = is_full
            return torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=v.dtype)

    def fake_segment_packbits(values, indptr, bitorder):
        return values.to(torch.uint8).contiguous(), indptr.to(torch.int32).contiguous()

    fake_flashomni = SimpleNamespace(
        attention=SimpleNamespace(BatchFlashOmniFAWithRaggedKVWrapper=_Wrapper),
        segment_packbits=fake_segment_packbits,
    )
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._flashomni_import", lambda: fake_flashomni)

    q = torch.randn(1, 4, 6, 8)
    k = torch.randn(1, 2, 10, 8)
    v = torch.randn(1, 2, 10, 16)

    out = _flashomni_upstream_attention(
        q,
        k,
        v,
        None,
        q_len=6,
        q_block_size=4,
        kv_block_size=5,
        backend="fa2",
        workspace_bytes=16,
        is_full=True,
        causal=True,
        pos_encoding_mode="ALIBI",
        use_fp16_qk_reduction=True,
        logits_soft_cap=30.0,
        sm_scale=0.5,
        rope_scale=2.0,
        rope_theta=1000.0,
    )

    assert calls["workspace_shape"] == (16,)
    assert calls["kv_layout"] == "NHD"
    assert calls["backend"] == "fa2"
    assert torch.equal(calls["qo_indptr"], torch.tensor([0, 6], dtype=torch.int32))
    assert torch.equal(calls["kv_indptr"], torch.tensor([0, 10], dtype=torch.int32))
    assert calls["plan_kwargs"]["num_qo_heads"] == 4
    assert calls["plan_kwargs"]["num_kv_heads"] == 2
    assert calls["plan_kwargs"]["head_dim_qk"] == 8
    assert calls["plan_kwargs"]["head_dim_vo"] == 16
    assert calls["plan_kwargs"]["causal"] is True
    assert calls["plan_kwargs"]["pos_encoding_mode"] == "ALIBI"
    assert calls["plan_kwargs"]["use_fp16_qk_reduction"] is True
    assert calls["plan_kwargs"]["logits_soft_cap"] == 30.0
    assert calls["plan_kwargs"]["sm_scale"] == 0.5
    assert calls["plan_kwargs"]["rope_scale"] == 2.0
    assert calls["plan_kwargs"]["rope_theta"] == 1000.0
    assert calls["run_shapes"] == {
        "q": (6, 4, 8),
        "k": (10, 2, 8),
        "v": (10, 2, 16),
    }
    assert calls["is_full"] is True
    assert out.shape == (1, 4, 6, 16)


def test_flashomni_upstream_attention_pads_non_native_head_dim(monkeypatch):
    calls = {}

    class _Wrapper:
        def __init__(self, workspace, kv_layout, backend):
            pass

        def plan(self, qo_indptr, kv_indptr, **kwargs):
            calls["plan_kwargs"] = dict(kwargs)
            q_blocks = torch.ceil(
                (qo_indptr[1:] - qo_indptr[:-1]) / kwargs["sparse_block_size_for_q"]
            ).to(torch.int32)
            kv_blocks = torch.ceil(
                (kv_indptr[1:] - kv_indptr[:-1]) / kwargs["sparse_block_size_for_kv"]
            ).to(torch.int32)
            self._sparse_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_info_indptr_base[1:] = torch.cumsum(q_blocks * kwargs["num_qo_heads"], 0)
            self._sparse_kv_info_indptr_base = torch.zeros_like(qo_indptr)
            self._sparse_kv_info_indptr_base[1:] = torch.cumsum(
                q_blocks * kv_blocks * kwargs["num_qo_heads"], 0
            )

        def run(self, q, k, v, *args):
            calls["run_shapes"] = {
                "q": tuple(q.shape),
                "k": tuple(k.shape),
                "v": tuple(v.shape),
            }
            return torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=v.dtype)

    fake_flashomni = SimpleNamespace(
        attention=SimpleNamespace(BatchFlashOmniFAWithRaggedKVWrapper=_Wrapper),
        segment_packbits=lambda values, indptr, bitorder: (values.to(torch.uint8), indptr.to(torch.int32)),
    )
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._flashomni_import", lambda: fake_flashomni)

    q = torch.randn(1, 2, 6, 96)
    out = _flashomni_upstream_attention(
        q,
        q,
        q,
        None,
        q_len=6,
        q_block_size=4,
        kv_block_size=4,
        backend="fa2",
        workspace_bytes=16,
        is_full=True,
    )

    assert _flashomni_native_head_dim(96) == 128
    assert calls["plan_kwargs"]["head_dim_qk"] == 128
    assert calls["plan_kwargs"]["head_dim_vo"] == 128
    assert calls["plan_kwargs"]["sm_scale"] == pytest.approx(96 ** -0.5)
    assert calls["run_shapes"] == {
        "q": (6, 2, 128),
        "k": (6, 2, 128),
        "v": (6, 2, 128),
    }
    assert out.shape == (1, 2, 6, 96)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashOmni native smoke requires CUDA")
def test_flashomni_owned_native_extension_executes_full_attention_cuda():
    try:
        flashomni = _flashomni_import()
    except ImportError as exc:
        pytest.skip(f"SparseVideo-owned FlashOmni extension is not built: {exc}")

    torch.manual_seed(0)
    batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 64
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    out = _flashomni_upstream_attention(
        query,
        key,
        value,
        None,
        q_len=seq_len,
        q_block_size=8,
        kv_block_size=8,
        backend="auto",
        workspace_bytes=8 * 1024 * 1024,
        is_full=True,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert "src/sparsevideo/kernels/native/flashomni" in str(Path(flashomni.__file__))
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashOmni paper_mmdit native smoke requires CUDA")
def test_flashomni_paper_mmdit_owned_native_extension_executes_cuda():
    try:
        _flashomni_import()
    except ImportError as exc:
        pytest.skip(f"SparseVideo-owned FlashOmni extension is not built: {exc}")

    torch.manual_seed(0)
    query = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)

    state = _FlashOmniPaperMMDiTState()
    _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.0,
        tau_kv=0.2,
        N=3,
        D=0,
        S_q=0.0,
        text_len=0,
        sparse_block_size_for_q=8,
        sparse_block_size_for_kv=8,
        implementation="upstream",
        backend="auto",
        workspace_bytes=8 * 1024 * 1024,
        state=state,
        step=1,
    )
    result = _flashomni_paper_mmdit_attention(
        query,
        query,
        query,
        tau_q=0.0,
        tau_kv=0.2,
        N=3,
        D=0,
        S_q=0.0,
        text_len=0,
        sparse_block_size_for_q=8,
        sparse_block_size_for_kv=8,
        implementation="upstream",
        backend="auto",
        workspace_bytes=8 * 1024 * 1024,
        state=state,
        step=2,
    )
    out = result.output

    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashOmni sparse GEMM native smoke requires CUDA")
def test_flashomni_owned_sparse_gemm_helpers_execute_cuda():
    try:
        _flashomni_import()
    except ImportError as exc:
        pytest.skip(f"SparseVideo-owned FlashOmni extension is not built: {exc}")

    torch.manual_seed(0)
    batch_size, seq_len, in_dim, num_heads, head_dim = 1, 16, 64, 2, 128
    sparse_q = torch.tensor([[[1], [0]]], device="cuda", dtype=torch.uint8)
    hidden_states = torch.randn(batch_size, seq_len, in_dim, device="cuda", dtype=torch.float16)
    q_linear = torch.nn.Linear(in_dim, num_heads * head_dim, bias=True, dtype=torch.float16).cuda()

    q_out = _flashomni_sparse_q_gemm(
        q_linear,
        hidden_states,
        sparse_q,
        num_heads=num_heads,
        sparse_q_size=128,
    ).view(batch_size, seq_len, num_heads, head_dim)
    expected_q = q_linear(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
    expected_q[:, :, 1] = 0
    torch.testing.assert_close(q_out, expected_q, rtol=2e-2, atol=2e-2)

    o_linear = torch.nn.Linear(num_heads * head_dim, 128, bias=True, dtype=torch.float16).cuda()
    attn_out = torch.randn(batch_size, seq_len, num_heads * head_dim, device="cuda", dtype=torch.float16)
    cache_bias = _flashomni_sparse_o_gemm_cache_bias(
        o_linear,
        attn_out,
        sparse_q,
        num_heads=num_heads,
        sparse_q_size=128,
    )
    o_out = _flashomni_sparse_o_gemm(
        o_linear,
        attn_out,
        sparse_q,
        num_heads=num_heads,
        sparse_q_size=128,
        cache_bias=cache_bias,
    )
    torch.testing.assert_close(o_out, o_linear(attn_out), rtol=3e-2, atol=3e-2)


def test_flashomni_explicit_sparse_info_packs_upstream_unpacked_bits():
    calls = {}

    def fake_segment_packbits(values, indptr, bitorder):
        calls["values"] = values.clone()
        calls["indptr"] = indptr.clone()
        calls["bitorder"] = bitorder
        return torch.tensor([0b10101101, 0b00000011], dtype=torch.uint8), torch.tensor([0, 2], dtype=torch.int32)

    flashomni = SimpleNamespace(segment_packbits=fake_segment_packbits)

    packed, packed_indptr = _flashomni_normalize_sparse_bits(
        flashomni,
        torch.ones(10, dtype=torch.uint8),
        torch.tensor([0, 10], dtype=torch.int32),
        torch.tensor([0, 10], dtype=torch.int32),
        "sparse_info",
        torch.device("cpu"),
    )

    assert torch.equal(calls["values"], torch.ones(10, dtype=torch.uint8))
    assert torch.equal(calls["indptr"], torch.tensor([0, 10], dtype=torch.int32))
    assert calls["bitorder"] == "little"
    assert torch.equal(packed, torch.tensor([0b10101101, 0b00000011], dtype=torch.uint8))
    assert torch.equal(packed_indptr, torch.tensor([0, 2], dtype=torch.int32))


def test_flashomni_explicit_sparse_info_accepts_already_packed_bits():
    calls = {"packed": False}

    def fake_segment_packbits(values, indptr, bitorder):
        calls["packed"] = True
        return values, indptr

    flashomni = SimpleNamespace(segment_packbits=fake_segment_packbits)

    packed, packed_indptr = _flashomni_normalize_sparse_bits(
        flashomni,
        torch.tensor([0b10101101, 0b00000011], dtype=torch.uint8),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 10], dtype=torch.int32),
        "sparse_info",
        torch.device("cpu"),
    )

    assert calls["packed"] is False
    assert torch.equal(packed, torch.tensor([0b10101101, 0b00000011], dtype=torch.uint8))
    assert torch.equal(packed_indptr, torch.tensor([0, 2], dtype=torch.int32))


def test_flashomni_explicit_sparse_info_rejects_wrong_layout():
    flashomni = SimpleNamespace(segment_packbits=lambda values, indptr, bitorder: (values, indptr))

    with pytest.raises(RuntimeError, match="does not match the current wrapper layout"):
        _flashomni_normalize_sparse_bits(
            flashomni,
            torch.ones(3, dtype=torch.uint8),
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([0, 10], dtype=torch.int32),
            "sparse_info",
            torch.device("cpu"),
        )


def test_flashomni_training_free_runtime_detection():
    module = SimpleNamespace(__file__="/repo/training_free/FlashOmni/flashomni/__init__.py")

    assert _is_training_free_runtime(module) is True


def test_flashomni_detects_local_owned_extension(tmp_path):
    package = tmp_path / "flashomni"
    (package / "jit").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "jit" / "aot_config.py").write_text("prebuilt_ops_uri = set()\n", encoding="utf-8")
    (package / "flashomni_kernels.abi3.so").write_bytes(b"")

    assert _has_flashomni_extension(tmp_path)


def test_flashomni_detects_owned_aot_config_layout(tmp_path):
    package = tmp_path / "flashomni"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "aot_config.py").write_text("prebuilt_ops_uri = set()\n", encoding="utf-8")
    (package / "flashomni_kernels.abi3.so").write_bytes(b"")

    assert _has_flashomni_extension(tmp_path)


def test_flashomni_loader_prefers_sparsevideo_owned_root(monkeypatch, tmp_path):
    package = tmp_path / "flashomni"
    (package / "jit").mkdir(parents=True)
    (package / "__init__.py").write_text("SELECTED = 'local'\n", encoding="utf-8")
    (package / "jit" / "aot_config.py").write_text("prebuilt_ops_uri = set()\n", encoding="utf-8")
    (package / "flashomni_kernels.abi3.so").write_bytes(b"")
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._candidate_flashomni_roots", lambda: [tmp_path])
    _clear_flashomni_modules()

    module = _flashomni_import()

    assert module.SELECTED == "local"


def test_flashomni_loader_caches_owned_runtime(monkeypatch, tmp_path):
    package = tmp_path / "flashomni"
    (package / "jit").mkdir(parents=True)
    (package / "__init__.py").write_text("SELECTED = 'cached'\n", encoding="utf-8")
    (package / "jit" / "aot_config.py").write_text("prebuilt_ops_uri = set()\n", encoding="utf-8")
    (package / "flashomni_kernels.abi3.so").write_bytes(b"")
    monkeypatch.setattr("sparsevideo.methods.flashomni.method._candidate_flashomni_roots", lambda: [tmp_path])
    _clear_flashomni_modules()

    first = _flashomni_import()
    second = _flashomni_import()

    assert first is second
    assert second.SELECTED == "cached"


def test_flashomni_env_root_cannot_select_external_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSEVIDEO_FLASHOMNI_ROOT", str(tmp_path))

    with pytest.raises(ImportError, match="outside the SparseVideo-owned runtime root"):
        _candidate_flashomni_roots()


def test_flashomni_env_root_rejects_training_free_runtime(monkeypatch, tmp_path):
    upstream_root = tmp_path / "training_free" / "FlashOmni"
    monkeypatch.setenv("SPARSEVIDEO_FLASHOMNI_ROOT", str(upstream_root))

    with pytest.raises(ImportError, match="inside training_free"):
        _candidate_flashomni_roots()


def test_flashomni_loader_rejects_environment_runtime_without_owned_root(monkeypatch, tmp_path):
    from sparsevideo.methods.flashomni import method as flashomni_method

    env_package = tmp_path / "envsite" / "flashomni"
    env_package.mkdir(parents=True)
    (env_package / "__init__.py").write_text("SELECTED = 'environment'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "envsite"))
    monkeypatch.setattr(flashomni_method, "_candidate_flashomni_roots", lambda: [])
    _clear_flashomni_modules()

    with pytest.raises(ImportError, match="SparseVideo-owned FlashOmni package"):
        _flashomni_import()


def test_flashomni_owned_runtime_sources_match_upstream_references():
    repo_root = Path(__file__).resolve().parents[1]
    owned_root = repo_root / "src/sparsevideo/kernels/native/flashomni"
    upstream_root = repo_root / "training_free/FlashOmni"

    for relative_path in ["custom_backend.py", "pyproject.toml", "setup.py", "version.txt"]:
        owned = owned_root / relative_path
        upstream = upstream_root / relative_path
        assert owned.read_bytes() == upstream.read_bytes()

    for relative_path in [
        "aot_build_utils",
        "csrc",
        "flashomni",
        "include/flashomni",
        "3rdparty/cutlass/include",
    ]:
        owned = owned_root / relative_path
        upstream = upstream_root / relative_path
        owned_files = sorted(
            path.relative_to(owned)
            for path in owned.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in {"_build_meta.py", "flashomni_kernels.abi3.so"}
        )
        upstream_files = sorted(
            path.relative_to(upstream)
            for path in upstream.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in {"_build_meta.py", "flashomni_kernels.abi3.so"}
        )
        assert owned_files == upstream_files
        for source in owned_files:
            assert (owned / source).read_bytes() == (upstream / source).read_bytes()
