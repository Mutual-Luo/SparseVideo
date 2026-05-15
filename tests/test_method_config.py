from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo


def test_public_methods_are_registered():
    assert sparsevideo.list_methods() == [
        "adacluster",
        "dense",
        "draft",
        "flashomni",
        "radial",
        "spargeattn",
        "sta",
        "svg1",
        "svg2",
        "svoo",
    ]


def test_upstream_default_names_are_exposed():
    assert sparsevideo.default_method_config("svg1")["sparsity"] == 0.25
    assert sparsevideo.default_method_config("svg2")["top_p_kmeans"] == 0.9
    assert sparsevideo.default_method_config("draft")["sparsity_ratio"] == 0.9
    assert sparsevideo.default_method_config("adacluster")["topk_num"] == 128
    assert sparsevideo.default_method_config("sta")["tile_size"] == [6, 8, 8]
    assert sparsevideo.default_method_config("sta")["window_size"] == [3, 3, 5]
    assert sparsevideo.default_method_config("flashomni")["implementation"] == "upstream"
    assert sparsevideo.default_method_config("flashomni")["backend"] == "auto"
    assert sparsevideo.default_method_config("svoo")["sparsity_csv_path"] == "sparsity_profiles/sparsity_results.csv"
    assert sparsevideo.default_method_config("svoo")["implementation"] == "native"
    assert sparsevideo.default_method_config("svoo")["sparse_backend"] == "flashinfer"
    assert sparsevideo.default_method_config("svoo")["context_length"] is None
    assert sparsevideo.default_method_config("svoo")["prompt_length"] is None


def test_svoo_inference_context_uses_upstream_720p_defaults():
    wan = sparsevideo.default_method_config("svoo", model_family="wan", model_key="wan21-t2v-1.3b")
    assert wan["num_q_centroids"] == 256
    assert wan["num_k_centroids"] == 1024
    assert wan["kmeans_iter_init"] == 2
    assert wan["kmeans_iter_step"] == 2
    assert wan["use_dynamic_min_kc_ratio"] is True

    hunyuan = sparsevideo.default_method_config("svoo", model_family="hunyuan_video")
    assert hunyuan["top_p_kmeans"] == 0.88
    assert hunyuan["start_reuse_step"] == 6


def test_svg2_inference_context_uses_upstream_720p_defaults():
    wan = sparsevideo.default_method_config("svg2", model_family="wan", model_key="wan21-t2v-1.3b")
    assert wan["first_times_fp"] == 0.2
    assert wan["first_layers_fp"] == 0.03
    assert wan["num_q_centroids"] == 300
    assert wan["num_k_centroids"] == 1000
    assert wan["min_kc_ratio"] == 0.10
    assert wan["kmeans_iter_init"] == 50
    assert wan["kmeans_iter_step"] == 2

    hunyuan = sparsevideo.default_method_config("svg2", model_family="hunyuan_video")
    assert hunyuan["num_q_centroids"] == 400
    assert hunyuan["num_k_centroids"] == 1000
    assert hunyuan["first_times_fp"] == 0.1


def test_svoo_apply_api_defaults_use_model_context():
    from sparsevideo.methods.svoo import SVOOMethod

    method = SVOOMethod(config={}, model_info=SimpleNamespace(model_type="wan", transformers=[object()]))
    assert method.config["num_q_centroids"] == 256
    assert method.config["kmeans_iter_step"] == 2

    wan22 = SVOOMethod(config={}, model_info=SimpleNamespace(model_type="wan", transformers=[object(), object()]))
    assert wan22.config["start_reuse_step"] == 9


def test_base_method_defaults_use_model_context():
    from sparsevideo.methods.adacluster import AdaClusterMethod

    method = AdaClusterMethod(config={}, model_info=SimpleNamespace(model_type="hunyuan_video"))
    assert method.config["topk_num"] == 94


def test_draft_dense_switch_is_not_exposed_as_sparse_method():
    from sparsevideo.methods.draft import DraftMethod

    with pytest.raises(NotImplementedError, match="block_sparse_attention=False"):
        DraftMethod(
            config={"block_sparse_attention": False},
            model_info=SimpleNamespace(model_type="wan"),
        )


def test_contextual_defaults_match_hunyuan_adacluster():
    config = sparsevideo.default_method_config("adacluster", model_family="hunyuan_video")
    assert config["topk_num"] == 94
    assert config["late_q_kernel_num"] == 250
    assert config["late_kv_kernel_num"] == 1243


def test_compatibility_aliases_do_not_become_primary_api():
    assert sparsevideo.normalize_method_config("svg2", {"budget": 0.8}) == {"top_p_kmeans": 0.8}
    with pytest.raises(ValueError):
        sparsevideo.normalize_method_config("draft", {"budget": 0.2})


def test_720p_token_layout_inference_matches_upstream_shapes():
    from sparsevideo.methods._layout import infer_video_frame_count, infer_video_frame_shape

    assert infer_video_frame_count(21 * 45 * 80, model_type="wan") == 21
    assert infer_video_frame_shape(21 * 45 * 80, model_type="wan") == (21, 45, 80)
    assert infer_video_frame_shape(18 * 48 * 80, model_type="wan") == (18, 48, 80)
    assert infer_video_frame_shape(33 * 45 * 80, model_type="hunyuan_video") == (33, 45, 80)
    assert infer_video_frame_shape(30 * 48 * 80, model_type="hunyuan_video") == (30, 48, 80)


def test_sta_seq_shape_override_must_match_video_tokens():
    from sparsevideo.methods.sta.method import _infer_video_shape

    assert _infer_video_shape(18 * 48 * 80, model_type="wan", seq_shape="18x48x80") == (18, 48, 80)
    with pytest.raises(ValueError, match="does not match"):
        _infer_video_shape(21 * 45 * 80, model_type="wan", seq_shape="18x48x80")


def test_svoo_hunyuan_text_padding_cluster_matches_upstream_policy():
    from sparsevideo.methods.svoo.text import pad_text_clusters
    import torch

    dynamic_map = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    q_sizes = torch.ones(1, 1, 2, dtype=torch.long)
    k_sizes = torch.ones(1, 1, 3, dtype=torch.long)
    q_sorted_indices = torch.arange(6, dtype=torch.int32).unsqueeze(0)

    full_map, q_sizes, k_sizes, q_sorted_indices = pad_text_clusters(
        dynamic_map, q_sizes, k_sizes, q_sorted_indices, text_len=4, prompt_length=3,
    )

    assert full_map.shape == (1, 1, 4, 5)
    assert q_sizes.tolist() == [[[1, 1, 3, 1]]]
    assert k_sizes.tolist() == [[[1, 1, 1, 3, 1]]]
    assert full_map[0, 0, :2, 3].all()
    assert not full_map[0, 0, :2, 4].any()
    assert full_map[0, 0, 2, :4].all()
    assert not full_map[0, 0, 2, 4]
    assert full_map[0, 0, 3, 4]
    assert not full_map[0, 0, 3, :4].any()
    assert q_sorted_indices.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
