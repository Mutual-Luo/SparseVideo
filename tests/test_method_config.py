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
    assert sparsevideo.default_method_config("svg1")["context_length"] is None
    assert sparsevideo.default_method_config("svg1")["prompt_length"] is None
    assert sparsevideo.default_method_config("svg2")["top_p_kmeans"] == 0.9
    assert sparsevideo.default_method_config("svg2")["context_length"] is None
    assert sparsevideo.default_method_config("svg2")["prompt_length"] is None
    assert sparsevideo.default_method_config("svg2")["allow_triton_fallback"] is False
    assert sparsevideo.default_method_config("draft")["sparsity_ratio"] == 0.9
    assert sparsevideo.default_method_config("draft")["allow_triton_fallback"] is False
    assert sparsevideo.default_method_config("adacluster")["topk_num"] == 128
    assert sparsevideo.default_method_config("sta")["tile_size"] == [6, 8, 8]
    assert sparsevideo.default_method_config("sta")["window_size"] == [3, 6, 10]
    assert sparsevideo.default_method_config("sta")["STA_mode"] == "STA_inference"
    assert sparsevideo.default_method_config("flashomni")["implementation"] == "upstream"
    assert sparsevideo.default_method_config("flashomni")["backend"] == "auto"
    assert sparsevideo.default_method_config("flashomni")["sparse_pattern"] == "explicit"
    assert sparsevideo.default_method_config("flashomni")["sparse_size"] == 128
    assert sparsevideo.default_method_config("flashomni")["spq_Q"] == 0.0
    assert sparsevideo.default_method_config("flashomni")["spq_KV"] == 0.8
    assert sparsevideo.default_method_config("flashomni")["tau_q"] == 0.5
    assert sparsevideo.default_method_config("flashomni")["tau_kv"] == 0.05
    assert sparsevideo.default_method_config("flashomni")["N"] == 6
    assert sparsevideo.default_method_config("flashomni")["D"] == 1
    assert sparsevideo.default_method_config("flashomni")["S_q"] == 0.3
    assert sparsevideo.default_method_config("flashomni")["use_sparse_gemm"] is True
    assert sparsevideo.default_method_config("flashomni")["causal"] is False
    assert sparsevideo.default_method_config("flashomni")["pos_encoding_mode"] == "NONE"
    assert sparsevideo.default_method_config("flashomni")["use_fp16_qk_reduction"] is False
    assert sparsevideo.default_method_config("flashomni")["logits_soft_cap"] == 0.0
    assert sparsevideo.default_method_config("flashomni")["sm_scale"] is None
    assert sparsevideo.default_method_config("flashomni")["rope_scale"] is None
    assert sparsevideo.default_method_config("flashomni")["rope_theta"] is None
    assert sparsevideo.default_method_config("radial")["allow_flex_fallback"] is False
    assert sparsevideo.default_method_config("spargeattn")["model_out_path"] is None
    assert sparsevideo.default_method_config("spargeattn")["mode"] == "topk"
    assert sparsevideo.default_method_config("spargeattn")["topk"] == 0.5
    assert sparsevideo.default_method_config("spargeattn")["cdfthreshd"] == 0.98
    assert sparsevideo.default_method_config("spargeattn")["pvthreshd"] == 50
    assert sparsevideo.default_method_config("spargeattn")["tensor_layout"] == "HND"
    assert sparsevideo.default_method_config("spargeattn")["sim_rule"] == "l1"
    assert sparsevideo.default_method_config("spargeattn")["l1"] == 0.07
    assert sparsevideo.default_method_config("spargeattn")["pv_l1"] == 0.08
    assert sparsevideo.default_method_config("spargeattn")["cos_sim"] == 0.98
    assert sparsevideo.default_method_config("spargeattn")["rmse"] == 0.07
    assert sparsevideo.default_method_config("spargeattn")["rearrange_kwargs"] == {}
    assert sparsevideo.default_method_config("spargeattn")["tune_pv"] is True
    hunyuan_sparge = sparsevideo.default_method_config("spargeattn", model_family="hunyuan_video")
    assert hunyuan_sparge["l1"] == 0.07
    assert hunyuan_sparge["pv_l1"] == 0.08
    assert hunyuan_sparge["tune_pv"] is True
    assert sparsevideo.default_method_config("svoo")["sparsity_csv_path"] == "sparsity_profiles/sparsity_results.csv"
    assert sparsevideo.default_method_config("svoo")["implementation"] == "native"
    assert sparsevideo.default_method_config("svoo")["sparse_backend"] == "flashinfer"
    assert sparsevideo.default_method_config("svoo")["enable_mem_save"] is True
    assert sparsevideo.default_method_config("svoo")["context_length"] is None
    assert sparsevideo.default_method_config("svoo")["prompt_length"] is None


def test_svoo_enable_mem_save_follows_upstream_env_default(monkeypatch):
    from sparsevideo.methods.svoo.config import default_config

    monkeypatch.delenv("SVOO_ENABLE_MEM_SAVE", raising=False)
    assert default_config()["enable_mem_save"] is True

    monkeypatch.setenv("SVOO_ENABLE_MEM_SAVE", "0")
    assert default_config()["enable_mem_save"] is False

    monkeypatch.setenv("SVOO_ENABLE_MEM_SAVE", "1")
    assert default_config()["enable_mem_save"] is True


def test_svoo_inference_context_uses_upstream_720p_defaults():
    wan = sparsevideo.default_method_config("svoo", model_family="wan", model_key="wan21-t2v-1.3b")
    assert wan["num_q_centroids"] == 256
    assert wan["num_k_centroids"] == 1024
    assert wan["kmeans_iter_init"] == 2
    assert wan["kmeans_iter_step"] == 2
    assert wan["use_dynamic_min_kc_ratio"] is True
    assert wan["sparsity_csv_path"].endswith("sparsity_wan_1.3B_t2v.csv")
    assert Path(wan["sparsity_csv_path"]).exists()

    wan22 = sparsevideo.default_method_config("svoo", model_family="wan", model_key="wan22-t2v-a14b")
    assert wan22["num_inference_steps"] == 40
    assert wan22["start_reuse_step"] == 9
    assert wan22["sparsity_csv_path"].endswith("sparsity_wan22_A14B_t2v.csv")
    assert Path(wan22["sparsity_csv_path"]).exists()

    hunyuan = sparsevideo.default_method_config("svoo", model_family="hunyuan_video")
    assert hunyuan["top_p_kmeans"] == 0.88
    assert hunyuan["start_reuse_step"] == 6
    assert hunyuan["sparsity_csv_path"].endswith("sparsity_hunyuan10_13B_t2v.csv")
    assert Path(hunyuan["sparsity_csv_path"]).exists()


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
    assert hunyuan["zero_step_kmeans_init"] is True
    assert hunyuan["context_length"] == 256
    assert hunyuan["prompt_length"] is None


def test_svg1_inference_context_uses_upstream_720p_defaults():
    wan = sparsevideo.default_method_config("svg1", model_family="wan", model_key="wan21-t2v-1.3b")
    assert wan["first_times_fp"] == 0.2
    assert wan["first_layers_fp"] == 0.03
    assert wan["num_sampled_rows"] == 64
    assert wan["sparsity"] == 0.3

    hunyuan = sparsevideo.default_method_config("svg1", model_family="hunyuan_video")
    assert hunyuan["first_times_fp"] == 0.1
    assert hunyuan["context_length"] == 256
    assert hunyuan["prompt_length"] is None
    assert hunyuan["first_layers_fp"] == 0.03
    assert hunyuan["num_sampled_rows"] == 64
    assert hunyuan["sparsity"] == 0.25


def test_radial_inference_context_uses_upstream_shell_defaults():
    wan = sparsevideo.default_method_config("radial", model_family="wan", model_key="wan21-t2v-14b")
    assert wan["dense_layers"] == 1
    assert wan["dense_timesteps"] == 12
    assert wan["decay_factor"] == 0.2
    assert wan["block_size"] == 128

    wan22 = sparsevideo.default_method_config("radial", model_family="wan", model_key="wan22-t2v-a14b")
    assert wan22["dense_layers"] == 1
    assert wan22["dense_timesteps"] == 11
    assert wan22["decay_factor"] == 0.8
    assert wan22["block_size"] == 64

    hunyuan = sparsevideo.default_method_config("radial", model_family="hunyuan_video")
    assert hunyuan["dense_layers"] == 0
    assert hunyuan["dense_timesteps"] == 12
    assert hunyuan["decay_factor"] == 0.95


def test_draft_inference_context_uses_upstream_defaults():
    wan = sparsevideo.default_method_config("draft", model_family="wan", model_key="wan21-t2v-14b")
    assert wan["pool_h"] == 8
    assert wan["pool_w"] == 16
    assert wan["latent_h"] is None
    assert wan["latent_w"] is None
    assert wan["visual_len"] is None
    assert wan["text_len"] == 0
    assert wan["sparsity_ratio"] == 0.75
    assert wan["batch_size"] is None
    assert wan["allow_triton_fallback"] is False

    hunyuan = sparsevideo.default_method_config("draft", model_family="hunyuan_video")
    assert hunyuan["pool_h"] == 8
    assert hunyuan["pool_w"] == 16
    assert hunyuan["latent_h"] == 48
    assert hunyuan["latent_w"] == 80
    assert hunyuan["visual_len"] == 126_720
    assert hunyuan["text_len"] == 256
    assert hunyuan["sparsity_ratio"] == 0.9


def test_sta_inference_context_uses_upstream_text_boundary_defaults():
    wan = sparsevideo.default_method_config("sta", model_family="wan", model_key="wan21-t2v-1.3b")
    assert wan["tile_size"] == [6, 8, 8]
    assert wan["window_size"] == [3, 6, 10]
    assert wan["has_text"] is False
    assert wan["STA_mode"] == "STA_inference"
    assert wan["mask_strategy_file_path"] is None

    wan14b = sparsevideo.default_method_config("sta", model_family="wan", model_key="wan21-t2v-14b")
    assert wan14b["mask_strategy_file_path"].endswith("mask_strategy_wan.json")
    assert Path(wan14b["mask_strategy_file_path"]).exists()

    hunyuan = sparsevideo.default_method_config("sta", model_family="hunyuan_video")
    assert hunyuan["tile_size"] == [6, 8, 8]
    assert hunyuan["window_size"] == [5, 6, 10]
    assert hunyuan["has_text"] is True
    assert hunyuan["mask_strategy_file_path"].endswith("mask_strategy_hunyuan.json")
    assert Path(hunyuan["mask_strategy_file_path"]).exists()


def test_svoo_apply_api_defaults_use_model_context():
    from sparsevideo.methods.svoo import SVOOMethod

    method = SVOOMethod(config={}, model_info=SimpleNamespace(model_type="wan", transformers=[object()]))
    assert method.config["num_q_centroids"] == 256
    assert method.config["kmeans_iter_step"] == 2
    assert method.config["sparsity_csv_path"].endswith("sparsity_wan_1.3B_t2v.csv")
    assert Path(method.config["sparsity_csv_path"]).exists()

    wan22 = SVOOMethod(config={}, model_info=SimpleNamespace(model_type="wan", transformers=[object(), object()]))
    assert wan22.config["num_inference_steps"] == 40
    assert wan22.config["start_reuse_step"] == 9
    assert wan22.config["sparsity_csv_path"].endswith("sparsity_wan22_A14B_t2v.csv")

    wan14 = SVOOMethod(
        config={},
        model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-14b", transformers=[object()]),
    )
    assert wan14.config["sparsity_csv_path"].endswith("sparsity_wan_14B_t2v.csv")


def test_svoo_rejects_missing_dynamic_sparsity_profile():
    from sparsevideo.methods.svoo import SVOOMethod

    with pytest.raises(FileNotFoundError, match="sparsity_csv_path"):
        SVOOMethod(
            config={"sparsity_csv_path": "/tmp/sparsevideo-missing-svoo-profile.csv"},
            model_info=SimpleNamespace(model_type="wan", transformers=[object()]),
        )


def test_base_method_defaults_use_model_context():
    from sparsevideo.methods.adacluster import AdaClusterMethod
    from sparsevideo.methods.draft import DraftMethod
    from sparsevideo.methods.radial import RadialMethod
    from sparsevideo.methods.sta import STAMethod

    method = AdaClusterMethod(config={}, model_info=SimpleNamespace(model_type="hunyuan_video"))
    assert method.config["topk_num"] == 94
    assert method.config["q_kernel_num"] == 250
    assert method.config["kv_kernel_num"] == 1243

    draft = DraftMethod(config={}, model_info=SimpleNamespace(model_type="wan"))
    assert draft.config["sparsity_ratio"] == 0.75
    assert draft.config["allow_triton_fallback"] is False

    radial = RadialMethod(config={}, model_info=SimpleNamespace(model_type="hunyuan_video"))
    assert radial.config["dense_timesteps"] == 12
    assert radial.config["decay_factor"] == 0.95

    sta_hunyuan = STAMethod(config={}, model_info=SimpleNamespace(model_type="hunyuan_video"))
    assert sta_hunyuan.config["window_size"] == [5, 6, 10]
    assert sta_hunyuan.config["has_text"] is True
    assert sta_hunyuan.config["mask_strategy_file_path"].endswith("mask_strategy_hunyuan.json")

    sta_wan = STAMethod(config={}, model_info=SimpleNamespace(model_type="wan", model_key="wan21-t2v-14b"))
    assert sta_wan.config["window_size"] == [3, 6, 10]
    assert sta_wan.config["has_text"] is False
    assert sta_wan.config["mask_strategy_file_path"].endswith("mask_strategy_wan.json")


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
    assert config["q_kernel_num"] == 250
    assert config["kv_kernel_num"] == 1243
    assert config["use_thresholded_kmeans_loop"] is False
    assert "late_q_kernel_num" not in config
    assert "late_kv_kernel_num" not in config


def test_contextual_defaults_match_wan_adacluster_runwan_fixed_clusters():
    config = sparsevideo.default_method_config("adacluster", model_family="wan")
    assert config["topk_num"] == 128
    assert config["q_kernel_num"] == 100
    assert config["kv_kernel_num"] == 500
    assert config["use_thresholded_kmeans_loop"] is False
    assert config["initial_q_kernel_num"] == 50
    assert config["initial_kv_kernel_num"] == 200
    assert config["q_distance_threshold"] == 9.0
    assert config["kv_distance_threshold"] == 5.5
    assert config["thresholded_kmeans_iter_time"] == 3
    assert config["thresholded_kmeans_max_iterations"] == 10


def test_local_aliases_are_not_primary_api():
    assert sparsevideo.normalize_method_config("svg2", {"top_p_kmeans": 0.8}) == {"top_p_kmeans": 0.8}
    assert sparsevideo.normalize_method_config("svoo", {"top_p_kmeans": 0.8}) == {"top_p_kmeans": 0.8}
    assert sparsevideo.normalize_method_config("spargeattn", {"value": 0.4}) == {"value": 0.4}
    assert sparsevideo.normalize_method_config("flashomni", {"sparse_kv_budget": 0.5}) == {"sparse_kv_budget": 0.5}
    assert sparsevideo.normalize_method_config("flashomni", {"cdfthreshd": 0.2}) == {"tau_kv": 0.2}
    assert sparsevideo.normalize_method_config("flashomni", {"tau_c": 0.3}) == {"tau_q": 0.3}

    local_aliases = {
        "svg1": {"skip_first_steps": 0.2},
        "svg2": {"budget": 0.2},
        "svoo": {"kmeans_iters": 2},
        "spargeattn": {"mask": "mask.pt"},
        "flashomni": {"pattern_source": "global_random"},
        "radial": {"skip_first_layers": 1},
        "sta": {"kernel_size": [3, 6, 10]},
        "adacluster": {"num_clusters": 200},
        "draft": {"budget": 0.2},
    }
    for method, config in local_aliases.items():
        with pytest.raises(ValueError):
            sparsevideo.normalize_method_config(method, config)

    with pytest.raises(ValueError):
        sparsevideo.normalize_method_config("adacluster", {"late_q_kernel_num": 250})


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

    full_map, q_sizes, k_sizes, _ = pad_text_clusters(
        dynamic_map,
        torch.ones(1, 1, 2, dtype=torch.long),
        torch.ones(1, 1, 3, dtype=torch.long),
        torch.arange(6, dtype=torch.int32).unsqueeze(0),
        text_len=4,
        prompt_length=4,
    )
    assert full_map.shape == (1, 1, 4, 5)
    assert q_sizes.tolist() == [[[1, 1, 4, 0]]]
    assert k_sizes.tolist() == [[[1, 1, 1, 4, 0]]]
    assert full_map[0, 0, 3, 4]
