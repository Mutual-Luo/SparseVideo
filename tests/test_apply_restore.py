from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo
from sparsevideo._diffsynth import _patch_diffsynth_ltx2_attention_forward
from sparsevideo._model_info import discover_model
from sparsevideo._step_tracker import StepTracker
from sparsevideo.processors.allegro import SparseAllegroAttnProcessor
from sparsevideo.processors.cogvideox import SparseCogVideoXAttnProcessor
from sparsevideo.processors.easyanimate import SparseEasyAnimateAttnProcessor
from sparsevideo.processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
from sparsevideo.processors.ltx_video import SparseLTXVideoAttnProcessor
from sparsevideo.processors.mochi import SparseMochiAttnProcessor
from sparsevideo.processors.wan import SparseWanAttnProcessor


class _Hook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _Attention:
    def __init__(self, name):
        self.name = name
        self.original_processor = object()
        self.processor = self.original_processor

    def get_processor(self):
        return self.processor

    def set_processor(self, processor):
        self.processor = processor


class _Block:
    def __init__(self, attr_name, attn_name):
        setattr(self, attr_name, _Attention(attn_name))


class WanTinyTransformer:
    def __init__(self):
        self.blocks = [_Block("attn1", "wan_attn_0"), _Block("attn1", "wan_attn_1")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class SkyReelsV2Transformer3DModel(WanTinyTransformer):
    pass


class WanAnimateTransformer3DModel(WanTinyTransformer):
    pass


class WanVACETransformer3DModel(WanTinyTransformer):
    pass


class HunyuanVideoTinyTransformer:
    def __init__(self):
        self.transformer_blocks = [_Block("attn", "hunyuan_dual_0")]
        self.single_transformer_blocks = [_Block("attn", "hunyuan_single_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class CogVideoXTransformer3DModel:
    def __init__(self):
        self.transformer_blocks = [_Block("attn1", "cog_attn_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class LTXVideoTransformer3DModel:
    def __init__(self):
        self.transformer_blocks = [_Block("attn1", "ltx_attn_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class LTXVideo2Transformer3DModel(LTXVideoTransformer3DModel):
    pass


class MotifVideoTransformer3DModel:
    pass


class AllegroTransformer3DModel:
    def __init__(self):
        self.transformer_blocks = [_Block("attn1", "allegro_attn_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class EasyAnimateTransformer3DModel:
    def __init__(self):
        self.transformer_blocks = [_Block("attn1", "easyanimate_attn_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class _MochiAttention:
    def __init__(self, name):
        self.name = name
        self.original_processor = object()
        self.processor = self.original_processor


class _MochiBlock:
    def __init__(self, attn_name):
        self.attn1 = _MochiAttention(attn_name)


class MochiTransformer3DModel:
    def __init__(self):
        self.transformer_blocks = [_MochiBlock("mochi_attn_0")]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs is True
        handle = _Hook()
        self.hooks.append((hook, handle))
        return handle


class SanaVideoTransformer3DModel:
    pass


class Kandinsky5Transformer3DModel:
    pass


class _Scheduler:
    def set_timesteps(self, steps):
        self.timesteps = [1000 - 10 * idx for idx in range(steps)]


class _DynamicShiftScheduler:
    def set_timesteps(self, steps, **kwargs):
        if "mu" not in kwargs:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `True`")
        self.timesteps = [1000 - 10 * idx for idx in range(steps)]


class _Pipe:
    def __init__(self, transformer, scheduler=None):
        self.transformer = transformer
        if scheduler is not None:
            self.scheduler = scheduler


class _DiffSynthAttention(torch.nn.Module):
    def __init__(self, heads=2):
        super().__init__()
        self.num_heads = heads
        self.original_forward_calls = 0

    def forward(self, q, k, v):
        self.original_forward_calls += 1
        return q + k + v


class _DiffSynthSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _DiffSynthAttention(heads=2)


class _DiffSynthBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DiffSynthSelfAttention()


class _DiffSynthWanModel(torch.nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DiffSynthBlock() for _ in range(blocks)])


class WanS2VModel(_DiffSynthWanModel):
    __module__ = "diffsynth.models.wan_video_dit_s2v"

    def __init__(self, blocks=2):
        super().__init__(blocks=blocks)
        self.audio_injector = object()


class WanToDanceModel(_DiffSynthWanModel):
    __module__ = "diffsynth.models.wan_video_dit"

    def __init__(self, blocks=1):
        super().__init__(blocks=blocks)
        self.music_injector = object()
        self.music_encoder = object()


class _DiffSynthLongCatAttention(torch.nn.Module):
    def __init__(self, heads=2):
        super().__init__()
        self.num_heads = heads
        self.original_process_calls = 0

    def _process_attn(self, q, k, v, shape):
        self.original_process_calls += 1
        return q + k + v


class _DiffSynthLongCatBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _DiffSynthLongCatAttention(heads=2)
        self.cross_attn = SimpleNamespace(_process_cross_attn=lambda *args, **kwargs: None)


class LongCatVideoTransformer3DModel(torch.nn.Module):
    __module__ = "diffsynth.models.longcat_video_dit"

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DiffSynthLongCatBlock()])


class _DiffSynthVaceModel(torch.nn.Module):
    def __init__(self, blocks=1):
        super().__init__()
        self.vace_blocks = torch.nn.ModuleList([_DiffSynthBlock() for _ in range(blocks)])


class DiffSynthWanVideoPipeline:
    __module__ = "diffsynth.pipelines.wan_video"

    def __init__(self):
        self.dit = _DiffSynthWanModel(blocks=2)
        self.dit2 = None
        self.vace = None
        self.vace2 = None
        self.model_fn_calls = 0

    def model_fn(self, *, timestep=None, **kwargs):
        self.model_fn_calls += 1
        return timestep


class DiffSynthMovaAudioVideoPipeline:
    __module__ = "diffsynth.pipelines.mova_audio_video"

    def __init__(self):
        self.video_dit = _DiffSynthWanModel(blocks=1)
        self.video_dit2 = _DiffSynthWanModel(blocks=1)
        self.audio_dit = object()
        self.dual_tower_bridge = object()
        self.vace = _DiffSynthVaceModel(blocks=1)

    def model_fn(self, *, timestep=None, **kwargs):
        return timestep


class _DiffSynthLTX2Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.dim_head = 3
        self.rope_type = None
        self.to_q = torch.nn.Linear(6, 6, bias=False)
        self.to_k = torch.nn.Linear(6, 6, bias=False)
        self.to_v = torch.nn.Linear(6, 6, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.to_gate_logits = None
        self.to_out = torch.nn.Sequential(torch.nn.Linear(6, 6, bias=False), torch.nn.Identity())
        self.original_forward_calls = 0
        with torch.no_grad():
            self.to_q.weight.copy_(torch.eye(6))
            self.to_k.weight.copy_(torch.eye(6))
            self.to_v.weight.copy_(torch.eye(6))
            self.to_out[0].weight.copy_(torch.eye(6))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        pe=None,
        k_pe=None,
        perturbation_mask=None,
        all_perturbed=False,
    ):
        self.original_forward_calls += 1
        return x + 1


class _DiffSynthLTX2Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _DiffSynthLTX2Attention()
        self.attn2 = object()
        self.audio_attn1 = object()
        self.audio_attn2 = object()
        self.audio_to_video_attn = object()
        self.video_to_audio_attn = object()


class _DiffSynthLTX2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_DiffSynthLTX2Block()])


class DiffSynthLTX2AudioVideoPipeline:
    __module__ = "diffsynth.pipelines.ltx2_audio_video"

    def __init__(self):
        self.dit = _DiffSynthLTX2Model()
        self.model_fn_calls = 0

    def model_fn(self, *, timestep=None, **kwargs):
        self.model_fn_calls += 1
        return timestep


class DiffSynthLongCatVideoPipeline:
    __module__ = "diffsynth.pipelines.wan_video"

    def __init__(self):
        self.dit = LongCatVideoTransformer3DModel()


class DiffSynthWanS2VVideoPipeline:
    __module__ = "diffsynth.pipelines.wan_video"

    def __init__(self):
        self.dit = WanS2VModel(blocks=1)


class DiffSynthWanToDanceVideoPipeline:
    __module__ = "diffsynth.pipelines.wan_video"

    def __init__(self):
        self.dit = WanToDanceModel(blocks=1)


class DiffSynthFluxImagePipeline:
    __module__ = "diffsynth.pipelines.flux_image"

    def __init__(self):
        self.dit = _DiffSynthWanModel(blocks=1)


class CogVideoXImageToVideoPipeline(_Pipe):
    pass


NEW_BACKBONE_PROCESSOR_METHODS = [
    ("svg1", {}),
    ("svg2", {}),
    ("spargeattn", {}),
    ("radial", {}),
    ("sta", {}),
    ("draft", {}),
    ("adacluster", {}),
    ("flashomni", {"sparse_pattern": "paper_mmdit", "max_order": 0, "use_sparse_gemm": False}),
]


DIFFSYNTH_APPLY_METHODS = [
    ("dense", {}),
    ("svg1", {}),
    ("svg2", {}),
    ("spargeattn", {}),
    ("radial", {}),
    ("sta", {}),
    ("draft", {}),
    ("adacluster", {}),
    ("flashomni", {"sparse_pattern": "global_random"}),
    ("svoo", {}),
]


def _diffsynth_apply_config(method, config):
    if method == "dense":
        return {}
    merged = {
        "dense_warmup_step_ratio": 0,
        "dense_warmup_layer_ratio": 0,
    }
    merged.update(config)
    return merged


def _processors(transformer, attr):
    return [getattr(block, attr).get_processor() for block in transformer.blocks]


def test_apply_and_restore_wan_sparse_processor():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    installed = _processors(transformer, "attn1")
    assert all(isinstance(processor, SparseWanAttnProcessor) for processor in installed)
    assert installed != original
    assert len(transformer.hooks) == 1
    assert transformer.hooks[0][1].removed is False

    handle.restore()
    assert _processors(transformer, "attn1") == original
    assert transformer.hooks[0][1].removed is True

    handle.restore()
    assert _processors(transformer, "attn1") == original


def test_public_apply_alias_installs_and_restores_sparse_processor():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    handle = sparsevideo.apply(pipe, method="svoo")

    installed = _processors(transformer, "attn1")
    assert all(isinstance(processor, SparseWanAttnProcessor) for processor in installed)
    assert installed != original

    handle.restore()
    assert _processors(transformer, "attn1") == original


@pytest.mark.parametrize(
    "method",
    ["svg1", "svg2", "spargeattn", "radial", "sta", "draft", "adacluster", "flashomni", "svoo"],
)
def test_apply_wan22_dual_transformer_uses_local_layer_indices(method):
    pipe = _Pipe(WanTinyTransformer())
    pipe.transformer_2 = WanTinyTransformer()

    handle = sparsevideo.apply_sparse_attention(pipe, method=method)

    try:
        assert [processor.layer_idx for processor in _processors(pipe.transformer, "attn1")] == [0, 1]
        assert [processor.layer_idx for processor in _processors(pipe.transformer_2, "attn1")] == [0, 1]
    finally:
        handle.restore()


def test_replace_attention_rejects_sta_for_wan21_t2v_13b_with_red_error(capsys):
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    pipe._name_or_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    original = _processors(transformer, "attn1")

    with pytest.raises(NotImplementedError, match="temporarily unsupported for Wan2.1-T2V-1.3B"):
        sparsevideo.replace_attention(pipe, method="sta")

    stderr = capsys.readouterr().err
    assert "\033[31mError:\033[0m" in stderr
    assert "has not found suitable STA parameters that balance efficiency and quality" in stderr
    assert _processors(transformer, "attn1") == original


def test_reapplying_sparse_method_restores_previous_handle_to_dense_first():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    first = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    first_installed = _processors(transformer, "attn1")
    assert first_installed != original
    assert transformer.hooks[0][1].removed is False

    second = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    assert first.summary()["restored"] is True
    assert transformer.hooks[0][1].removed is True
    assert _processors(transformer, "attn1") != first_installed

    second.restore()
    assert _processors(transformer, "attn1") == original
    assert transformer.hooks[1][1].removed is True


def test_dense_apply_restores_active_sparse_method():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    sparse = sparsevideo.apply_sparse_attention(pipe, method="svoo")
    dense = sparsevideo.apply_sparse_attention(pipe, method="dense")

    assert sparse.summary()["restored"] is True
    assert dense.summary()["method_class"] == "DenseMethod"
    assert _processors(transformer, "attn1") == original


def test_sparse_attention_handle_summary_records_installed_processors_and_restore_state():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")
    summary = handle.summary()
    installed = _processors(transformer, "attn1")

    assert summary["model_type"] == "wan"
    assert summary["model_key"] is None
    assert summary["num_self_attn_layers"] == 2
    assert summary["installed_processor_count"] == 2
    assert summary["installed_processor_paths"] == [
        "transformer.blocks.0.attn1",
        "transformer.blocks.1.attn1",
    ]
    assert set(summary["current_processor_classes"].values()) == {
        "sparsevideo.processors.wan.SparseWanAttnProcessor"
    }
    assert summary["step_tracker_hook_count"] == 1
    assert summary["restore_callback_count"] == 1
    assert summary["method_class"] == "SVOOMethod"
    assert summary["method_runtime"] == {
        "total_calls": 0,
        "dispatch_counts": {},
        "backend_counts": {},
        "last_dispatch": None,
    }
    assert summary["restored"] is False

    query = torch.randn(1, 4, 2, 3)
    installed[0].attn_fn(query, query, query, None)
    runtime = handle.summary()["method_runtime"]
    assert runtime["total_calls"] == 1
    assert runtime["dispatch_counts"] == {"dense": 1}
    assert runtime["backend_counts"] == {"torch_sdpa": 1}
    assert runtime["last_dispatch"] == {
        "dispatch": "dense",
        "backend": "torch_sdpa",
        "layer_idx": 0,
        "step": 0,
    }

    handle.restore()
    restored = handle.summary()
    assert restored["restored"] is True
    assert set(restored["current_processor_classes"].values()) == {"builtins.object"}


def test_discover_model_enumerates_diffsynth_wan_self_attention_modules():
    pipe = DiffSynthWanVideoPipeline()

    model_info = discover_model(pipe)

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_type == "wan"
    assert model_info.num_self_attn_layers == 2
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "dit.blocks.0.self_attn.attn",
        "dit.blocks.1.self_attn.attn",
    ]


def test_discover_model_uses_explicit_diffsynth_model_key():
    pipe = DiffSynthWanVideoPipeline()
    pipe._sparsevideo_model_key = "wan22-ti2v-5b"

    model_info = discover_model(pipe)

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_key == "wan22-ti2v-5b"


@pytest.mark.parametrize(
    ("identity", "expected_key"),
    [
        ("DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", "wan21-speedcontrol-1.3b"),
        ("PAI/Wan2.1-Fun-1.3B-Control", "wan21-fun-1.3b-control"),
        ("PAI/Wan2.1-Fun-14B-InP", "wan21-fun-14b-inp"),
        ("PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", "wan21-fun-v11-1.3b-control-camera"),
        ("PAI/Wan2.2-Fun-A14B-Control-Camera", "wan22-fun-a14b-control-camera"),
        ("Wan-AI/Wan2.2-S2V-14B", "wan22-s2v-14b"),
        ("ByteDance/Video-As-Prompt-Wan2.1-14B", "video-as-prompt-wan21-14b"),
        ("krea/krea-realtime-video", "krea-realtime-video"),
    ],
)
def test_discover_model_infers_diffsynth_wan_family_model_keys(identity, expected_key):
    pipe = DiffSynthWanVideoPipeline()
    pipe.dit.model_id = identity

    model_info = discover_model(pipe)

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_key == expected_key


def test_discover_model_enumerates_diffsynth_wan_dual_dit_modules():
    pipe = DiffSynthWanVideoPipeline()
    pipe.dit2 = _DiffSynthWanModel(blocks=2)

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["num_self_attn_layers"] == 4
    assert summary["patched_attention_paths"] == [
        "dit.blocks.0.self_attn.attn",
        "dit.blocks.1.self_attn.attn",
        "dit2.blocks.0.self_attn.attn",
        "dit2.blocks.1.self_attn.attn",
    ]

    handle.restore()
    assert handle.summary()["restored"] is True


def test_discover_model_infers_diffsynth_vace_from_auxiliary_module_only():
    pipe = DiffSynthWanVideoPipeline()
    pipe.dit = None
    pipe.vace = _DiffSynthVaceModel(blocks=1)

    model_info = discover_model(pipe)

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_key == "wan21-vace-1.3b"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "vace.vace_blocks.0.self_attn.attn",
    ]


def test_apply_and_restore_diffsynth_wan_vace_attention_modules():
    pipe = DiffSynthWanVideoPipeline()
    pipe.vace = _DiffSynthVaceModel(blocks=2)
    dit_attn = pipe.dit.blocks[0].self_attn.attn
    vace_attn = pipe.vace.vace_blocks[0].self_attn.attn
    q = torch.randn(1, 4, 4)

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svg2",
        config={"dense_warmup_step_ratio": 1.0},
    )
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["patched_attention_paths"] == [
        "dit.blocks.0.self_attn.attn",
        "dit.blocks.1.self_attn.attn",
        "vace.vace_blocks.0.self_attn.attn",
        "vace.vace_blocks.1.self_attn.attn",
    ]

    dit_out = dit_attn(q, q, q)
    vace_out = vace_attn(q, q, q)

    assert dit_out.shape == q.shape
    assert vace_out.shape == q.shape
    assert dit_attn.original_forward_calls == 0
    assert vace_attn.original_forward_calls == 0

    handle.restore()
    restored_dit = dit_attn(q, q, q)
    restored_vace = vace_attn(q, q, q)

    assert torch.allclose(restored_dit, q + q + q)
    assert torch.allclose(restored_vace, q + q + q)
    assert dit_attn.original_forward_calls == 1
    assert vace_attn.original_forward_calls == 1
    assert handle.summary()["restored"] is True


def test_diffsynth_summary_reports_unpatched_auxiliary_attention_paths():
    pipe = DiffSynthWanVideoPipeline()
    pipe.vap = object()
    pipe.animate_adapter = object()

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["patched_attention_paths"] == [
        "dit.blocks.0.self_attn.attn",
        "dit.blocks.1.self_attn.attn",
    ]
    assert summary["unpatched_attention_paths"] == [
        "animate_adapter.scaled_dot_product_attention",
        "vap.MotWanAttentionBlock.flash_attention",
    ]
    assert any("VAP/MotWanModel" in note for note in summary["pipeline_notes"])
    assert any("Animate adapter" in note for note in summary["pipeline_notes"])

    handle.restore()


def test_discover_model_reports_diffsynth_s2v_audio_injector_as_unpatched():
    model_info = discover_model(DiffSynthWanS2VVideoPipeline())

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.unpatched_attention_paths == [
        "dit.audio_injector.injector.*.attn",
    ]
    assert any("S2V audio-injector" in note for note in model_info.pipeline_notes)


def test_discover_model_reports_diffsynth_wantodance_music_attention_as_unpatched():
    model_info = discover_model(DiffSynthWanToDanceVideoPipeline())

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.unpatched_attention_paths == [
        "dit.music_injector.injector.*.attn",
        "dit.music_encoder.*.self_attn",
    ]
    assert any("WanToDance music-injector" in note for note in model_info.pipeline_notes)
    assert any("WanToDance music-encoder" in note for note in model_info.pipeline_notes)


def test_discover_model_enumerates_diffsynth_mova_video_and_vace_modules():
    pipe = DiffSynthMovaAudioVideoPipeline()

    model_info = discover_model(pipe)

    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_type == "wan"
    assert model_info.model_key == "mova-720p"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "video_dit.blocks.0.self_attn.attn",
        "video_dit2.blocks.0.self_attn.attn",
        "vace.vace_blocks.0.self_attn.attn",
    ]
    assert model_info.unpatched_attention_paths == [
        "audio_dit.blocks.*.self_attn.attn",
        "dual_tower_bridge.*.attn",
    ]
    assert any("MOVA audio DiT" in note for note in model_info.pipeline_notes)
    assert any("MOVA dual-tower bridge" in note for note in model_info.pipeline_notes)


def test_apply_and_restore_diffsynth_mova_video_attention_patch():
    pipe = DiffSynthMovaAudioVideoPipeline()
    attn = pipe.video_dit.blocks[0].self_attn.attn
    q = torch.randn(1, 4, 4)

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svg2",
        config={"dense_warmup_step_ratio": 1.0},
    )
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["model_key"] == "mova-720p"
    assert summary["patched_attention_paths"] == [
        "vace.vace_blocks.0.self_attn.attn",
        "video_dit.blocks.0.self_attn.attn",
        "video_dit2.blocks.0.self_attn.attn",
    ]
    assert summary["unpatched_attention_paths"] == [
        "audio_dit.blocks.*.self_attn.attn",
        "dual_tower_bridge.*.attn",
    ]

    out = attn(q, q, q)

    assert out.shape == q.shape
    assert attn.original_forward_calls == 0

    handle.restore()
    restored_out = attn(q, q, q)

    assert torch.allclose(restored_out, q + q + q)
    assert attn.original_forward_calls == 1


def test_apply_and_restore_diffsynth_wan_forward_patch_tracks_timestep():
    pipe = DiffSynthWanVideoPipeline()
    attn = pipe.dit.blocks[0].self_attn.attn
    q = torch.randn(1, 4, 4)

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svg2",
        config={"dense_warmup_step_ratio": 1.0},
    )
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["installed_processor_count"] == 0
    assert summary["patched_attention_count"] == 2
    assert summary["patched_attention_paths"] == [
        "dit.blocks.0.self_attn.attn",
        "dit.blocks.1.self_attn.attn",
    ]

    pipe.model_fn(timestep=torch.tensor([987.0]))
    out = attn(q, q, q)

    assert out.shape == q.shape
    assert attn.original_forward_calls == 0
    runtime = handle.summary()["method_runtime"]
    assert runtime["total_calls"] == 1
    assert runtime["dispatch_counts"] == {"dense": 1}
    assert runtime["last_dispatch"]["step"] == 1

    handle.restore()
    restored_out = attn(q, q, q)

    assert attn.original_forward_calls == 1
    assert torch.allclose(restored_out, q + q + q)
    assert handle.summary()["restored"] is True


def test_diffsynth_wan_forward_patch_rejects_invalid_qkv_shape():
    pipe = DiffSynthWanVideoPipeline()
    attn = pipe.dit.blocks[0].self_attn.attn
    q = torch.randn(1, 4, 5)

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    try:
        with pytest.raises(RuntimeError, match="cannot split q channels=5 over num_heads=2"):
            attn(q, q, q)
    finally:
        handle.restore()


def test_reapplying_diffsynth_sparse_method_restores_previous_forward_patch():
    pipe = DiffSynthWanVideoPipeline()
    attn = pipe.dit.blocks[0].self_attn.attn
    q = torch.randn(1, 4, 4)

    first = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    first_forward = attn.forward
    second = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    assert first.summary()["restored"] is True
    assert second.summary()["restored"] is False
    assert attn.forward is not first_forward
    assert second.summary()["patched_attention_count"] == 2

    second.restore()
    restored_out = attn(q, q, q)

    assert torch.allclose(restored_out, q + q + q)
    assert attn.original_forward_calls == 1


def test_dense_apply_restores_active_diffsynth_sparse_forward_patch():
    pipe = DiffSynthWanVideoPipeline()
    attn = pipe.dit.blocks[0].self_attn.attn
    q = torch.randn(1, 4, 4)

    sparse = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    dense = sparsevideo.apply_sparse_attention(pipe, method="dense")

    assert sparse.summary()["restored"] is True
    assert dense.summary()["method_class"] == "DenseMethod"
    restored_out = attn(q, q, q)
    assert torch.allclose(restored_out, q + q + q)
    assert attn.original_forward_calls == 1


def test_apply_diffsynth_invokes_backend_safe_model_patch_callbacks(monkeypatch):
    from sparsevideo.methods.svg2.method import SVG2Method

    pipe = DiffSynthWanVideoPipeline()
    calls = []
    restored = []

    def install_model_patches(self, model_info):
        calls.append(model_info.pipeline_backend)
        return [lambda: restored.append(True)]

    monkeypatch.setattr(SVG2Method, "install_model_patches", install_model_patches)

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    assert calls == ["diffsynth"]
    assert handle.summary()["restore_callback_count"] == 4

    handle.restore()
    assert restored == [True]
    assert handle.summary()["restored"] is True


def test_apply_diffsynth_rejects_unified_sequence_parallel_patch_path():
    pipe = DiffSynthWanVideoPipeline()
    pipe.use_unified_sequence_parallel = True

    with pytest.raises(NotImplementedError, match="unified sequence parallel"):
        sparsevideo.apply_sparse_attention(pipe, method="svg2")


def test_apply_and_restore_diffsynth_ltx2_video_self_attention_patch():
    pipe = DiffSynthLTX2AudioVideoPipeline()
    attn = pipe.dit.transformer_blocks[0].attn1
    hidden_states = torch.randn(1, 4, 6)

    model_info = discover_model(pipe)
    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_type == "ltx_video"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "dit.transformer_blocks.0.attn1",
    ]
    assert "dit.transformer_blocks.*.attn2" in model_info.unpatched_attention_paths
    assert "dit.transformer_blocks.*.audio_attn1" in model_info.unpatched_attention_paths
    assert "dit.transformer_blocks.*.audio_to_video_attn" in model_info.unpatched_attention_paths

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svg2",
        config={"dense_warmup_step_ratio": 1.0},
    )
    summary = handle.summary()

    assert summary["model_type"] == "ltx_video"
    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["patched_attention_paths"] == ["dit.transformer_blocks.0.attn1"]
    assert "dit.transformer_blocks.*.attn2" in summary["unpatched_attention_paths"]

    out = attn(hidden_states)
    masked_out = attn(hidden_states, mask=torch.zeros(1, 1, 4, 4))

    assert out.shape == hidden_states.shape
    assert masked_out.shape == hidden_states.shape
    assert attn.original_forward_calls == 0
    runtime = handle.summary()["method_runtime"]
    assert runtime["total_calls"] == 2
    assert runtime["dispatch_counts"] == {"dense": 2}
    assert runtime["backend_counts"]["diffusers_dispatch"] == 1

    handle.restore()
    restored_out = attn(hidden_states)

    assert attn.original_forward_calls == 1
    assert torch.allclose(restored_out, hidden_states + 1)
    assert handle.summary()["restored"] is True


def test_diffsynth_ltx2_attention_patch_passes_mask_to_sparsevideo_method():
    attn = _DiffSynthLTX2Attention()
    hidden_states = torch.randn(1, 4, 6)
    mask = torch.zeros(1, 1, 4, 4)
    seen = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        seen["query_shape"] = tuple(query.shape)
        seen["mask"] = attention_mask
        seen["kwargs"] = kwargs
        return query

    restore = _patch_diffsynth_ltx2_attention_forward(
        attn,
        attn_fn,
        StepTracker(model_type="ltx_video"),
        "dit.transformer_blocks.0.attn1",
    )

    out = attn(hidden_states, mask=mask)

    assert out.shape == hidden_states.shape
    assert seen["query_shape"] == (1, 4, 2, 3)
    assert seen["mask"] is mask
    assert seen["kwargs"]["pipeline_backend"] == "diffsynth"
    assert seen["kwargs"]["cache_key_suffix"] == "dit.transformer_blocks.0.attn1"
    assert attn.original_forward_calls == 0

    restore()
    restored_out = attn(hidden_states, mask=mask)

    assert attn.original_forward_calls == 1
    assert torch.allclose(restored_out, hidden_states + 1)


def test_diffsynth_ltx2_attention_patch_preserves_rotary_embedding_path(monkeypatch):
    calls = []

    def fake_apply_rotary_emb(tensor, pe, rope_type):
        calls.append((tensor.shape, pe, rope_type))
        return tensor + 1

    monkeypatch.setitem(globals(), "apply_rotary_emb", fake_apply_rotary_emb)
    attn = _DiffSynthLTX2Attention()
    hidden_states = torch.randn(1, 4, 6)
    pe = object()
    k_pe = object()
    seen = {}

    def attn_fn(query, key, value, attention_mask, **kwargs):
        seen["query"] = query
        seen["key"] = key
        return query

    restore = _patch_diffsynth_ltx2_attention_forward(
        attn,
        attn_fn,
        StepTracker(model_type="ltx_video"),
        "dit.transformer_blocks.0.attn1",
    )

    out = attn(hidden_states, pe=pe, k_pe=k_pe)

    assert out.shape == hidden_states.shape
    assert len(calls) == 2
    assert calls[0] == (torch.Size([1, 4, 6]), pe, None)
    assert calls[1] == (torch.Size([1, 4, 6]), k_pe, None)
    assert torch.allclose(seen["query"], hidden_states.add(1).unflatten(-1, (2, 3)))
    assert torch.allclose(seen["key"], hidden_states.add(1).unflatten(-1, (2, 3)))

    restore()
    restored_out = attn(hidden_states, pe=pe, k_pe=k_pe)

    assert attn.original_forward_calls == 1
    assert torch.allclose(restored_out, hidden_states + 1)


def test_diffsynth_ltx2_attention_patch_applies_gate_logits():
    class GateLogits(torch.nn.Module):
        def forward(self, x):
            logits = torch.tensor([-1.0, 1.0], dtype=x.dtype, device=x.device)
            return logits.view(1, 1, 2).expand(x.shape[0], x.shape[1], 2)

    attn = _DiffSynthLTX2Attention()
    attn.to_gate_logits = GateLogits()
    hidden_states = torch.randn(1, 4, 6)

    def attn_fn(query, key, value, attention_mask, **kwargs):
        return query

    restore = _patch_diffsynth_ltx2_attention_forward(
        attn,
        attn_fn,
        StepTracker(model_type="ltx_video"),
        "dit.transformer_blocks.0.attn1",
    )

    out = attn(hidden_states)
    gates = 2.0 * torch.sigmoid(torch.tensor([-1.0, 1.0], dtype=hidden_states.dtype))
    expected = hidden_states.view(1, 4, 2, 3).mul(gates.view(1, 1, 2, 1)).view(1, 4, 6)

    assert torch.allclose(out, expected)
    assert attn.original_forward_calls == 0

    restore()
    restored_out = attn(hidden_states)
    assert attn.original_forward_calls == 1
    assert torch.allclose(restored_out, hidden_states + 1)


def test_apply_and_restore_diffsynth_longcat_process_attn_patch():
    pipe = DiffSynthLongCatVideoPipeline()
    attn = pipe.dit.blocks[0].attn
    q = torch.randn(1, 2, 4, 3)

    model_info = discover_model(pipe)
    assert model_info.pipeline_backend == "diffsynth"
    assert model_info.model_type == "wan"
    assert model_info.model_key == "longcat-video"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "dit.blocks.0.attn._process_attn",
    ]
    assert model_info.unpatched_attention_paths == [
        "dit.blocks.*.cross_attn._process_cross_attn",
    ]

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svg2",
        config={"dense_warmup_step_ratio": 1.0},
    )
    summary = handle.summary()

    assert summary["patched_attention_paths"] == ["dit.blocks.0.attn._process_attn"]

    out = attn._process_attn(q, q, q, (1, 2, 2))

    assert out.shape == q.shape
    assert attn.original_process_calls == 0
    runtime = handle.summary()["method_runtime"]
    assert runtime["total_calls"] == 1
    assert runtime["dispatch_counts"] == {"dense": 1}

    handle.restore()
    restored_out = attn._process_attn(q, q, q, (1, 2, 2))

    assert attn.original_process_calls == 1
    assert torch.allclose(restored_out, q + q + q)
    assert handle.summary()["restored"] is True


def test_diffsynth_longcat_process_patch_rejects_invalid_heads_shape():
    pipe = DiffSynthLongCatVideoPipeline()
    attn = pipe.dit.blocks[0].attn
    q = torch.randn(1, 3, 4, 3)

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")
    try:
        with pytest.raises(RuntimeError, match="expected q heads=2, got 3"):
            attn._process_attn(q, q, q, shape=None)
    finally:
        handle.restore()


@pytest.mark.parametrize(
    ("pipe_factory", "expected_paths", "expected_unpatched_paths"),
    [
        pytest.param(
            DiffSynthWanVideoPipeline,
            [
                "dit.blocks.0.self_attn.attn",
                "dit.blocks.1.self_attn.attn",
            ],
            [],
            id="wan",
        ),
        pytest.param(
            DiffSynthMovaAudioVideoPipeline,
            [
                "vace.vace_blocks.0.self_attn.attn",
                "video_dit.blocks.0.self_attn.attn",
                "video_dit2.blocks.0.self_attn.attn",
            ],
            [
                "audio_dit.blocks.*.self_attn.attn",
                "dual_tower_bridge.*.attn",
            ],
            id="mova",
        ),
        pytest.param(
            DiffSynthLTX2AudioVideoPipeline,
            ["dit.transformer_blocks.0.attn1"],
            [
                "dit.transformer_blocks.*.attn2",
                "dit.transformer_blocks.*.audio_attn1",
                "dit.transformer_blocks.*.audio_attn2",
                "dit.transformer_blocks.*.audio_to_video_attn",
                "dit.transformer_blocks.*.video_to_audio_attn",
            ],
            id="ltx2",
        ),
        pytest.param(
            DiffSynthLongCatVideoPipeline,
            ["dit.blocks.0.attn._process_attn"],
            ["dit.blocks.*.cross_attn._process_cross_attn"],
            id="longcat",
        ),
        pytest.param(
            DiffSynthWanS2VVideoPipeline,
            ["dit.blocks.0.self_attn.attn"],
            ["dit.audio_injector.injector.*.attn"],
            id="s2v",
        ),
        pytest.param(
            DiffSynthWanToDanceVideoPipeline,
            ["dit.blocks.0.self_attn.attn"],
            ["dit.music_encoder.*.self_attn", "dit.music_injector.injector.*.attn"],
            id="wantodance",
        ),
    ],
)
@pytest.mark.parametrize(("method", "config"), DIFFSYNTH_APPLY_METHODS)
def test_apply_and_restore_diffsynth_public_methods_on_supported_pipeline_shapes(
    pipe_factory,
    expected_paths,
    expected_unpatched_paths,
    method,
    config,
):
    pipe = pipe_factory()

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method=method,
        config=_diffsynth_apply_config(method, config),
    )
    summary = handle.summary()

    assert summary["pipeline_backend"] == "diffsynth"
    assert summary["num_self_attn_layers"] == len(expected_paths)
    assert summary["unpatched_attention_paths"] == expected_unpatched_paths
    if method == "dense":
        assert summary["patched_attention_paths"] == []
        assert summary["patched_attention_count"] == 0
    else:
        assert summary["patched_attention_paths"] == expected_paths
        assert summary["patched_attention_count"] == len(expected_paths)

    handle.restore()
    assert handle.summary()["restored"] is True


def test_diffsynth_apply_method_matrix_tracks_public_methods():
    assert {method for method, _ in DIFFSYNTH_APPLY_METHODS} == set(sparsevideo.list_methods())


def test_discover_model_does_not_treat_diffsynth_image_dit_as_wan_video():
    with pytest.raises(ValueError, match="Pipeline has no \\.transformer"):
        discover_model(DiffSynthFluxImagePipeline())


def test_runtime_summary_records_dense_dispatch_for_cpu_dense_gate_methods():
    query = torch.randn(1, 4, 2, 3)
    cases = [
        ("svg1", {}, "SVG1Method", "torch_sdpa"),
        ("svg2", {}, "SVG2Method", "torch_sdpa"),
        ("svoo", {}, "SVOOMethod", "torch_sdpa"),
        ("draft", {}, "DraftMethod", "torch_sdpa"),
        ("spargeattn", {"dense_warmup_step_ratio": 1.0}, "SpargeAttnMethod", "diffusers_dispatch"),
    ]

    for method, config, method_class, backend in cases:
        transformer = WanTinyTransformer()
        handle = sparsevideo.apply_sparse_attention(_Pipe(transformer), method=method, config=config)
        try:
            processor = _processors(transformer, "attn1")[0]
            processor.attn_fn(query, query, query, None)
            runtime = handle.summary()["method_runtime"]

            assert handle.summary()["method_class"] == method_class
            assert runtime["total_calls"] == 1
            assert runtime["dispatch_counts"] == {"dense": 1}
            assert runtime["backend_counts"] == {backend: 1}
            assert runtime["last_dispatch"]["dispatch"] == "dense"
            assert runtime["last_dispatch"]["backend"] == backend
        finally:
            handle.restore()


def test_runtime_summary_records_adacluster_hunyuan_dense_gate():
    query = torch.randn(1, 4, 2, 3)
    transformer = HunyuanVideoTinyTransformer()
    handle = sparsevideo.apply_sparse_attention(_Pipe(transformer), method="adacluster")
    try:
        processor = transformer.transformer_blocks[0].attn.get_processor()
        processor.attn_fn(query, query, query, None)
        runtime = handle.summary()["method_runtime"]

        assert handle.summary()["method_class"] == "AdaClusterMethod"
        assert runtime["total_calls"] == 1
        assert runtime["dispatch_counts"] == {"dense": 1}
        assert runtime["backend_counts"] == {"torch_sdpa": 1}
    finally:
        handle.restore()


def test_public_restore_sparse_attention_function_restores_handle():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")
    assert _processors(transformer, "attn1") != original

    sparsevideo.restore_sparse_attention(handle)
    assert _processors(transformer, "attn1") == original

    sparsevideo.restore_sparse_attention(handle)
    assert _processors(transformer, "attn1") == original


def test_replace_attention_returns_pipe_and_pipe_restore_uses_active_handle():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    returned = sparsevideo.replace_attention(pipe, method="svoo")

    assert returned is pipe
    assert _processors(transformer, "attn1") != original

    sparsevideo.restore_sparse_attention(pipe)
    assert _processors(transformer, "attn1") == original


def test_apply_sparse_attention_rejects_legacy_svg_warmup_keys():
    for method in ("svg1", "svg2", "svoo"):
        transformer = WanTinyTransformer()
        pipe = _Pipe(transformer, scheduler=_Scheduler())

        with pytest.raises(ValueError, match="Unknown config keys"):
            sparsevideo.apply_sparse_attention(
                pipe,
                method=method,
                config={"first_times_fp": 0.2, "first_layers_fp": 0.03},
            )


def test_replace_attention_auto_sets_cuda_arch_list(monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index: (8, 0) if index == 0 else (9, 0),
    )

    sparsevideo.replace_attention(_Pipe(WanTinyTransformer()), method="dense")

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0;9.0"


def test_replace_attention_preserves_user_cuda_arch_list(monkeypatch):
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))

    sparsevideo.replace_attention(_Pipe(WanTinyTransformer()), method="dense")

    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0"


def test_apply_and_restore_svoo_restores_wan_fast_block_patch():
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock

    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original_forward = WanTransformerBlock.forward
    original_model_forward = WanTransformer3DModel.forward

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    assert WanTransformerBlock.forward is not original_forward
    assert WanTransformer3DModel.forward is not original_model_forward

    handle.restore()
    assert WanTransformerBlock.forward is original_forward
    assert WanTransformer3DModel.forward is original_model_forward


def test_apply_and_restore_svoo_restores_hunyuan_sparse_forward_patch():
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTransformer3DModel,
        HunyuanVideoTransformerBlock,
    )

    transformer = HunyuanVideoTinyTransformer()
    pipe = _Pipe(transformer)
    original_single_forward = HunyuanVideoSingleTransformerBlock.forward
    original_block_forward = HunyuanVideoTransformerBlock.forward
    original_model_forward = HunyuanVideoTransformer3DModel.forward

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    assert HunyuanVideoSingleTransformerBlock.forward is not original_single_forward
    assert HunyuanVideoTransformerBlock.forward is not original_block_forward
    assert HunyuanVideoTransformer3DModel.forward is not original_model_forward

    handle.restore()
    assert HunyuanVideoSingleTransformerBlock.forward is original_single_forward
    assert HunyuanVideoTransformerBlock.forward is original_block_forward
    assert HunyuanVideoTransformer3DModel.forward is original_model_forward


def test_apply_and_restore_spargeattn_restores_hunyuan_forward_patch(monkeypatch):
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    monkeypatch.setattr(
        "sparsevideo.methods.spargeattn.method._load_spas_sage_attn_functions",
        lambda: (lambda *args, **kwargs: args[0], lambda *args, **kwargs: args[0]),
    )
    transformer = HunyuanVideoTinyTransformer()
    pipe = _Pipe(transformer)
    original_model_forward = HunyuanVideoTransformer3DModel.forward

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="spargeattn",
        config={"dense_warmup_step_ratio": 1.0},
    )

    assert HunyuanVideoTransformer3DModel.forward is not original_model_forward
    assert handle.summary()["restore_callback_count"] == 2

    handle.restore()
    assert HunyuanVideoTransformer3DModel.forward is original_model_forward


def test_apply_and_restore_flashomni_restores_hunyuan_forward_taylor_patch():
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
        HunyuanVideoTransformer3DModel,
        HunyuanVideoTransformerBlock,
    )

    transformer = HunyuanVideoTinyTransformer()
    pipe = _Pipe(transformer)
    original_single_forward = HunyuanVideoSingleTransformerBlock.forward
    original_block_forward = HunyuanVideoTransformerBlock.forward
    original_model_forward = HunyuanVideoTransformer3DModel.forward

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="flashomni",
        config={"sparse_pattern": "paper_mmdit"},
    )

    assert HunyuanVideoSingleTransformerBlock.forward is not original_single_forward
    assert HunyuanVideoTransformerBlock.forward is not original_block_forward
    assert HunyuanVideoTransformer3DModel.forward is not original_model_forward
    assert handle.summary()["restore_callback_count"] == 1

    handle.restore()
    assert HunyuanVideoSingleTransformerBlock.forward is original_single_forward
    assert HunyuanVideoTransformerBlock.forward is original_block_forward
    assert HunyuanVideoTransformer3DModel.forward is original_model_forward


def test_dense_baseline_is_noop_and_does_not_install_hooks():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer)
    original = _processors(transformer, "attn1")

    handle = sparsevideo.apply_sparse_attention(pipe, method="dense")

    assert _processors(transformer, "attn1") == original
    assert transformer.hooks == []

    handle.restore()
    assert _processors(transformer, "attn1") == original
    assert transformer.hooks == []
    assert handle.summary() == {
        "pipeline_backend": "diffusers",
        "diffsynth_version": None,
        "model_type": "wan",
        "model_key": None,
        "num_self_attn_layers": 2,
        "installed_processor_count": 0,
        "installed_processor_paths": [],
        "patched_attention_count": 0,
        "patched_attention_paths": [],
        "unpatched_attention_paths": [],
        "pipeline_notes": [],
        "current_processor_classes": {},
        "step_tracker_hook_count": 0,
        "restore_callback_count": 0,
        "method_class": "DenseMethod",
        "method_config": {},
        "method_runtime": {
            "total_calls": 0,
            "dispatch_counts": {},
            "backend_counts": {},
            "last_dispatch": None,
        },
        "restored": True,
    }


def test_discover_model_infers_wan14b_model_key_from_config():
    transformer = WanTinyTransformer()
    transformer.config = SimpleNamespace(_name_or_path="/models/Wan2.1-T2V-14B-Diffusers")
    pipe = _Pipe(transformer)

    model_info = discover_model(pipe)

    assert model_info.model_type == "wan"
    assert model_info.model_key == "wan21-t2v-14b"


def test_discover_model_infers_wan22_model_key_from_two_transformers():
    pipe = _Pipe(WanTinyTransformer())
    pipe.transformer_2 = WanTinyTransformer()

    model_info = discover_model(pipe)

    assert model_info.model_type == "wan"
    assert model_info.model_key == "wan22-t2v-a14b"


def test_discover_model_treats_skyreels_v2_as_wan_family():
    transformer = SkyReelsV2Transformer3DModel()
    transformer.config = SimpleNamespace(
        _name_or_path="/models/SkyReels-V2-T2V-14B-720P-Diffusers"
    )
    pipe = _Pipe(transformer)

    model_info = discover_model(pipe)

    assert model_info.model_type == "wan"
    assert model_info.model_key == "skyreels-v2-t2v-14b"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.blocks.0.attn1",
        "transformer.blocks.1.attn1",
    ]


def test_discover_model_uses_hunyuan_t2v_model_key():
    model_info = discover_model(_Pipe(HunyuanVideoTinyTransformer()))

    assert model_info.model_type == "hunyuan_video"
    assert model_info.model_key == "hunyuan-t2v"


@pytest.mark.parametrize(
    ("transformer_cls", "name_or_path", "expected_key"),
    [
        (
            WanAnimateTransformer3DModel,
            "/models/Wan2.2-Animate-14B-Diffusers",
            "wan22-animate-14b",
        ),
        (
            WanVACETransformer3DModel,
            "/models/Wan2.1-VACE-1.3B-diffusers",
            "wan21-vace-1.3b",
        ),
        (
            WanVACETransformer3DModel,
            "/models/Wan2.1-VACE-14B-diffusers",
            "wan21-vace-14b",
        ),
    ],
)
def test_discover_model_infers_tier1_wan_family_model_keys(transformer_cls, name_or_path, expected_key):
    transformer = transformer_cls()
    transformer.config = SimpleNamespace(_name_or_path=name_or_path)
    pipe = _Pipe(transformer)

    model_info = discover_model(pipe)

    assert model_info.model_type == "wan"
    assert model_info.model_key == expected_key
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.blocks.0.attn1",
        "transformer.blocks.1.attn1",
    ]


def test_discover_model_enumerates_real_tier1_wan_family_diffusers_classes():
    diffusers = pytest.importorskip("diffusers")

    cases = [
        (
            diffusers.WanAnimateTransformer3DModel,
            {
                "num_layers": 1,
                "num_attention_heads": 1,
                "attention_head_dim": 4,
                "ffn_dim": 16,
                "text_dim": 8,
                "freq_dim": 8,
                "image_dim": None,
            },
        ),
        (
            diffusers.WanVACETransformer3DModel,
            {
                "num_layers": 1,
                "num_attention_heads": 1,
                "attention_head_dim": 4,
                "ffn_dim": 16,
                "text_dim": 8,
                "freq_dim": 8,
                "image_dim": None,
                "vace_layers": [0],
            },
        ),
        (
            diffusers.SkyReelsV2Transformer3DModel,
            {
                "num_layers": 1,
                "num_attention_heads": 1,
                "attention_head_dim": 4,
                "ffn_dim": 16,
                "text_dim": 8,
                "freq_dim": 8,
                "image_dim": None,
            },
        ),
    ]

    for cls, kwargs in cases:
        model_info = discover_model(_Pipe(cls(**kwargs)))

        assert model_info.model_type == "wan"
        assert [path for path, _ in model_info.iter_self_attn_modules()] == [
            "transformer.blocks.0.attn1",
        ]


def test_apply_and_restore_hunyuan_sparse_processor():
    transformer = HunyuanVideoTinyTransformer()
    pipe = _Pipe(transformer)
    original_dual = [block.attn.get_processor() for block in transformer.transformer_blocks]
    original_single = [block.attn.get_processor() for block in transformer.single_transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method="svoo")

    installed = [
        block.attn.get_processor()
        for block in transformer.transformer_blocks + transformer.single_transformer_blocks
    ]
    assert all(isinstance(processor, SparseHunyuanVideoAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1

    handle.restore()
    assert [block.attn.get_processor() for block in transformer.transformer_blocks] == original_dual
    assert [block.attn.get_processor() for block in transformer.single_transformer_blocks] == original_single
    assert transformer.hooks[0][1].removed is True


@pytest.mark.parametrize(("method", "config"), NEW_BACKBONE_PROCESSOR_METHODS)
def test_apply_and_restore_cogvideox_sparse_processor(method, config):
    transformer = CogVideoXTransformer3DModel()
    pipe = _Pipe(transformer)
    original = [block.attn1.get_processor() for block in transformer.transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method=method, config=config)

    installed = [block.attn1.get_processor() for block in transformer.transformer_blocks]
    assert all(isinstance(processor, SparseCogVideoXAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1
    assert handle.summary()["model_type"] == "cogvideox"
    assert handle.summary()["model_key"] == "cogvideox-t2v"

    handle.restore()
    assert [block.attn1.get_processor() for block in transformer.transformer_blocks] == original
    assert transformer.hooks[0][1].removed is True


def test_discover_model_infers_cogvideox_i2v_model_key():
    transformer = CogVideoXTransformer3DModel()
    pipe = CogVideoXImageToVideoPipeline(transformer)

    model_info = discover_model(pipe)

    assert model_info.model_type == "cogvideox"
    assert model_info.model_key == "cogvideox-i2v"


@pytest.mark.parametrize(("method", "config"), NEW_BACKBONE_PROCESSOR_METHODS)
def test_apply_and_restore_ltx_sparse_processor(method, config):
    transformer = LTXVideoTransformer3DModel()
    pipe = _Pipe(transformer)
    original = [block.attn1.get_processor() for block in transformer.transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method=method, config=config)

    installed = [block.attn1.get_processor() for block in transformer.transformer_blocks]
    assert all(isinstance(processor, SparseLTXVideoAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1
    assert handle.summary()["model_type"] == "ltx_video"
    assert handle.summary()["model_key"] == "ltx-video"

    handle.restore()
    assert [block.attn1.get_processor() for block in transformer.transformer_blocks] == original
    assert transformer.hooks[0][1].removed is True


def test_discover_model_enumerates_real_ltx_video_diffusers_class():
    from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel as DiffusersLTX

    transformer = DiffusersLTX(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=8,
        out_channels=8,
        cross_attention_dim=16,
        caption_channels=16,
    )
    model_info = discover_model(_Pipe(transformer))

    assert model_info.model_type == "ltx_video"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.transformer_blocks.0.attn1",
    ]


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (
            MotifVideoTransformer3DModel(),
            "MotifVideo is not available in this Diffusers installation",
        ),
        (
            LTXVideo2Transformer3DModel(),
            "LTX Video 2 is not available in this Diffusers installation",
        ),
    ],
)
def test_discover_model_rejects_unknown_future_backbones_before_generic_matching(transformer, message):
    with pytest.raises(ValueError, match=message):
        discover_model(_Pipe(transformer))


@pytest.mark.parametrize(("method", "config"), NEW_BACKBONE_PROCESSOR_METHODS)
def test_apply_and_restore_allegro_sparse_processor(method, config):
    transformer = AllegroTransformer3DModel()
    pipe = _Pipe(transformer)
    original = [block.attn1.get_processor() for block in transformer.transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method=method, config=config)

    installed = [block.attn1.get_processor() for block in transformer.transformer_blocks]
    assert all(isinstance(processor, SparseAllegroAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1
    assert handle.summary()["model_type"] == "allegro"
    assert handle.summary()["model_key"] == "allegro"

    handle.restore()
    assert [block.attn1.get_processor() for block in transformer.transformer_blocks] == original
    assert transformer.hooks[0][1].removed is True


def test_discover_model_enumerates_real_allegro_diffusers_class():
    from diffusers.models.transformers.transformer_allegro import (
        AllegroTransformer3DModel as DiffusersAllegro,
    )

    transformer = DiffusersAllegro(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        cross_attention_dim=16,
        caption_channels=16,
        sample_height=4,
        sample_width=4,
        sample_frames=2,
    )
    model_info = discover_model(_Pipe(transformer))

    assert model_info.model_type == "allegro"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.transformer_blocks.0.attn1",
    ]


@pytest.mark.parametrize(("method", "config"), NEW_BACKBONE_PROCESSOR_METHODS)
def test_apply_and_restore_easyanimate_sparse_processor(method, config):
    transformer = EasyAnimateTransformer3DModel()
    pipe = _Pipe(transformer)
    original = [block.attn1.get_processor() for block in transformer.transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method=method, config=config)

    installed = [block.attn1.get_processor() for block in transformer.transformer_blocks]
    assert all(isinstance(processor, SparseEasyAnimateAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1
    assert handle.summary()["model_type"] == "easyanimate"
    assert handle.summary()["model_key"] == "easyanimate-v5-t2v-12b"

    handle.restore()
    assert [block.attn1.get_processor() for block in transformer.transformer_blocks] == original
    assert transformer.hooks[0][1].removed is True


def test_discover_model_enumerates_real_easyanimate_diffusers_class():
    from diffusers.models.transformers.transformer_easyanimate import (
        EasyAnimateTransformer3DModel as DiffusersEasyAnimate,
    )

    transformer = DiffusersEasyAnimate(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        sample_width=4,
        sample_height=4,
        time_embed_dim=16,
        text_embed_dim=8,
    )
    model_info = discover_model(_Pipe(transformer))

    assert model_info.model_type == "easyanimate"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.transformer_blocks.0.attn1",
    ]


@pytest.mark.parametrize(("method", "config"), NEW_BACKBONE_PROCESSOR_METHODS)
def test_apply_and_restore_mochi_sparse_processor_uses_processor_attribute(method, config):
    transformer = MochiTransformer3DModel()
    pipe = _Pipe(transformer)
    original = [block.attn1.processor for block in transformer.transformer_blocks]

    handle = sparsevideo.apply_sparse_attention(pipe, method=method, config=config)

    installed = [block.attn1.processor for block in transformer.transformer_blocks]
    assert all(isinstance(processor, SparseMochiAttnProcessor) for processor in installed)
    assert len(transformer.hooks) == 1
    assert handle.summary()["model_type"] == "mochi"
    assert handle.summary()["model_key"] == "mochi-1"

    handle.restore()
    assert [block.attn1.processor for block in transformer.transformer_blocks] == original
    assert transformer.hooks[0][1].removed is True


def test_discover_model_enumerates_real_mochi_diffusers_class():
    from diffusers.models.transformers.transformer_mochi import MochiTransformer3DModel as DiffusersMochi

    transformer = DiffusersMochi(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=16,
        pooled_projection_dim=16,
        text_embed_dim=16,
        time_embed_dim=32,
    )
    model_info = discover_model(_Pipe(transformer))

    assert model_info.model_type == "mochi"
    assert [path for path, _ in model_info.iter_self_attn_modules()] == [
        "transformer.transformer_blocks.0.attn1",
    ]


@pytest.mark.parametrize(
    ("transformer", "message"),
    [
        (
            SanaVideoTransformer3DModel(),
            "SanaVideo uses Diffusers' SanaLinearAttnProcessor3_0 linear attention",
        ),
        (
            Kandinsky5Transformer3DModel(),
            "Kandinsky5 exposes native sparse attention controls",
        ),
    ],
)
def test_discover_model_rejects_deferred_non_processor_swap_backbones(transformer, message):
    with pytest.raises(ValueError, match=message):
        discover_model(_Pipe(transformer))


def test_discover_model_rejects_real_sana_video_diffusers_class():
    from diffusers.models.transformers.transformer_sana_video import (
        SanaVideoTransformer3DModel as DiffusersSanaVideo,
    )

    transformer = DiffusersSanaVideo(
        in_channels=4,
        out_channels=4,
        num_attention_heads=1,
        attention_head_dim=4,
        num_layers=1,
        num_cross_attention_heads=1,
        cross_attention_head_dim=4,
        cross_attention_dim=8,
        caption_channels=8,
        sample_size=4,
        patch_size=(1, 2, 2),
    )

    with pytest.raises(ValueError, match="linear attention"):
        discover_model(_Pipe(transformer))


def test_discover_model_rejects_real_kandinsky5_diffusers_class():
    from diffusers.models.transformers.transformer_kandinsky import (
        Kandinsky5Transformer3DModel as DiffusersKandinsky5,
    )

    transformer = DiffusersKandinsky5(
        in_visual_dim=4,
        in_text_dim=8,
        in_text_dim2=8,
        time_dim=8,
        out_visual_dim=4,
        model_dim=16,
        ff_dim=32,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(4, 4, 8),
    )

    with pytest.raises(ValueError, match="native sparse attention controls"):
        discover_model(_Pipe(transformer))
