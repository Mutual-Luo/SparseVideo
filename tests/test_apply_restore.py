from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo
from sparsevideo._model_info import discover_model
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


class CogVideoXImageToVideoPipeline(_Pipe):
    pass


NEW_BACKBONE_PROCESSOR_METHODS = [
    ("svg1", {}),
    ("svg2", {}),
    ("spargeattn", {"mode": "full"}),
    ("radial", {}),
    ("sta", {}),
    ("draft", {}),
    ("adacluster", {}),
    ("flashomni", {"sparse_pattern": "paper_mmdit", "max_order": 0, "use_sparse_gemm": False}),
]


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


def test_runtime_summary_records_dense_dispatch_for_cpu_dense_gate_methods():
    query = torch.randn(1, 4, 2, 3)
    cases = [
        ("svg1", {}, "SVG1Method", "torch_sdpa"),
        ("svg2", {}, "SVG2Method", "torch_sdpa"),
        ("svoo", {}, "SVOOMethod", "torch_sdpa"),
        ("draft", {}, "DraftMethod", "torch_sdpa"),
        ("spargeattn", {"mode": "full"}, "SpargeAttnMethod", "diffusers_dispatch"),
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


def test_apply_sparse_attention_resolves_scheduler_first_times_fp_like_upstream_inference():
    for method in ("svg1", "svg2", "svoo"):
        transformer = WanTinyTransformer()
        pipe = _Pipe(transformer, scheduler=_Scheduler())

        handle = sparsevideo.apply_sparse_attention(pipe, method=method)

        assert handle._method_instance.config["first_times_fp"] == 909.0
        assert handle._method_instance.config["num_inference_steps"] == 50
        handle.restore()


def test_apply_sparse_attention_resolves_scheduler_first_times_fp_with_dynamic_shift_scheduler():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer, scheduler=_DynamicShiftScheduler())

    handle = sparsevideo.apply_sparse_attention(pipe, method="svg2")

    assert handle._method_instance.config["first_times_fp"] == 909.0
    assert handle._method_instance.config["num_inference_steps"] == 50
    handle.restore()


def test_apply_sparse_attention_keeps_resolved_first_times_fp_threshold():
    transformer = WanTinyTransformer()
    pipe = _Pipe(transformer, scheduler=_Scheduler())

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="svoo",
        config={"first_times_fp": 925},
    )

    assert handle._method_instance.config["first_times_fp"] == 925
    handle.restore()


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
        "model_type": "wan",
        "model_key": None,
        "num_self_attn_layers": 2,
        "installed_processor_count": 0,
        "installed_processor_paths": [],
        "current_processor_classes": {},
        "step_tracker_hook_count": 0,
        "restore_callback_count": 0,
        "method_class": "DenseMethod",
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
