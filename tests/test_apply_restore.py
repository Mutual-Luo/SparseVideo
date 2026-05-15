from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sparsevideo
from sparsevideo.processors.hunyuan_video import SparseHunyuanVideoAttnProcessor
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


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


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
