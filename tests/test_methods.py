import torch

import sparsevideo
from sparsevideo.methods.draft import _draft_attention
from sparsevideo.methods.sta import _sta_attention


class _Hook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _Attn:
    def __init__(self):
        self.processor = object()
        self.history = []

    def get_processor(self):
        return self.processor

    def set_processor(self, processor):
        self.processor = processor
        self.history.append(processor)


class _Block:
    def __init__(self):
        self.attn1 = _Attn()


class _WanTransformer:
    def __init__(self):
        self.blocks = [_Block(), _Block()]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        handle = _Hook()
        self.hooks.append((hook, with_kwargs, handle))
        return handle


class _Pipe:
    def __init__(self):
        self.transformer = _WanTransformer()


class _CogVideoXBlock:
    def __init__(self):
        self.attn1 = _Attn()


class _CogVideoXTransformer:
    def __init__(self):
        self.transformer_blocks = [_CogVideoXBlock()]
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        handle = _Hook()
        self.hooks.append((hook, with_kwargs, handle))
        return handle


class _CogVideoXPipe:
    def __init__(self):
        self.transformer = _CogVideoXTransformer()


def test_public_methods_are_registered():
    assert set(sparsevideo.list_methods()) == {
        "dense",
        "svg1",
        "svg2",
        "spargeattn",
        "radial",
        "sta",
        "draft",
        "adacluster",
        "flashomni",
        "svoo",
    }


def test_apply_sparse_attention_restore_on_wan_like_pipeline():
    pipe = _Pipe()
    originals = [block.attn1.get_processor() for block in pipe.transformer.blocks]

    handle = sparsevideo.apply_sparse_attention(
        pipe,
        method="draft",
        config={"skip_first_steps": 999},
    )

    patched = [block.attn1.get_processor() for block in pipe.transformer.blocks]
    assert patched != originals

    handle.restore()

    restored = [block.attn1.get_processor() for block in pipe.transformer.blocks]
    assert restored == originals
    assert all(h.removed for _, _, h in pipe.transformer.hooks)


def test_dense_is_baseline_for_any_discovered_model():
    pipe = _CogVideoXPipe()
    original = pipe.transformer.transformer_blocks[0].attn1.get_processor()

    handle = sparsevideo.apply_sparse_attention(pipe, method="dense")

    assert pipe.transformer.transformer_blocks[0].attn1.get_processor() is original
    handle.restore()
    assert pipe.transformer.transformer_blocks[0].attn1.get_processor() is original


def test_sta_wan_cpu_smoke_shape_without_text_prefix():
    query = torch.randn(1, 13 * 4 * 4, 2, 16)

    output = _sta_attention(
        query,
        query,
        query,
        tile_size=(2, 2, 2),
        kernel_size=(3, 3, 3),
        model_type="wan",
        text_len=0,
    )

    assert output.shape == query.shape


def test_draft_wan_cpu_dense_fallback_shape_without_text_prefix():
    query = torch.randn(1, 13 * 4 * 4, 2, 16)

    output = _draft_attention(
        query,
        query,
        query,
        budget=0.5,
        pool_h=2,
        pool_w=2,
        model_type="wan",
        text_len=0,
    )

    assert output.shape == query.shape
