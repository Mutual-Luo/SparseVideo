# SparseVideo-Owned Draft Block-Sparse Backend

This directory vendors the MIT Han Lab `Block-Sparse-Attention` runtime source
needed by Draft Attention. The upstream Draft implementation calls
`block_sparse_attn_func`, so SparseVideo treats this backend as required kernel
parity for serious Draft inference.

Source:

- Upstream repo: `https://github.com/mit-han-lab/Block-Sparse-Attention`
- Copied revision: `6ec5a27a0cd6bd92ea6296698d64e460c73da27e`
- License: BSD 3-Clause, preserved in `LICENSE`

Build in place for the current A100 environment:

```bash
CUDA_HOME=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDA_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDACXX=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/nvcc \
PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin:$PATH \
LD_LIBRARY_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/lib:/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
BLOCK_SPARSE_ATTN_CUDA_ARCHS=80 \
src/sparsevideo/kernels/native/draft_block_sparse/setup.sh
```

By default `setup.sh` uses `BLOCK_SPARSE_ATTN_BUILD_MODE=full`, which builds
the complete upstream extension including dense attention, split-KV, and
backward kernels. For a faster local Draft-only build, set
`BLOCK_SPARSE_ATTN_BUILD_MODE=draft_inference`; that mode keeps the upstream
Python/C++ API surface but compiles only the MIT block-sparse forward kernel
that Draft inference calls (`fwd_block`, head_dim 128, fp16/bf16, sm80).

This copy keeps the upstream Python package and CUDA/C++ kernel source. It
intentionally excludes only git metadata, generated build outputs, caches, docs,
and tests. It keeps the CUTLASS headers from the upstream submodule revision
`a75b4ac483166189a45290783cb0a18af5ff0ea5` under `csrc/cutlass/` because this
older Block-Sparse-Attention revision does not compile against the newer
CUTLASS headers used by other kernels.
