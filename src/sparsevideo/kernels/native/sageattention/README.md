# SparseVideo-Owned SageAttention Runtime

This directory vendors the SageAttention dense runtime used by radial-attention
when `use_sage_attention=True` enters the upstream dense warmup branch.
`training_free/radial-attention/third_party/SageAttention` remains
reference-only and is not imported at SparseVideo runtime.

Source:

- Upstream repo: `https://github.com/thu-ml/SageAttention`
- Copied revision: `628a89f1619b474ef2ed735b1e907ebca57fc1cd`
- License: Apache 2.0, preserved in `LICENSE`

Only runtime/build files needed by `sageattention.sageattn` are kept:
`sageattention/`, `csrc/`, `setup.py`, and `LICENSE`. Assets, examples,
benchmarks, git metadata, caches, and build outputs are intentionally excluded.

Build in place for the current A100 environment:

```bash
CUDA_HOME=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDA_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDACXX=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/nvcc \
PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin:$PATH \
LD_LIBRARY_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/lib:/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
EXT_PARALLEL=4 \
src/sparsevideo/kernels/native/sageattention/setup.sh
```

After a successful A100 build, this directory should contain:

```text
sageattention/_qattn_sm80*.so
sageattention/_fused*.so
```
