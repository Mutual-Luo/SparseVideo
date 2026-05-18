SparseVideo-owned FlashOmni runtime source.

This directory is a buildable runtime-source copy of `training_free/FlashOmni`
for SparseVideo runtime use. The `training_free/` checkout remains
reference-only; SparseVideo should import this local runtime when its AOT
extension is built.

Build on the SparseVideo environment:

```bash
CUDA_HOME=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDA_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo \
CUDACXX=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/nvcc \
PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin:$PATH \
LD_LIBRARY_PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/lib:/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
FLASHOMNI_ENABLE_AOT=1 \
src/sparsevideo/kernels/native/flashomni/setup.sh
```

This copy keeps the upstream FlashOmni Python package, CUDA/C++ source, AOT
generators, generated sparse attention instantiations, and CUTLASS headers
needed by upstream `setup.py`. It intentionally excludes only git metadata,
benchmarks/docs, build outputs, caches, and unrelated checkout state.
