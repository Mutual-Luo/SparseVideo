SparseVideo-owned FastVideo STA H100 source.

This directory is a buildable runtime-source copy of FastVideo's
`fastvideo-kernel` runtime source for the H100/ThunderKittens STA path. The
`training_free/` checkout remains reference-only; SparseVideo should not import
it at runtime.

The current A100 environment should use SparseVideo's SM80 block-sparse CUDA STA
path. The H100 extension is only useful on Hopper targets and should be built with
`FASTVIDEO_KERNEL_BUILD_TK=ON` and a Hopper CUDA architecture.

Build on a Hopper-capable build environment with CMake >= 3.26:

```bash
PATH=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin:$PATH \
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=9.0a \
CMAKE_CUDA_ARCHITECTURES=90a \
FASTVIDEO_KERNEL_BUILD_TK=ON \
src/sparsevideo/kernels/native/sta_h100/setup.sh
```

On this checkout, the conda environment provides a new enough CMake. If the
host compiler lacks C++20 support for ThunderKittens headers, set `CC`, `CXX`,
and `CUDAHOSTCXX` to the conda GCC/G++ wrappers before running the script.

This copy keeps upstream CMake metadata, Python wrappers, C++/CUDA sources,
ThunderKittens headers, CUTLASS headers, and licenses. It intentionally excludes
only git metadata, build outputs, caches, docs, demos, and unrelated checkout
state.
