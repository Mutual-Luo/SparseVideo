# Native Kernel Extension

SparseVideo-owned fused C++/CUDA kernels live here. Runtime discovery only
searches `SPARSEVIDEO_NATIVE_KERNEL_ROOT` and this package's `build/`
directory. It does not search `training_free/`.

`svg_svoo_fused/` is the local native source for the fused RMSNorm/RoPE
extension used by SVG and SVOO paths. It keeps only the actually called source
and headers in this package. Third-party headers come from optional installed
packages, mainly Torch and `flashinfer-python`, so SparseVideo does not vendor
large external repositories into the wheel.

`spargeattn/` is the local native source for SpargeAttn's `_qattn` and `_fused`
extensions. SparseVideo rejects `training_free/SpargeAttn` as a runtime package;
build this owned source for SpargeAttn/radial Sage runtime parity. Environment
`spas_sage_attn` packages are not accepted for SparseVideo runtime parity.

`sageattention/` is the local native source for radial-attention's dense
SageAttention warmup path when `use_sage_attention=True`. SparseVideo rejects
`training_free/radial-attention/third_party/SageAttention` and environment
`sageattention` packages as runtime parity backends.

`flashomni/` is a buildable local runtime-source copy of FlashOmni's Python
runtime, CUDA/C++ source, AOT generators, generated kernels, and CUTLASS
headers.
SparseVideo requires this local runtime once `flashomni_kernels*.so` is built;
environment `flashomni` packages are not accepted for runtime parity.

`sta_h100/` is a buildable local runtime-source copy of FastVideo's H100/TK
STA source. The current A100 environment uses the SparseVideo-owned SM80
block-sparse CUDA backend in `draft_block_sparse/` with STA's tile-window mask;
build `sta_h100/` only on a Hopper target.

Build for the current environment:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
src/sparsevideo/kernels/native/svg_svoo_fused/setup.sh
```

Build SpargeAttn kernels for the current environment:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
src/sparsevideo/kernels/native/spargeattn/setup.sh
```

Build SageAttention kernels for radial dense Sage warmup:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
EXT_PARALLEL=4 \
src/sparsevideo/kernels/native/sageattention/setup.sh
```

Build FlashOmni AOT kernels for the current environment:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
FLASHOMNI_ENABLE_AOT=1 \
src/sparsevideo/kernels/native/flashomni/setup.sh
```

Build FastVideo STA H100 kernels on a Hopper-capable build environment with
CMake >= 3.26:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
CMAKE_CUDA_ARCHITECTURES=90a \
FASTVIDEO_KERNEL_BUILD_TK=ON \
src/sparsevideo/kernels/native/sta_h100/setup.sh
```
