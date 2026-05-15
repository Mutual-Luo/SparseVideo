# Native Kernel Extension

SparseVideo-owned fused C++/CUDA kernels live here. Runtime discovery only
searches `SPARSEVIDEO_NATIVE_KERNEL_ROOT` and this package's `build/`
directory. It does not search `training_free/`.

`svg_svoo_fused/` is the local native source for the fused RMSNorm/RoPE
extension used by SVG and SVOO paths. It keeps only the actually called source
and headers in this package. Third-party headers come from optional installed
packages, mainly Torch and `flashinfer-python`, so SparseVideo does not vendor
large external repositories into the wheel.

Build for the current environment:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
src/sparsevideo/kernels/native/svg_svoo_fused/setup.sh
```
