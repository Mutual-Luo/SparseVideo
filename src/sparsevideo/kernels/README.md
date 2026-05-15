# Shared Kernels

This package is for reusable kernel primitives used by multiple sparse
attention methods, such as generic k-means, block-sparse attention, FlashInfer
wrappers, and shared fused norm/RoPE helpers.

The shared fused norm/RoPE helper can use a SparseVideo-owned `_kernels`
C++/CUDA extension when it has been built under `sparsevideo.kernels.native`
or supplied via `SPARSEVIDEO_NATIVE_KERNEL_ROOT`, controlled by
`SPARSEVIDEO_FUSED_KERNEL_BACKEND=auto|native|triton|pytorch`. The default is
`auto`: use `_kernels` if present, otherwise keep the lightweight Triton path.

Do not put method policy, config, or upstream-specific orchestration here.
If a kernel or helper is private to one method, place it under that method
package, for example `sparsevideo.methods.svoo.kernels`.
