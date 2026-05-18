# SparseVideo-Owned SpargeAttn Kernels

This directory vendors the SpargeAttn runtime source that SparseVideo needs for
`spargeattn` sparse modes. `training_free/SpargeAttn` remains reference-only and
must not be imported at runtime.

Build in place for the current environment:

```bash
PYTHON=/home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python \
TORCH_CUDA_ARCH_LIST=8.0 \
src/sparsevideo/kernels/native/spargeattn/setup.sh
```

After a successful build, this directory should contain:

```text
spas_sage_attn/_qattn*.so
spas_sage_attn/_fused*.so
```

`scripts/infer.py --method spargeattn --method-config mode=topk --method-config
value=0.5 --dry-run` reports whether those SparseVideo-owned extensions are
available.
