# Block Sparse Attention

SparseVideo uses this upstream project for Draft Attention backend parity.

Upstream repository: `https://github.com/mit-han-lab/Block-Sparse-Attention`

The upstream Draft Attention code imports:

```python
from block_sparse_attn import block_sparse_attn_func
```

and dispatches `block_sparse_attn_func(...)` for the sparse attention path.
SparseVideo must therefore own and dispatch an equivalent backend for Draft
benchmark and quality claims.
