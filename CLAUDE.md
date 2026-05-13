# SparseVideo Project Instructions

SparseVideo should make sparse attention plug-and-play for video diffusion generation.

The package is an extension layer for existing ecosystems, especially Diffusers and DiffSynth-Studio. Users should be
able to enable a sparse attention method, switch methods with minimal code changes, compare results fairly, and restore
the original dense attention path.

## Success Criteria

- One small public API for applying and restoring sparse attention.
- Unified method registration and config handling.
- Support real video diffusion pipelines, not only toy tensors.
- Correctness tests before performance claims.
- Optional kernels must not be required for basic import.
- New methods should be easy to add without model-specific hacks.
- The project should remain lightweight and pip-package friendly.

## Included Training-Free Methods

Use these public names:

1. `svg1`: `training_free/Sparse-VideoGen` SVG method.
2. `svg2`: `training_free/Sparse-VideoGen` second method.
3. `spargeattn`: `training_free/SpargeAttn`.
4. `radial`: `training_free/radial-attention`.
5. `sta`: `training_free/FastVideo` Sliding Tile Attention.
6. `draft`: `training_free/draft-attention`.
7. `adacluster`: `training_free/Adacluster`.
8. `flashomni`: `training_free/FlashOmni`.
9. `svoo`: `training_free/SVOO`.

Always keep `dense` as the baseline and fallback.

## Engineering Rules

- Think before coding. State assumptions and ask when guessing could cause damage.
- Keep changes surgical. Touch only files needed for the task.
- Prefer simple, explicit code over speculative abstractions.
- Preserve existing style unless changing it is necessary.
- Do not install old upstream method requirements into the main environment blindly.
- Treat `training_free/` repositories as references unless a task explicitly says to run them.
- Verify the result with the narrowest useful command or test.

## Environment

```text
Python: /home/dataset-assist-0/luojy/miniconda3/envs/sparsevideo/bin/python
Models: /home/dataset-assist-0/luojy/models
```
