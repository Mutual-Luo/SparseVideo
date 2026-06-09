from __future__ import annotations

from typing import Optional, Tuple


STA_UNSUPPORTED_REASON = (
    "STA is not supported in this release. Please choose another method, "
    "e.g. 'svg2', 'svoo', 'spargeattn', or 'radial'."
)

DIFFSYNTH_STA_UNSUPPORTED_REASON = (
    "STA is not supported for the DiffSynth backend. Use the Diffusers backend "
    "for STA inference."
)

STA_WAN21_T2V_13B_UNSUPPORTED_REASON = (
    "STA is temporarily unsupported for Wan2.1-T2V-1.3B. The current version "
    "has not found suitable STA parameters that balance efficiency and quality "
    "for this model."
)

MOCHI_SPARSE_ATTENTION_WARNING = (
    "Mochi-specific sparse attention warning: sparse attention is not recommended for Mochi. "
    "On Mochi, sparse attention is not expected to provide a speedup over "
    "dense while preserving output quality. Use dense attention for "
    "quality-sensitive Mochi runs."
)

UNSUPPORTED_METHODS_BY_MODEL_KEY: dict[tuple[str, str], str] = {
    ("sta", "wan21-t2v-1.3b"): STA_WAN21_T2V_13B_UNSUPPORTED_REASON,
}

LIMITED_METHODS_BY_MODEL_TYPE: dict[str, Tuple[str, ...]] = {}

UNVALIDATED_METHOD_REASONS = {
    "svg1": (
        "SVG1 placement/FlexAttention has not been validated for this backbone's "
        "text/video token layout."
    ),
    "spargeattn": (
        "SpargeAttn is not enabled on this backbone because its text/video token "
        "ownership and attention-mask semantics have not been validated against "
        "SparseVideo's owned SpargeAttn kernels."
    ),
    "radial": (
        "Radial layout and block-size assumptions have not been validated for this "
        "backbone's video/text layout."
    ),
    "sta": (
        "FastVideo STA parity depends on upstream seq_shape and mask-strategy layouts; "
        "this backbone has no validated STA mask/layout evidence."
    ),
    "draft": (
        "Draft attention parity depends on upstream video/text latent layout and the "
        "owned block-sparse backend; this backbone has not been ported."
    ),
    "adacluster": (
        "AdaCluster upstream cluster profiles are validated only for Wan/Hunyuan "
        "layouts; this backbone has no validated profile."
    ),
    "flashomni": (
        "FlashOmni's default explicit path requires caller-provided sparse-info "
        "tensors, and no upstream-compatible video sparse-info policy is validated "
        "for this backbone."
    ),
}


def unvalidated_method_reason(
    method: str,
    *,
    smoke_methods: Optional[Tuple[str, ...]] = None,
) -> str:
    reason = UNVALIDATED_METHOD_REASONS.get(
        method,
        "This method has not been validated for this backbone's attention layout.",
    )
    if smoke_methods:
        smoke_label = "/".join(("dense", *smoke_methods))
        return f"{reason} Current smoke coverage is {smoke_label} only."
    return reason


def unsupported_method_model_reason(method: str, model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    return UNSUPPORTED_METHODS_BY_MODEL_KEY.get((method, model_key))
