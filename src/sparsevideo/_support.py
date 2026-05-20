from __future__ import annotations

from typing import Optional, Tuple


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
