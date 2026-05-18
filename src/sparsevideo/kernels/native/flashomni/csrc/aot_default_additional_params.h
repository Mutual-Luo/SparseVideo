#define BATCH_SPARSEFA_ADDITIONAL_FUNC_PARAMS , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_mask_indptr, std::optional<at::Tensor> maybe_alibi_slopes, double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta

#define BATCH_SPARSEFA_ADDITIONAL_PARAMS_SETTER params.maybe_custom_mask = maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()): nullptr; \
params.maybe_mask_indptr = maybe_mask_indptr ? static_cast<int32_t*>(maybe_mask_indptr->data_ptr()): nullptr; \
params.maybe_alibi_slopes = maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()): nullptr; \
params.logits_soft_cap = logits_soft_cap; \
params.sm_scale = sm_scale; \
params.rope_rcp_scale = rope_rcp_scale; \
params.rope_rcp_theta = rope_rcp_theta;

