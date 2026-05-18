from __future__ import annotations

import torch


def pad_text_clusters(dynamic_map, q_sizes, k_sizes, q_sorted_indices, text_len: int, prompt_length=None):
    if text_len <= 0:
        return dynamic_map, q_sizes, k_sizes, q_sorted_indices

    B, H, nqc, nkc = dynamic_map.shape
    device = dynamic_map.device
    if prompt_length is None:
        prompt_len = int(text_len)
    else:
        prompt_len = max(0, min(int(prompt_length), int(text_len)))
    unprompt_len = int(text_len) - prompt_len

    full_map = torch.zeros(B, H, nqc + 2, nkc + 2, dtype=torch.bool, device=device)
    full_map[:, :, :nqc, :nkc] = dynamic_map
    full_map[:, :, :nqc, nkc] = True
    full_map[:, :, nqc, : nkc + 1] = True
    full_map[:, :, nqc + 1, nkc + 1] = True

    prompt_q = torch.full((B, H, 1), prompt_len, dtype=q_sizes.dtype, device=q_sizes.device)
    prompt_k = torch.full((B, H, 1), prompt_len, dtype=k_sizes.dtype, device=k_sizes.device)
    unprompt_q = torch.full((B, H, 1), unprompt_len, dtype=q_sizes.dtype, device=q_sizes.device)
    unprompt_k = torch.full((B, H, 1), unprompt_len, dtype=k_sizes.dtype, device=k_sizes.device)
    q_sizes = torch.cat([q_sizes, prompt_q, unprompt_q], dim=-1)
    k_sizes = torch.cat([k_sizes, prompt_k, unprompt_k], dim=-1)

    video_len = q_sorted_indices.shape[-1]
    tail = torch.arange(video_len, video_len + text_len, device=device, dtype=q_sorted_indices.dtype)
    tail = tail.expand(q_sorted_indices.shape[0], text_len)
    q_sorted_indices = torch.cat([q_sorted_indices, tail], dim=-1)
    return full_map, q_sizes, k_sizes, q_sorted_indices
