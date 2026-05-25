import math
import torch

# Try to load the C++ extension
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    sta_fwd = getattr(fastvideo_kernel_ops, "sta_fwd", None)
except ImportError:
    sta_fwd = None


def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: list,
    text_length: int,
    has_text: bool = True,
    seq_shape: str = "30x48x80",
) -> torch.Tensor:
    # Check if the specific op is available
    if sta_fwd is None:
        raise RuntimeError("fastvideo_kernel_ops.sta_fwd is required for STA H100 inference")

    seq_length = q.shape[2]
    shape_map = {"30x48x80": 1, "36x48x48": 2, "18x48x80": 3}

    if has_text:
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)

    output = torch.empty_like(q)
    flag = shape_map[seq_shape]

    for head_idx, (t, h, w) in enumerate(window_size):
        # Per-head slices are not contiguous in the batch dimension when batch>1
        # (they keep the original head-stride). The TK kernel assumes contiguous
        # [B, H, S, D] layout, so we materialize a contiguous [B,1,S,D] view.
        q_h = q[:, head_idx:head_idx + 1].contiguous()
        k_h = k[:, head_idx:head_idx + 1].contiguous()
        v_h = v[:, head_idx:head_idx + 1].contiguous()
        o_h = torch.empty_like(q_h)
        sta_fwd(
            q_h, k_h,
            v_h, o_h,
            t, h, w, text_length, False, has_text, flag
        )
        output[:, head_idx:head_idx + 1] = o_h

    if has_text:
        sta_fwd(q.contiguous(), k.contiguous(), v.contiguous(), output, 3, 3, 3, text_length, True, True, flag)

    return output[:, :, :seq_length]
