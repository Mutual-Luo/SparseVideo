from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLayout:
    context_len: int
    video_len: int
    tail_len: int = 0

    @property
    def vid_start(self) -> int:
        return self.context_len

    @property
    def video_end(self) -> int:
        return self.context_len + self.video_len


def infer_video_token_layout(num_tokens: int, model_type: str, text_len: int = 0) -> TokenLayout:
    if text_len < 0 or text_len > num_tokens:
        raise ValueError(f"Invalid text_len={text_len} for sequence length {num_tokens}")

    if model_type == "hunyuan_video":
        return TokenLayout(context_len=0, video_len=num_tokens - text_len, tail_len=text_len)

    # SparseVideo patches Wan attn1 self-attention only, whose sequence is video tokens.
    return TokenLayout(context_len=0, video_len=num_tokens, tail_len=0)
