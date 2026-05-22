from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional


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

    if model_type in ("hunyuan_video", "cogvideox", "mochi", "easyanimate"):
        return TokenLayout(context_len=0, video_len=num_tokens - text_len, tail_len=text_len)

    # SparseVideo patches self-attention only for Wan and LTX, whose sequence is video tokens.
    return TokenLayout(context_len=0, video_len=num_tokens, tail_len=0)


def parse_seq_shape(seq_shape: Optional[str | tuple[int, int, int] | list[int]]) -> Optional[tuple[int, int, int]]:
    if seq_shape is None:
        return None
    if isinstance(seq_shape, str):
        parts = seq_shape.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid seq_shape={seq_shape!r}; expected TxHxW, e.g. 21x45x80")
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    if len(seq_shape) != 3:
        raise ValueError(f"Invalid seq_shape={seq_shape!r}; expected three integers")
    return tuple(int(part) for part in seq_shape)


def infer_video_frame_shape(
    video_len: int,
    model_type: str,
    seq_shape: Optional[str | tuple[int, int, int] | list[int]] = None,
) -> tuple[int, int, int]:
    override = parse_seq_shape(seq_shape)
    if override is not None:
        if override[0] * override[1] * override[2] != video_len:
            raise ValueError(
                f"seq_shape={override[0]}x{override[1]}x{override[2]} does not match "
                f"video token length {video_len}"
            )
        return override

    # These match the upstream 720p paths we care about first. The order matters:
    # Wan 81-frame Diffusers runs as 21x45x80, while FastVideo's native STA test
    # shape is 18x48x80. Hunyuan 129-frame SVG uses 33x45x80; FastVideo STA uses
    # 30x48x80 with text padded after the image tokens.
    known_shapes = {
        "wan": ((21, 45, 80), (18, 48, 80), (33, 45, 80)),
        "hunyuan_video": ((33, 45, 80), (30, 48, 80), (21, 45, 80)),
        "cogvideox": ((13, 30, 45), (13, 45, 80), (13, 60, 90), (25, 45, 80)),
        "ltx_video": ((21, 16, 22), (21, 22, 40)),
        "allegro": ((22, 45, 80), (8, 45, 80)),
        "easyanimate": ((13, 16, 16), (13, 30, 45), (25, 45, 80)),
    }
    for shape in known_shapes.get(model_type, ()):
        if shape[0] * shape[1] * shape[2] == video_len:
            return shape

    frame_order = {
        "wan": (21, 18, 33, 17, 13, 9, 5, 25, 2, 1),
        "hunyuan_video": (33, 30, 21, 17, 13, 9, 5, 25, 18, 2, 1),
        "cogvideox": (13, 25, 17, 9, 5, 21, 33, 2, 1),
        "ltx_video": (21, 17, 13, 9, 5, 25, 33, 2, 1),
        "allegro": (22, 8, 17, 13, 9, 5, 25, 33, 2, 1),
        "easyanimate": (13, 17, 9, 5, 25, 21, 33, 2, 1),
    }.get(model_type, (33, 30, 21, 18, 17, 13, 9, 5, 25))
    for frames in frame_order:
        if video_len % frames == 0:
            height, width = _factor_spatial_grid(video_len // frames)
            return frames, height, width

    frames = 1
    spatial = max(1, video_len)
    height, width = _factor_spatial_grid(spatial)
    return frames, height, width


def infer_video_frame_count(video_len: int, model_type: str) -> int:
    return infer_video_frame_shape(video_len, model_type=model_type)[0]


def _factor_spatial_grid(spatial: int, preferred_ratio: float = 9 / 16) -> tuple[int, int]:
    best_h = 1
    best_w = spatial
    best_score = float("inf")
    for height in range(1, int(spatial**0.5) + 1):
        if spatial % height != 0:
            continue
        width = spatial // height
        score = abs((height / width) - preferred_ratio)
        if score < best_score:
            best_h = height
            best_w = width
            best_score = score
    if best_h * best_w == spatial:
        return best_h, best_w
    height = int(spatial**0.5)
    return height, ceil(spatial / max(1, height))
