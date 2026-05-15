from __future__ import annotations

from math import floor


def resolve_first_layers(first_layers_fp, total_layers):
    value = float(first_layers_fp or 0)
    if value <= 0:
        return 0
    if value < 1:
        return int(floor(value * total_layers))
    return min(total_layers, int(value))


def resolve_first_steps(first_times_fp, num_inference_steps):
    value = float(first_times_fp or 0)
    if value <= 0:
        return 0
    if value < 1:
        return int(floor(value * int(num_inference_steps)))
    return int(value)
