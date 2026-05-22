from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


DEFAULT_WAN_1280X768_CANDIDATES = (
    (3, 1, 10),
    (1, 5, 7),
    (3, 3, 3),
    (1, 6, 5),
    (1, 3, 10),
    (3, 6, 1),
)
DEFAULT_WAN_1280X768_FULL_WINDOW = (3, 6, 10)
DEFAULT_HUNYUAN_1280X768_CANDIDATES = (
    (5, 3, 3),
    (1, 6, 10),
    (3, 3, 5),
    (5, 1, 10),
    (5, 6, 1),
)
DEFAULT_HUNYUAN_1280X768_FULL_WINDOW = (5, 6, 10)
MODEL_STRATEGY_SHAPES = {
    "wan21-t2v-1.3b": (50, 30, 12),
    "wan21-vace-1.3b": (50, 30, 12),
    "wan21-t2v-14b": (50, 40, 40),
    "wan21-i2v-14b": (50, 40, 40),
    "wan21-vace-14b": (50, 40, 40),
    "skyreels-v2-t2v-14b": (50, 40, 40),
    "skyreels-v2-i2v-14b": (50, 40, 40),
    "wan22-t2v-a14b": (40, 40, 40),
    "wan22-i2v-a14b": (40, 40, 40),
    "wan22-animate-14b": (20, 40, 40),
    "hunyuan-t2v": (50, 60, 24),
    "hunyuan-i2v": (50, 60, 24),
    "cogvideox-t2v": (50, 42, 48),
    "cogvideox-i2v": (50, 42, 48),
    "ltx-video": (50, 28, 32),
    "ltx-video-i2v": (50, 28, 32),
    "allegro": (100, 32, 24),
    "mochi-1": (64, 48, 24),
    "easyanimate-v5-t2v-12b": (50, 48, 48),
}
MODEL_ALIASES = {
    "wan1.3b": "wan21-t2v-1.3b",
    "wan21-1.3b": "wan21-t2v-1.3b",
    "vace": "wan21-vace-1.3b",
    "wan-vace": "wan21-vace-1.3b",
    "wan21-vace": "wan21-vace-1.3b",
    "wan14b": "wan21-t2v-14b",
    "wan21-14b": "wan21-t2v-14b",
    "wan-i2v": "wan21-i2v-14b",
    "wan14b-i2v": "wan21-i2v-14b",
    "wan21-i2v": "wan21-i2v-14b",
    "wan22": "wan22-t2v-a14b",
    "wan22-a14b": "wan22-t2v-a14b",
    "wan22-i2v": "wan22-i2v-a14b",
    "wananimate": "wan22-animate-14b",
    "wan-animate": "wan22-animate-14b",
    "wan22-animate": "wan22-animate-14b",
    "hunyuan": "hunyuan-t2v",
    "skyreels": "skyreels-v2-t2v-14b",
    "skyreels-v2": "skyreels-v2-t2v-14b",
    "skyreels-v2-t2v": "skyreels-v2-t2v-14b",
    "skyreels-i2v": "skyreels-v2-i2v-14b",
    "skyreels-v2-i2v": "skyreels-v2-i2v-14b",
    "cog": "cogvideox-t2v",
    "cogvideox": "cogvideox-t2v",
    "cog-i2v": "cogvideox-i2v",
    "cogvideox-5b-i2v": "cogvideox-i2v",
    "ltx": "ltx-video",
    "ltx-i2v": "ltx-video-i2v",
    "mochi": "mochi-1",
    "easyanimate": "easyanimate-v5-t2v-12b",
    "easyanimate-v5": "easyanimate-v5-t2v-12b",
}


def parse_window(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") or raw.startswith("("):
            value = ast.literal_eval(raw)
        else:
            value = raw.split(",")
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"STA window must be a 3-item sequence, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"STA window must contain 3 integers, got {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


def parse_windows(values: Any | None) -> list[tuple[int, int, int]]:
    if values is None:
        return list(DEFAULT_WAN_1280X768_CANDIDATES)
    if isinstance(values, str):
        raw = values.strip()
        if raw.startswith("["):
            values = ast.literal_eval(raw)
        else:
            values = [item.strip() for item in raw.split(";") if item.strip()]
    return [parse_window(item) for item in values]


def window_key(window: Sequence[int]) -> str:
    t, h, w = parse_window(window)
    return f"{t},{h},{w}"


def window_list(window: Sequence[int]) -> list[int]:
    t, h, w = parse_window(window)
    return [t, h, w]


def normalize_model_key(model: str) -> str:
    key = str(model).strip().lower()
    return MODEL_ALIASES.get(key, key)


def model_strategy_shape(model: str) -> tuple[int, int, int]:
    key = normalize_model_key(model)
    if key not in MODEL_STRATEGY_SHAPES:
        raise KeyError(f"STA search has no model shape default for {model!r}")
    return MODEL_STRATEGY_SHAPES[key]


def model_search_defaults(model: str | None) -> dict[str, Any]:
    key = normalize_model_key(model) if model is not None else ""
    if key.startswith("hunyuan"):
        return {
            "candidates": [window_list(item) for item in DEFAULT_HUNYUAN_1280X768_CANDIDATES],
            "full_window": window_list(DEFAULT_HUNYUAN_1280X768_FULL_WINDOW),
            "skip_time_steps": 15,
        }
    return {
        "candidates": [window_list(item) for item in DEFAULT_WAN_1280X768_CANDIDATES],
        "full_window": window_list(DEFAULT_WAN_1280X768_FULL_WINDOW),
        "skip_time_steps": 12,
    }


def head_losses(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[list[float], list[float]]:
    if reference.shape != candidate.shape:
        raise ValueError(f"STA search shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
    diff = (candidate.detach().float() - reference.detach().float())
    l1 = diff.abs().mean(dim=(0, 1, 3))
    l2 = diff.pow(2).mean(dim=(0, 1, 3))
    return l1.cpu().tolist(), l2.cpu().tolist()


class MaskSearchRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        prompt_id: str | None = None,
        candidates: Iterable[Sequence[int]] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        if not self.output_dir.is_absolute():
            self.output_dir = Path.cwd() / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates = [window_list(candidate) for candidate in (candidates or DEFAULT_WAN_1280X768_CANDIDATES)]
        safe_prompt_id = _safe_name(prompt_id or "prompt")
        unique = f"{int(time.time())}_{socket.gethostname()}_{os.getpid()}"
        self.path = self.output_dir / f"mask_search_{safe_prompt_id}_{unique}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")

    def record(
        self,
        *,
        step: int,
        layer: int,
        l1_loss: dict[str, list[float]],
        l2_loss: dict[str, list[float]],
    ) -> None:
        payload = {
            "format": "sparsevideo_sta_mask_search_v1",
            "step": int(step),
            "layer": int(layer),
            "candidates": self.candidates,
            "L1_loss": l1_loss,
            "L2_loss": l2_loss,
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:80] or "prompt"


def tune_search_results(
    search_dir: str | Path,
    output_file: str | Path,
    *,
    model: str | None = None,
    candidates: Iterable[Sequence[int]] | None = None,
    full_window: Sequence[int] | None = None,
    skip_time_steps: int | None = None,
    timesteps: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
) -> dict[str, Any]:
    defaults = model_search_defaults(model)
    candidate_windows = [window_list(item) for item in (defaults["candidates"] if candidates is None else candidates)]
    candidate_keys = [window_key(item) for item in candidate_windows]
    full = window_list(defaults["full_window"] if full_window is None else full_window)
    if skip_time_steps is None:
        skip_time_steps = int(defaults["skip_time_steps"])
    full_key = window_key(full)
    sums: dict[str, dict[tuple[int, int, int], float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[tuple[int, int, int], int]] = defaultdict(lambda: defaultdict(int))
    extent = {"step": -1, "layer": -1, "heads": 0}
    files = list(_iter_search_files(Path(search_dir)))
    if not files:
        raise FileNotFoundError(f"No STA search result files found in {search_dir}")

    used_files = []
    for path in files:
        if path.suffix == ".jsonl":
            used = _accumulate_jsonl(path, sums, counts, extent)
        else:
            used = _accumulate_upstream_json(path, sums, counts, extent)
        if used:
            used_files.append(path)
    if not used_files:
        raise ValueError(f"No usable STA search loss records found in {search_dir}")
    if model is not None:
        model_timesteps, model_layers, model_heads = model_strategy_shape(model)
        timesteps = timesteps if timesteps is not None else model_timesteps
        layers = layers if layers is not None else model_layers
        heads = heads if heads is not None else model_heads
    if timesteps is None:
        timesteps = extent["step"] + 1
    if layers is None:
        layers = extent["layer"] + 1
    if heads is None:
        heads = extent["heads"]
    if int(timesteps) <= 0 or int(layers) <= 0 or int(heads) <= 0:
        raise ValueError(
            "STA search records did not expose a positive steps/layers/heads shape; "
            f"inferred {(timesteps, layers, heads)}"
        )

    strategy: dict[str, list[int]] = {}
    strategy_counts = {key: 0 for key in [*candidate_keys, full_key]}
    total_tokens = 0
    total_length = 0
    full_tokens = full[0] * full[1] * full[2]
    for step in range(int(timesteps)):
        for layer in range(int(layers)):
            for head in range(int(heads)):
                if step < int(skip_time_steps):
                    chosen = full
                    chosen_key = full_key
                else:
                    chosen_key = _best_candidate_key(candidate_keys, sums, counts, step, layer, head)
                    chosen = window_list(chosen_key if chosen_key is not None else full)
                    chosen_key = chosen_key or full_key
                strategy[f"{step}_{layer}_{head}"] = chosen
                strategy_counts[chosen_key] = strategy_counts.get(chosen_key, 0) + 1
                total_tokens += chosen[0] * chosen[1] * chosen[2]
                total_length += full_tokens

    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(strategy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sparsity = 1.0 - (total_tokens / total_length) if total_length else 0.0
    return {
        "output_file": str(output_path),
        "entries": len(strategy),
        "timesteps": int(timesteps),
        "layers": int(layers),
        "heads": int(heads),
        "skip_time_steps": int(skip_time_steps),
        "sparsity": sparsity,
        "strategy_counts": strategy_counts,
        "search_files": [str(path) for path in used_files],
    }


def summarize_strategy(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = sorted({int(key.split("_")[0]) for key in data})
    layers = sorted({int(key.split("_")[1]) for key in data})
    heads = sorted({int(key.split("_")[2]) for key in data})
    counts: dict[str, int] = defaultdict(int)
    for value in data.values():
        counts[window_key(value)] += 1
    return {
        "entries": len(data),
        "timesteps": len(steps),
        "layers": len(layers),
        "heads": len(heads),
        "strategy_counts": dict(sorted(counts.items())),
    }


def _iter_search_files(search_dir: Path) -> Iterable[Path]:
    for pattern in ("*.jsonl", "*.json"):
        yield from sorted(search_dir.glob(pattern))


def _accumulate_jsonl(
    path: Path,
    sums: dict[str, dict[tuple[int, int, int], float]],
    counts: dict[str, dict[tuple[int, int, int], int]],
    extent: dict[str, int],
) -> bool:
    used = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "step" not in payload or "layer" not in payload or "L2_loss" not in payload:
                continue
            used = True
            step = int(payload["step"])
            layer = int(payload["layer"])
            for raw_key, losses in payload["L2_loss"].items():
                _update_extent(extent, step, layer, len(losses))
                key = window_key(raw_key)
                for head, loss in enumerate(losses):
                    index = (step, layer, int(head))
                    sums[key][index] += float(loss)
                    counts[key][index] += 1
    return used


def _accumulate_upstream_json(
    path: Path,
    sums: dict[str, dict[tuple[int, int, int], float]],
    counts: dict[str, dict[tuple[int, int, int], int]],
    extent: dict[str, int],
) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "L2_loss" not in payload:
        return False
    used = False
    for raw_key, steps in payload["L2_loss"].items():
        key = window_key(raw_key)
        for step, layers in enumerate(steps):
            for layer, heads in enumerate(layers):
                for head, loss in enumerate(heads):
                    used = True
                    _update_extent(extent, int(step), int(layer), len(heads))
                    index = (int(step), int(layer), int(head))
                    sums[key][index] += float(loss)
                    counts[key][index] += 1
    return used


def _update_extent(extent: dict[str, int], step: int, layer: int, heads: int) -> None:
    extent["step"] = max(extent["step"], int(step))
    extent["layer"] = max(extent["layer"], int(layer))
    extent["heads"] = max(extent["heads"], int(heads))


def _best_candidate_key(
    candidate_keys: Sequence[str],
    sums: dict[str, dict[tuple[int, int, int], float]],
    counts: dict[str, dict[tuple[int, int, int], int]],
    step: int,
    layer: int,
    head: int,
) -> str | None:
    best_key = None
    best_loss = None
    index = (int(step), int(layer), int(head))
    for key in candidate_keys:
        count = counts[key].get(index, 0)
        if count <= 0:
            continue
        loss = sums[key][index] / count
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_key = key
    return best_key


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SparseVideo STA mask search utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune = subparsers.add_parser("tune", help="Tune an STA mask strategy from SparseVideo search JSONL files")
    tune.add_argument("--search-dir", required=True)
    tune.add_argument("--output-file", required=True)
    tune.add_argument("--model", default=None, help="Optional backbone alias used for default timesteps/layers/heads")
    tune.add_argument("--skip-time-steps", type=int, default=None)
    tune.add_argument("--timesteps", type=int, default=None)
    tune.add_argument("--layers", type=int, default=None)
    tune.add_argument("--heads", type=int, default=None)
    tune.add_argument("--candidates", default=None, help="Semicolon-separated windows, e.g. '3,1,10;1,5,7'")
    tune.add_argument("--full-window", default=None)

    inspect = subparsers.add_parser("inspect", help="Summarize an STA mask strategy JSON file")
    inspect.add_argument("strategy_file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "tune":
        summary = tune_search_results(
            args.search_dir,
            args.output_file,
            model=args.model,
            candidates=None if args.candidates is None else parse_windows(args.candidates),
            full_window=None if args.full_window is None else parse_window(args.full_window),
            skip_time_steps=args.skip_time_steps,
            timesteps=args.timesteps,
            layers=args.layers,
            heads=args.heads,
        )
    else:
        summary = summarize_strategy(args.strategy_file)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
