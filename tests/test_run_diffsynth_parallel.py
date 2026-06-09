from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_diffsynth_parallel.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("sparsevideo_run_diffsynth_parallel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skip_existing_requires_exact_current_output_file(tmp_path):
    runner = _load_runner_module()
    cmd = (
        "python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method radial "
        f"--prompt-file example/t2v/1.txt --output-dir {tmp_path}"
    )

    old_output = tmp_path / "wan22-t2v-a14b" / "radial" / "seed0_704x1248_121f.mp4"
    old_output.parent.mkdir(parents=True)
    old_output.write_bytes(b"old")

    assert not runner._output_exists(cmd, "wan22-t2v-a14b", "radial")

    expected = tmp_path / "wan22-t2v-a14b" / "radial" / "seed0_480x832_81f.mp4"
    expected.write_bytes(b"current")

    assert runner._output_exists(cmd, "wan22-t2v-a14b", "radial")


def test_skip_existing_honors_explicit_output_shape_and_file(tmp_path):
    runner = _load_runner_module()
    shaped_cmd = (
        "python scripts/infer_diffsynth.py --model wan22-t2v-a14b --method svg1 "
        "--height 704 --width 1248 --num-frames 121 --seed 3 "
        f"--prompt-file example/t2v/1.txt --output-dir {tmp_path}"
    )
    shaped_output = tmp_path / "wan22-t2v-a14b" / "svg1" / "seed3_704x1248_121f.mp4"
    shaped_output.parent.mkdir(parents=True)
    shaped_output.write_bytes(b"shaped")
    assert runner._output_exists(shaped_cmd, "wan22-t2v-a14b", "svg1")

    explicit_output = tmp_path / "manual.mp4"
    explicit_cmd = (
        "python scripts/infer_diffsynth.py --model longcat-video --method radial "
        f"--prompt-file example/t2v/1.txt --longcat-video example/animate/1.mp4 --output-file {explicit_output}"
    )
    assert not runner._output_exists(explicit_cmd, "longcat-video", "radial")
    explicit_output.write_bytes(b"manual")
    assert runner._output_exists(explicit_cmd, "longcat-video", "radial")
