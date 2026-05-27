from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsevideo._step_tracker import StepTracker


def test_step_tracker_counts_one_based_denoising_steps_from_kwargs():
    tracker = StepTracker("wan")

    tracker._hook(None, (), {"timestep": torch.tensor([999])})

    assert tracker.step == 1
    assert tracker.global_step == 1
    assert tracker.loop == 0
    assert tracker.timestep == 999.0

    tracker._hook(None, (), {"timestep": torch.tensor([999])})
    assert tracker.step == 1
    assert tracker.global_step == 1

    tracker._hook(None, (), {"timestep": torch.tensor([925])})
    assert tracker.step == 2
    assert tracker.global_step == 2
    assert tracker.loop == 0
    assert tracker.timestep == 925.0


def test_step_tracker_resets_local_step_for_segmented_scheduler_loop():
    tracker = StepTracker("wan", num_inference_steps_fn=lambda: 3)

    for timestep in (999, 500, 0):
        tracker._hook(None, (), {"timestep": torch.tensor([timestep])})

    assert tracker.step == 3
    assert tracker.global_step == 3
    assert tracker.loop == 0

    tracker._hook(None, (), {"timestep": torch.tensor([999])})

    assert tracker.step == 1
    assert tracker.global_step == 4
    assert tracker.loop == 1
    assert tracker.timestep == 999.0


def test_step_tracker_does_not_reset_before_expected_loop_length():
    tracker = StepTracker("wan", num_inference_steps_fn=lambda: 4)

    for timestep in (999, 500, 750):
        tracker._hook(None, (), {"timestep": torch.tensor([timestep])})

    assert tracker.step == 3
    assert tracker.global_step == 3
    assert tracker.loop == 0
    assert tracker.timestep == 750.0


def test_step_tracker_extracts_large_expanded_timestep_tensor():
    tracker = StepTracker("wan")

    tracker._hook(None, (), {"timestep": torch.full((4, 32), 999.0)})

    assert tracker.step == 1
    assert tracker.timestep == 999.0

    tracker._hook(None, (), {"timestep": torch.full((4, 32), 999.0)})
    assert tracker.step == 1

    tracker._hook(None, (), {"timestep": torch.full((4, 32), 925.0)})
    assert tracker.step == 2
    assert tracker.timestep == 925.0


def test_step_tracker_extracts_large_positional_timestep_tensor_for_wan():
    tracker = StepTracker("wan")

    tracker._hook(None, (object(), torch.full((2, 64), 1000.0)), {})

    assert tracker.step == 1
    assert tracker.timestep == 1000.0


def test_step_tracker_uses_ltx_video_max_for_mixed_conditioning_timestep_tensor():
    tracker = StepTracker("ltx_video")

    tracker._hook(None, (object(), object(), torch.tensor([0.0, 999.0, 999.0])), {})

    assert tracker.step == 1
    assert tracker.timestep == 999.0

    tracker._hook(None, (object(), object(), torch.tensor([0.0, 999.0, 999.0])), {})
    assert tracker.step == 1

    tracker._hook(None, (object(), object(), torch.tensor([0.0, 925.0, 925.0])), {})
    assert tracker.step == 2
    assert tracker.timestep == 925.0
