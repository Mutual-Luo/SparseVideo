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
    assert tracker.timestep == 999.0

    tracker._hook(None, (), {"timestep": torch.tensor([999])})
    assert tracker.step == 1

    tracker._hook(None, (), {"timestep": torch.tensor([925])})
    assert tracker.step == 2
    assert tracker.timestep == 925.0


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
