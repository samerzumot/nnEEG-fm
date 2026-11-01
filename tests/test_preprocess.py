"""Tests for MNE preprocessing execution engine."""

from __future__ import annotations

import mne
import pytest
import torch

from nneeg.contracts import get_contract
from nneeg.fingerprint import fingerprint_dataset
from nneeg.planner import generate_plan
from nneeg.preprocess import preprocess_raw


def test_preprocess_output_tensor_shape_and_dtype(
    synthetic_clean_raw: mne.io.BaseRaw,
) -> None:
    """Verify output tensor has exact (N, C, T) shape, float32 dtype, and contract channels."""
    contract = get_contract("labram-base")
    fp = fingerprint_dataset(synthetic_clean_raw)
    plan = generate_plan(fp, contract)

    tensor = preprocess_raw(synthetic_clean_raw, plan, contract=contract)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32

    # Shape: (N, C, T)
    # Duration is 10 s at 200 Hz = 2000 samples.
    # window_samples = 1600 (8.0 s), non-overlapping stride = 1600.
    # N = (2000 - 1600) // 1600 + 1 = 1 window.
    # C = 19 channels.
    # T = 1600 samples.
    assert tensor.ndim == 3
    n_windows, n_channels, n_times = tensor.shape
    assert n_windows == 1
    assert n_channels == len(contract.expected_channels)
    assert n_times == contract.window_samples


def test_preprocess_channel_interpolation_ladder(
    synthetic_missing_channels_raw: mne.io.BaseRaw,
) -> None:
    """Verify spherical-spline interpolation ladder step produces full 19 contract channels."""
    contract = get_contract("labram-base")
    # Dataset only has 17 channels (missing F3, F4)
    assert len(synthetic_missing_channels_raw.ch_names) == 17

    fp = fingerprint_dataset(synthetic_missing_channels_raw)
    plan = generate_plan(fp, contract)
    assert plan.channel_step.action == "spherical_spline_interpolation"

    tensor = preprocess_raw(synthetic_missing_channels_raw, plan, contract=contract)
    assert tensor.shape[1] == 19  # Full contract channels restored via interpolation
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_preprocess_insufficient_channels_raises(
    synthetic_insufficient_channels_raw: mne.io.BaseRaw,
) -> None:
    """Verify preprocessing raises ValueError when channel ladder cannot satisfy contract."""
    contract = get_contract("labram-base")
    fp = fingerprint_dataset(synthetic_insufficient_channels_raw)
    plan = generate_plan(fp, contract)

    with pytest.raises(ValueError, match="Cannot preprocess EEG"):
        preprocess_raw(synthetic_insufficient_channels_raw, plan, contract=contract)


def test_preprocess_normalization_per_channel(
    synthetic_clean_raw: mne.io.BaseRaw,
) -> None:
    """Verify z-score normalization ensures zero mean and unit variance per channel."""
    contract = get_contract("labram-base")
    fp = fingerprint_dataset(synthetic_clean_raw)
    plan = generate_plan(fp, contract)

    tensor = preprocess_raw(synthetic_clean_raw, plan, contract=contract)

    # Across time dimension (dim=-1)
    means = tensor.mean(dim=-1)
    stds = tensor.std(dim=-1)

    torch.testing.assert_close(
        means,
        torch.zeros_like(means),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        stds,
        torch.ones_like(stds),
        atol=1e-3,
        rtol=1e-3,
    )
