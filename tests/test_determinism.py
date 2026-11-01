"""Tests for deterministic execution and reproducibility."""

from __future__ import annotations

import mne
import torch

from nneeg.contracts import get_contract
from nneeg.fingerprint import fingerprint_dataset
from nneeg.planner import generate_plan
from nneeg.preprocess import preprocess_raw
from nneeg.utils.determinism import set_seed


def test_preprocessing_deterministic_same_hardware(
    synthetic_50hz_noise_raw: mne.io.BaseRaw,
) -> None:
    """Verify same-hardware execution of the preprocessing pipeline produces identical outputs.

    Note: Cross-device floating-point execution across differing SIMD units/CPUs is not guaranteed
    to be bit-exact due to FMA (fused multiply-add) and threading differences, but same-hardware
    execution with fixed seeds must be reproducible within tight tolerance (atol=1e-6, rtol=1e-5).
    """
    contract = get_contract("labram-base")
    fp = fingerprint_dataset(synthetic_50hz_noise_raw)
    plan = generate_plan(fp, contract)

    # First run
    set_seed(42)
    out1 = preprocess_raw(synthetic_50hz_noise_raw, plan, contract=contract, seed=42)

    # Second run on identical input
    set_seed(42)
    out2 = preprocess_raw(synthetic_50hz_noise_raw, plan, contract=contract, seed=42)

    assert out1.shape == out2.shape
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)


def test_fingerprint_deterministic(
    synthetic_clean_raw: mne.io.BaseRaw,
) -> None:
    """Verify dataset fingerprinting is completely deterministic."""
    fp1 = fingerprint_dataset(synthetic_clean_raw)
    fp2 = fingerprint_dataset(synthetic_clean_raw)

    assert fp1.sampling_rate == fp2.sampling_rate
    assert fp1.duration_sec == fp2.duration_sec
    assert fp1.line_noise.detected_freq == fp2.line_noise.detected_freq
    assert fp1.line_noise.peak_snr_50 == fp2.line_noise.peak_snr_50
    assert fp1.amplitude_stats.mean == fp2.amplitude_stats.mean
    assert fp1.amplitude_stats.std == fp2.amplitude_stats.std
