"""Tests for rule-based pipeline planner."""

from __future__ import annotations

import mne

from nneeg.contracts import FMContract, get_contract
from nneeg.fingerprint import fingerprint_dataset
from nneeg.planner import PipelinePlan, generate_plan


def test_planner_rules_fire_for_resampling_and_notch(
    synthetic_50hz_noise_raw: mne.io.BaseRaw,
) -> None:
    """Verify resampling and notch rules fire when fs differs and line noise is detected."""
    fp = fingerprint_dataset(synthetic_50hz_noise_raw)
    contract = get_contract("labram-base")

    plan = generate_plan(fp, contract)
    assert isinstance(plan, PipelinePlan)

    # Rule 1: Resampling rule must fire (dataset fs=250 -> contract target_fs=200)
    assert plan.resample_step.action == "resample_poly"
    assert plan.resample_step.parameters["source_fs"] == 250.0
    assert plan.resample_step.parameters["target_fs"] == 200.0
    assert "differs from FM contract target" in plan.resample_step.reason

    # Rule 2: Line noise notch rule must fire for 50 Hz
    assert plan.notch_step.action == "apply_notch"
    assert plan.notch_step.parameters["detected_freq"] == 50.0
    assert 50.0 in plan.notch_step.parameters["frequencies"]
    assert "Prominent line noise peak detected" in plan.notch_step.reason

    # Rule 3: Bandpass filter per contract
    assert plan.bandpass_step.action == "apply_bandpass"
    assert plan.bandpass_step.parameters["l_freq"] == 0.1
    assert plan.bandpass_step.parameters["h_freq"] == 75.0
    assert "Applying FM contract pretraining band-pass filter" in plan.bandpass_step.reason

    # Rule 4: Native subset channel handling
    assert plan.channel_step.action == "native_subset"
    assert "All 19 expected contract channels are present" in plan.channel_step.reason

    # Rule 5: Amplitude scaling
    assert plan.scaling_step.action == "scale_uV"
    assert plan.scaling_step.parameters["scale_factor"] == 1e6

    # Rule 6: Windowing
    assert plan.windowing_step.action == "sliding_window"
    assert plan.windowing_step.parameters["window_samples"] == 1600


def test_planner_omits_resampling_and_notch_when_unneeded(
    synthetic_clean_raw: mne.io.BaseRaw,
) -> None:
    """Verify resampling and notch rules are omitted when fs matches and data is clean."""
    fp = fingerprint_dataset(synthetic_clean_raw)
    contract = get_contract("labram-base")

    plan = generate_plan(fp, contract)

    # Resampling should be omitted (200 Hz == 200 Hz)
    assert plan.resample_step.action == "none"
    assert "matches FM contract target" in plan.resample_step.reason

    # Notch filter should be omitted (no line noise peak)
    assert plan.notch_step.action == "none"
    assert "omitted to avoid unnecessary" in plan.notch_step.reason


def test_planner_channel_ladder_spherical_spline(
    synthetic_missing_channels_raw: mne.io.BaseRaw,
) -> None:
    """Verify ladder step 2 (spherical-spline interpolation) fires when channels are missing."""
    fp = fingerprint_dataset(synthetic_missing_channels_raw)
    contract = get_contract("labram-base")

    plan = generate_plan(fp, contract)
    assert plan.channel_step.action == "spherical_spline_interpolation"
    assert "F3" in plan.channel_step.parameters["missing_channels"]
    assert "F4" in plan.channel_step.parameters["missing_channels"]
    assert "Executing spherical-spline interpolation" in plan.channel_step.reason


def test_planner_channel_ladder_insufficient_channels(
    synthetic_insufficient_channels_raw: mne.io.BaseRaw,
) -> None:
    """Verify ladder step 3 (error) fires when fewer than 4 standard channels are available."""
    fp = fingerprint_dataset(synthetic_insufficient_channels_raw)
    contract = get_contract("labram-base")

    plan = generate_plan(fp, contract)
    assert plan.channel_step.action == "error_insufficient_channels"
    assert "below the minimum of 4 required" in plan.channel_step.reason


def test_planner_bandpass_nyquist_clamping() -> None:
    """Verify upper bandpass cutoff clamps when contract frequency exceeds Nyquist."""
    # Create contract with 120 Hz high cutoff but target_fs = 200 Hz (Nyquist = 100 Hz)
    contract = FMContract(
        name="high-freq-model",
        target_fs=200.0,
        bandpass_low=0.5,
        bandpass_high=120.0,
        notch_freqs=(),
        patch_samples=200,
        window_samples=1600,
        expected_channels=("FP1", "FP2", "C3", "C4"),
        verification_notes={"test": "Test Nyquist clamp."},
    )
    # Dummy fingerprint with 200 Hz
    from nneeg.fingerprint import (
        AmplitudeStats,
        DatasetFingerprint,
        HighFreqNoiseProxy,
        LineNoiseDetection,
    )

    fp = DatasetFingerprint(
        sampling_rate=200.0,
        all_sampling_rates=[200.0],
        channel_names=["FP1", "FP2", "C3", "C4"],
        standardized_channels=["FP1", "FP2", "C3", "C4"],
        n_channels=4,
        duration_sec=10.0,
        file_count=1,
        line_noise=LineNoiseDetection(None, 1.0, 1.0, [], 0.0),
        amplitude_stats=AmplitudeStats(-1.0, 1.0, 0.0, 0.1, 0.0, 0.1, "V"),
        high_freq_noise=HighFreqNoiseProxy(None, False, "test"),
        raw_file_formats=[".fif"],
    )
    plan = generate_plan(fp, contract)
    assert plan.bandpass_step.parameters["h_freq"] == 99.0  # Nyquist (100) - 1.0
    assert "clamped from 120.0 Hz to 99.0 Hz" in plan.bandpass_step.reason


def test_plan_serialization(synthetic_clean_raw: mne.io.BaseRaw) -> None:
    """Verify JSON serialization round-trip of PipelinePlan."""
    fp = fingerprint_dataset(synthetic_clean_raw)
    contract = get_contract("labram-base")
    plan = generate_plan(fp, contract)

    json_str = plan.to_json()
    assert isinstance(json_str, str)

    restored = PipelinePlan.from_dict(plan.to_dict())
    assert restored.contract_name == plan.contract_name
    assert restored.target_fs == plan.target_fs
    assert restored.channel_step.action == plan.channel_step.action
    assert len(restored.steps) == len(plan.steps)
