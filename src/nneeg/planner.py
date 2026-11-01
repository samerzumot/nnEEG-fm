"""Rule-based pipeline configuration and planning for frozen EEG Foundation Models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from nneeg.contracts import FMContract
from nneeg.fingerprint import DatasetFingerprint
from nneeg.utils.channels import CANONICAL_1020_UPPER, clean_channel_name, match_channel_to_target


@dataclass
class PlanStep:
    """A discrete preprocessing step with parameters and reasoning log."""

    step_name: str
    action: str
    rule_name: str
    parameters: dict[str, Any]
    reason: str


@dataclass
class PipelinePlan:
    """Complete, deterministic preprocessing plan matching a dataset to an FM contract."""

    contract_name: str
    dataset_summary: dict[str, Any]
    target_fs: float
    resample_step: PlanStep
    notch_step: PlanStep
    bandpass_step: PlanStep
    channel_step: PlanStep
    scaling_step: PlanStep
    windowing_step: PlanStep
    normalization_step: PlanStep
    steps: list[PlanStep]

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize plan to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelinePlan:
        """Reconstruct plan from dictionary."""
        d = dict(data)
        d["resample_step"] = PlanStep(**d["resample_step"])
        d["notch_step"] = PlanStep(**d["notch_step"])
        d["bandpass_step"] = PlanStep(**d["bandpass_step"])
        d["channel_step"] = PlanStep(**d["channel_step"])
        d["scaling_step"] = PlanStep(**d["scaling_step"])
        d["windowing_step"] = PlanStep(**d["windowing_step"])
        d["normalization_step"] = PlanStep(**d["normalization_step"])
        d["steps"] = [PlanStep(**s) for s in d["steps"]]
        return cls(**d)


def plan_resampling(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 1: Resample via rational resample_poly to contract fs."""
    current_fs = fingerprint.sampling_rate
    target_fs = contract.target_fs

    if abs(current_fs - target_fs) > 1e-3:
        return PlanStep(
            step_name="resample",
            action="resample_poly",
            rule_name="rule_rational_resampling",
            parameters={
                "source_fs": current_fs,
                "target_fs": target_fs,
                "method": "rational_polyphase",
            },
            reason=(
                f"Dataset sampling rate ({current_fs:.1f} Hz) differs from FM contract target "
                f"({target_fs:.1f} Hz). Planning rational polyphase resampling to eliminate temporal mismatch."
            ),
        )
    else:
        return PlanStep(
            step_name="resample",
            action="none",
            rule_name="rule_rational_resampling",
            parameters={"source_fs": current_fs, "target_fs": target_fs},
            reason=(
                f"Dataset sampling rate ({current_fs:.1f} Hz) matches FM contract target "
                f"({target_fs:.1f} Hz). Resampling omitted."
            ),
        )


def plan_notch_filter(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 2: Notch filter at detected line frequency and harmonics."""
    detected = fingerprint.line_noise.detected_freq
    nyquist = contract.target_fs / 2.0

    if detected is not None:
        # Find all harmonics strictly below Nyquist
        freqs_to_notch: list[float] = []
        mult = 1
        while mult * detected < nyquist:
            freqs_to_notch.append(round(mult * detected, 1))
            mult += 1

        snr = (
            fingerprint.line_noise.peak_snr_50
            if detected == 50.0
            else fingerprint.line_noise.peak_snr_60
        )
        return PlanStep(
            step_name="notch_filter",
            action="apply_notch",
            rule_name="rule_line_noise_notch",
            parameters={
                "frequencies": freqs_to_notch,
                "notch_widths": 2.0,
                "detected_freq": detected,
                "peak_snr": snr,
            },
            reason=(
                f"Prominent line noise peak detected at {detected:.1f} Hz (peak SNR={snr:.1f}, "
                f"confidence={fingerprint.line_noise.confidence:.2f}). Planning zero-phase notch filter "
                f"at {freqs_to_notch} Hz (strictly below Nyquist {nyquist:.1f} Hz)."
            ),
        )
    else:
        return PlanStep(
            step_name="notch_filter",
            action="none",
            rule_name="rule_line_noise_notch",
            parameters={
                "peak_snr_50": fingerprint.line_noise.peak_snr_50,
                "peak_snr_60": fingerprint.line_noise.peak_snr_60,
            },
            reason=(
                f"No prominent line noise peak detected above baseline (50 Hz SNR="
                f"{fingerprint.line_noise.peak_snr_50:.1f}, 60 Hz SNR="
                f"{fingerprint.line_noise.peak_snr_60:.1f}). Notch filter omitted to avoid unnecessary phase/amplitude distortion."
            ),
        )


def plan_bandpass_filter(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 3: Band-pass per FM contract (NOT per task)."""
    l_freq = contract.bandpass_low
    h_freq = contract.bandpass_high
    nyquist = contract.target_fs / 2.0

    if l_freq is None and h_freq is None:
        return PlanStep(
            step_name="bandpass_filter",
            action="none",
            rule_name="rule_contract_bandpass",
            parameters={},
            reason="FM contract specifies no bandpass filtering requirements.",
        )

    # Check Nyquist compliance
    actual_h_freq = h_freq
    adjusted_for_nyquist = False
    if h_freq is not None and h_freq >= nyquist:
        actual_h_freq = round(nyquist - 1.0, 1)
        adjusted_for_nyquist = True

    reason = (
        f"Applying FM contract pretraining band-pass filter [{l_freq}, {actual_h_freq}] Hz "
        f"(zero-phase FIR filter) to match foundation model spectral representation."
    )
    if adjusted_for_nyquist:
        reason += (
            f" Note: Upper cutoff clamped from {h_freq} Hz to {actual_h_freq} Hz to comply with "
            f"Nyquist limit ({nyquist:.1f} Hz) at target sampling rate."
        )

    return PlanStep(
        step_name="bandpass_filter",
        action="apply_bandpass",
        rule_name="rule_contract_bandpass",
        parameters={
            "l_freq": l_freq,
            "h_freq": actual_h_freq,
            "filter_type": "fir_zero_phase",
        },
        reason=reason,
    )


def plan_channel_handling(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 4: Channel handling ladder: native subset -> spherical-spline interpolation."""
    expected_channels = list(contract.expected_channels)
    dataset_channels = fingerprint.channel_names

    # Map dataset channels to contract expected channels
    matched_map: dict[str, str] = {}
    for d_ch in dataset_channels:
        matched = match_channel_to_target(d_ch, expected_channels)
        if matched:
            matched_map[matched] = d_ch

    missing = [ch for ch in expected_channels if ch not in matched_map]

    # Ladder Step 1: Native subset
    if not missing:
        return PlanStep(
            step_name="channel_handling",
            action="native_subset",
            rule_name="rule_channel_ladder",
            parameters={
                "target_channels": expected_channels,
                "channel_mapping": matched_map,
            },
            reason=(
                f"All {len(expected_channels)} expected contract channels are present natively "
                f"in the dataset. Selecting native subset and aligning to canonical FM channel order."
            ),
        )

    # Ladder Step 2: Spherical-spline interpolation
    # Count how many existing channels map to standard 10-20 system
    dataset_std_mapped = [
        ch for ch in dataset_channels if clean_channel_name(ch) in CANONICAL_1020_UPPER
    ]

    if len(dataset_std_mapped) >= 4:
        return PlanStep(
            step_name="channel_handling",
            action="spherical_spline_interpolation",
            rule_name="rule_channel_ladder",
            parameters={
                "target_channels": expected_channels,
                "missing_channels": missing,
                "available_standard_channels": dataset_std_mapped,
                "channel_mapping": matched_map,
                "interpolation_method": "spline",
            },
            reason=(
                f"Dataset is missing {len(missing)} expected contract channels: {missing}. "
                f"Ladder step 2 fired: Executing spherical-spline interpolation using {len(dataset_std_mapped)} "
                f"available standard 10-20 electrodes to reconstruct the full canonical {len(expected_channels)}-channel montage."
            ),
        )

    # Ladder Step 3: Insufficient channels for spherical interpolation
    return PlanStep(
        step_name="channel_handling",
        action="error_insufficient_channels",
        rule_name="rule_channel_ladder",
        parameters={
            "target_channels": expected_channels,
            "missing_channels": missing,
            "available_standard_count": len(dataset_std_mapped),
        },
        reason=(
            f"Dataset has only {len(dataset_std_mapped)} standard 10-20 channels available, which is below "
            f"the minimum of 4 required for spherical-spline interpolation. Cannot satisfy FM contract."
        ),
    )


def plan_amplitude_scaling(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 5: Scale amplitude to contract unit (e.g. Volts to microvolts)."""
    target_unit = contract.amplitude_unit.lower()
    source_unit = fingerprint.amplitude_stats.unit.lower()

    if target_unit == "uv" and source_unit == "v":
        return PlanStep(
            step_name="amplitude_scaling",
            action="scale_uV",
            rule_name="rule_amplitude_unit",
            parameters={"scale_factor": 1e6, "from_unit": "V", "to_unit": "uV"},
            reason=(
                "FM contract expects amplitude in microvolts (uV). Scaling MNE raw Volts "
                "by factor 1,000,000 (1e6) to match pretrained dynamic range."
            ),
        )
    else:
        return PlanStep(
            step_name="amplitude_scaling",
            action="none",
            rule_name="rule_amplitude_unit",
            parameters={"unit": target_unit},
            reason=f"Dataset is already in expected amplitude unit '{target_unit}'. Scaling omitted.",
        )


def plan_windowing(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 6: Segment signal into FM contract input windows and patches."""
    window_samples = contract.window_samples
    target_fs = contract.target_fs
    window_sec = window_samples / target_fs
    patch_samples = contract.patch_samples

    return PlanStep(
        step_name="windowing",
        action="sliding_window",
        rule_name="rule_contract_windowing",
        parameters={
            "window_samples": window_samples,
            "window_sec": round(window_sec, 3),
            "patch_samples": patch_samples,
            "stride_samples": window_samples,  # non-overlapping default
        },
        reason=(
            f"Segmenting continuous signal into non-overlapping windows of {window_samples} samples "
            f"({window_sec:.2f} s at {target_fs} Hz) with {patch_samples}-sample patches matching FM token architecture."
        ),
    )


def plan_normalization(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PlanStep:
    """Rule 7: Normalization according to FM contract."""
    norm_type = contract.normalization
    return PlanStep(
        step_name="normalization",
        action=norm_type,
        rule_name="rule_contract_normalization",
        parameters={"method": norm_type, "epsilon": 1e-8},
        reason=(
            f"Applying '{norm_type}' across the temporal dimension per channel "
            f"(zero mean, unit variance) as specified in FM pretraining contract."
        ),
    )


def generate_plan(
    fingerprint: DatasetFingerprint,
    contract: FMContract,
) -> PipelinePlan:
    """Generate a deterministic PipelinePlan matching a dataset fingerprint to an FM contract.

    Parameters
    ----------
    fingerprint : DatasetFingerprint
        Dataset fingerprint capturing sampling rate, line noise, and montage.
    contract : FMContract
        Target foundation model preprocessing contract.

    Returns
    -------
    PipelinePlan
        Structured plan detailing every preprocessing step with rule justifications.
    """
    resample_step = plan_resampling(fingerprint, contract)
    notch_step = plan_notch_filter(fingerprint, contract)
    bandpass_step = plan_bandpass_filter(fingerprint, contract)
    channel_step = plan_channel_handling(fingerprint, contract)
    scaling_step = plan_amplitude_scaling(fingerprint, contract)
    windowing_step = plan_windowing(fingerprint, contract)
    norm_step = plan_normalization(fingerprint, contract)

    steps = [
        resample_step,
        channel_step,
        resample_step if resample_step.action != "none" else None,
        notch_step if notch_step.action != "none" else None,
        bandpass_step if bandpass_step.action != "none" else None,
        scaling_step if scaling_step.action != "none" else None,
        windowing_step,
        norm_step,
    ]
    # Filter out None steps
    active_steps = [s for s in steps if s is not None]

    summary = {
        "dataset_fs": fingerprint.sampling_rate,
        "n_channels": fingerprint.n_channels,
        "duration_sec": fingerprint.duration_sec,
        "detected_line_noise": fingerprint.line_noise.detected_freq,
        "amplitude_unit": fingerprint.amplitude_stats.unit,
    }

    return PipelinePlan(
        contract_name=contract.name,
        dataset_summary=summary,
        target_fs=contract.target_fs,
        resample_step=resample_step,
        notch_step=notch_step,
        bandpass_step=bandpass_step,
        channel_step=channel_step,
        scaling_step=scaling_step,
        windowing_step=windowing_step,
        normalization_step=norm_step,
        steps=active_steps,
    )
