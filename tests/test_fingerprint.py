"""Tests for EEG dataset fingerprinting module."""

from __future__ import annotations

import json

import mne
import numpy as np

from nneeg.fingerprint import (
    DatasetFingerprint,
    compute_amplitude_stats,
    fingerprint_dataset,
)


def test_fingerprint_clean_dataset(synthetic_clean_raw: mne.io.BaseRaw) -> None:
    """Verify fingerprinting clean synthetic EEG without prominent line noise."""
    fp = fingerprint_dataset(synthetic_clean_raw)
    assert isinstance(fp, DatasetFingerprint)
    assert fp.sampling_rate == 200.0
    assert fp.n_channels == 19
    assert fp.duration_sec == 10.0
    # Clean dataset should not detect dominant line noise peak
    assert fp.line_noise.detected_freq is None
    # High frequency noise proxy should be computed (fs=200 >= 160)
    assert fp.high_freq_noise.calculated is True
    assert fp.high_freq_noise.power_ratio is not None
    assert fp.high_freq_noise.skip_reason is None


def test_fingerprint_detects_50hz_line_noise(
    synthetic_50hz_noise_raw: mne.io.BaseRaw,
) -> None:
    """Verify accurate detection of 50 Hz line noise and harmonics."""
    fp = fingerprint_dataset(synthetic_50hz_noise_raw)
    assert fp.sampling_rate == 250.0
    assert fp.line_noise.detected_freq == 50.0
    assert fp.line_noise.peak_snr_50 > 3.0
    assert fp.line_noise.confidence > 0.5
    # Harmonic at 100 Hz should be detected (Nyquist is 125 Hz)
    assert 100.0 in fp.line_noise.harmonics_present


def test_fingerprint_detects_60hz_line_noise(
    synthetic_60hz_noise_raw: mne.io.BaseRaw,
) -> None:
    """Verify accurate detection of 60 Hz line noise."""
    fp = fingerprint_dataset(synthetic_60hz_noise_raw)
    assert fp.sampling_rate == 250.0
    assert fp.line_noise.detected_freq == 60.0
    assert fp.line_noise.peak_snr_60 > 3.0
    assert fp.line_noise.confidence > 0.5


def test_fingerprint_gracefully_skips_hf_proxy_on_low_fs(
    synthetic_low_fs_raw: mne.io.BaseRaw,
) -> None:
    """Verify fs < 160 Hz skips HF noise proxy and records explanation."""
    fp = fingerprint_dataset(synthetic_low_fs_raw)
    assert fp.sampling_rate == 128.0
    assert fp.high_freq_noise.calculated is False
    assert fp.high_freq_noise.power_ratio is None
    assert fp.high_freq_noise.skip_reason is not None
    assert "160.0 Hz threshold" in fp.high_freq_noise.skip_reason
    assert "Nyquist frequency (64.0 Hz)" in fp.high_freq_noise.skip_reason


def test_amplitude_stats_calculation() -> None:
    """Verify amplitude statistics match known numerical distribution."""
    data = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float64)
    stats = compute_amplitude_stats(data)
    assert stats.min_val == -1.0
    assert stats.max_val == 2.0
    assert stats.mean == 0.5
    assert stats.median == 0.5
    assert stats.iqr > 0
    assert stats.unit == "V"


def test_fingerprint_serialization(synthetic_clean_raw: mne.io.BaseRaw) -> None:
    """Verify JSON round-trip serialization of DatasetFingerprint."""
    fp = fingerprint_dataset(synthetic_clean_raw)
    json_str = fp.to_json()
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert parsed["sampling_rate"] == 200.0

    restored = DatasetFingerprint.from_dict(parsed)
    assert restored.sampling_rate == fp.sampling_rate
    assert restored.channel_names == fp.channel_names
    assert restored.line_noise.detected_freq == fp.line_noise.detected_freq
    assert restored.high_freq_noise.calculated == fp.high_freq_noise.calculated
