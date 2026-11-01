"""Pytest fixtures for synthetic EEG generation."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nneeg.contracts import LABRAM_CHANNELS


@pytest.fixture
def standard_19_channels() -> list[str]:
    """Standard 10-20 19-channel set used by LaBraM."""
    return list(LABRAM_CHANNELS)


@pytest.fixture
def synthetic_clean_raw(standard_19_channels: list[str]) -> mne.io.BaseRaw:
    """Create synthetic clean EEG raw data (200 Hz, 10 s, standard 19 channels)."""
    sfreq = 200.0
    duration = 10.0
    n_times = int(sfreq * duration)
    n_channels = len(standard_19_channels)

    t = np.arange(n_times) / sfreq
    # Known 10 Hz alpha wave with small noise in Volts (~10 uV)
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[i] = 10e-6 * np.sin(2 * np.pi * 10 * t + phase) + 1e-6 * rng.standard_normal(n_times)

    info = mne.create_info(ch_names=standard_19_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_50hz_noise_raw(standard_19_channels: list[str]) -> mne.io.BaseRaw:
    """Create synthetic EEG with 50 Hz line noise (250 Hz, 10 s)."""
    sfreq = 250.0
    duration = 10.0
    n_times = int(sfreq * duration)
    n_channels = len(standard_19_channels)

    t = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(n_channels):
        # 10 Hz signal + strong 50 Hz line noise + 100 Hz harmonic
        signal = 10e-6 * np.sin(2 * np.pi * 10 * t)
        noise_50 = 30e-6 * np.sin(2 * np.pi * 50 * t)
        noise_100 = 8e-6 * np.sin(2 * np.pi * 100 * t)
        data[i] = signal + noise_50 + noise_100 + 1e-6 * rng.standard_normal(n_times)

    info = mne.create_info(ch_names=standard_19_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_60hz_noise_raw(standard_19_channels: list[str]) -> mne.io.BaseRaw:
    """Create synthetic EEG with 60 Hz line noise (250 Hz, 10 s)."""
    sfreq = 250.0
    duration = 10.0
    n_times = int(sfreq * duration)
    n_channels = len(standard_19_channels)

    t = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    rng = np.random.default_rng(42)
    for i in range(n_channels):
        signal = 10e-6 * np.sin(2 * np.pi * 10 * t)
        noise_60 = 35e-6 * np.sin(2 * np.pi * 60 * t)
        data[i] = signal + noise_60 + 1e-6 * rng.standard_normal(n_times)

    info = mne.create_info(ch_names=standard_19_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_low_fs_raw() -> mne.io.BaseRaw:
    """Create synthetic EEG with fs = 128 Hz (< 160 Hz) to test HF proxy handling."""
    sfreq = 128.0
    duration = 10.0
    n_times = int(sfreq * duration)
    ch_names = ["Fp1", "Fp2", "C3", "C4"]

    rng = np.random.default_rng(42)
    data = 10e-6 * rng.standard_normal((len(ch_names), n_times))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_missing_channels_raw() -> mne.io.BaseRaw:
    """Create synthetic EEG missing F3 and F4 channels (17 channels)."""
    sfreq = 200.0
    duration = 10.0
    n_times = int(sfreq * duration)
    # Exclude F3, F4 from 19 standard channels
    subset = [ch for ch in LABRAM_CHANNELS if ch not in ("F3", "F4")]

    rng = np.random.default_rng(42)
    data = 10e-6 * rng.standard_normal((len(subset), n_times))
    info = mne.create_info(ch_names=subset, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def synthetic_insufficient_channels_raw() -> mne.io.BaseRaw:
    """Create synthetic EEG with only 2 channels (< 4 minimum for spherical interpolation)."""
    sfreq = 200.0
    duration = 10.0
    n_times = int(sfreq * duration)
    subset = ["Fp1", "Fp2"]

    rng = np.random.default_rng(42)
    data = 10e-6 * rng.standard_normal((len(subset), n_times))
    info = mne.create_info(ch_names=subset, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw
