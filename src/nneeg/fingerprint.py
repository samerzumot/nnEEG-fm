"""Dataset fingerprinting module for EEG data.

Detects sampling rates, channel nomenclature/montage, duration, line-noise frequencies
(50/60 Hz and harmonics), amplitude distribution statistics, and high-frequency noise proxies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.signal as signal

from nneeg.utils.channels import clean_channel_name
from nneeg.utils.io import discover_eeg_files, load_raw_file


@dataclass
class AmplitudeStats:
    """Statistical summary of signal amplitudes."""

    min_val: float
    max_val: float
    mean: float
    std: float
    median: float
    iqr: float
    unit: str  # e.g., 'V' (MNE default) or 'uV'


@dataclass
class LineNoiseDetection:
    """Line noise peak detection results at 50 Hz and 60 Hz."""

    detected_freq: float | None  # 50.0, 60.0, or None if no prominent peak
    peak_snr_50: float
    peak_snr_60: float
    harmonics_present: list[float]
    confidence: float  # Score between 0.0 and 1.0


@dataclass
class HighFreqNoiseProxy:
    """High-frequency noise proxy (>80 Hz power ratio).

    Skipped gracefully when sampling rate < 160 Hz (Nyquist < 80 Hz).
    """

    power_ratio: float | None
    calculated: bool
    skip_reason: str | None = None


@dataclass
class DatasetFingerprint:
    """Comprehensive dataset fingerprint capturing metadata and signal characteristics."""

    sampling_rate: float
    all_sampling_rates: list[float]
    channel_names: list[str]
    standardized_channels: list[str]
    n_channels: int
    duration_sec: float
    file_count: int
    line_noise: LineNoiseDetection
    amplitude_stats: AmplitudeStats
    high_freq_noise: HighFreqNoiseProxy
    raw_file_formats: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert fingerprint to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize fingerprint to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFingerprint:
        """Construct DatasetFingerprint from dictionary."""
        d = dict(data)
        d["line_noise"] = LineNoiseDetection(**d["line_noise"])
        d["amplitude_stats"] = AmplitudeStats(**d["amplitude_stats"])
        d["high_freq_noise"] = HighFreqNoiseProxy(**d["high_freq_noise"])
        return cls(**d)


def _compute_welch_psd(
    data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average Welch PSD across channels.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Frequencies array and mean PSD array across channels.
    """
    n_times = data.shape[-1]
    nperseg = min(n_times, int(4 * sfreq))
    if nperseg < 8:
        nperseg = n_times

    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    # Average across channels if 2D
    if psd.ndim > 1:
        mean_psd = np.mean(psd, axis=0)
    else:
        mean_psd = psd
    return freqs, mean_psd


def _get_peak_snr(
    f0: float,
    freqs: np.ndarray,
    psd: np.ndarray,
    half_width: float = 1.0,
    base_low: float = 5.0,
    base_high: float = 2.0,
) -> float:
    """Calculate Signal-to-Noise ratio of a spectral peak relative to neighboring baseline.

    Parameters
    ----------
    f0 : float
        Target peak center frequency in Hz.
    freqs : np.ndarray
        Frequency bins.
    psd : np.ndarray
        Power spectral density values.
    half_width : float
        Peak half-bandwidth in Hz.
    base_low : float
        Baseline outer limit in Hz.
    base_high : float
        Baseline inner guard limit in Hz.

    Returns
    -------
    float
        Peak power divided by median baseline power.
    """
    nyquist = freqs[-1]
    if f0 >= nyquist:
        return 0.0

    peak_mask = (freqs >= max(0.0, f0 - half_width)) & (freqs <= min(nyquist, f0 + half_width))
    base_mask = ((freqs >= max(0.0, f0 - base_low)) & (freqs <= max(0.0, f0 - base_high))) | (
        (freqs >= min(nyquist, f0 + base_high)) & (freqs <= min(nyquist, f0 + base_low))
    )

    if not np.any(peak_mask) or not np.any(base_mask):
        return 0.0

    peak_val = float(np.max(psd[peak_mask]))
    base_val = float(np.median(psd[base_mask]))
    if base_val <= 1e-18:
        base_val = 1e-18

    return peak_val / base_val


def detect_line_noise(
    data: np.ndarray,
    sfreq: float,
    snr_threshold: float = 3.0,
) -> LineNoiseDetection:
    """Detect line noise frequency (50 Hz or 60 Hz) and harmonics via Welch peak analysis.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    snr_threshold : float
        Minimum peak SNR to declare line noise present.

    Returns
    -------
    LineNoiseDetection
        Object containing detected line noise frequency, SNRs, harmonics, and confidence.
    """
    freqs, mean_psd = _compute_welch_psd(data, sfreq)
    nyquist = sfreq / 2.0

    snr_50 = _get_peak_snr(50.0, freqs, mean_psd) if nyquist > 50.0 else 0.0
    snr_60 = _get_peak_snr(60.0, freqs, mean_psd) if nyquist > 60.0 else 0.0

    detected: float | None = None
    confidence = 0.0
    harmonics: list[float] = []

    # Check 50 Hz vs 60 Hz dominance
    if snr_50 >= snr_threshold and snr_50 > 1.5 * snr_60:
        detected = 50.0
        # Check 100 Hz harmonic if within Nyquist
        if nyquist > 100.0:
            snr_100 = _get_peak_snr(100.0, freqs, mean_psd)
            if snr_100 >= 2.0:
                harmonics.append(100.0)
        confidence = float(min(1.0, (snr_50 - snr_threshold) / (snr_threshold * 2.0) + 0.5))
    elif snr_60 >= snr_threshold and snr_60 > 1.5 * snr_50:
        detected = 60.0
        # Check 120 Hz harmonic if within Nyquist
        if nyquist > 120.0:
            snr_120 = _get_peak_snr(120.0, freqs, mean_psd)
            if snr_120 >= 2.0:
                harmonics.append(120.0)
        confidence = float(min(1.0, (snr_60 - snr_threshold) / (snr_threshold * 2.0) + 0.5))

    return LineNoiseDetection(
        detected_freq=detected,
        peak_snr_50=round(snr_50, 2),
        peak_snr_60=round(snr_60, 2),
        harmonics_present=harmonics,
        confidence=round(confidence, 3),
    )


def compute_hf_noise_proxy(
    data: np.ndarray,
    sfreq: float,
    cutoff_hz: float = 80.0,
) -> HighFreqNoiseProxy:
    """Compute high-frequency noise proxy as the ratio of high-frequency power to total power.

    Gracefully skips calculation when sfreq < 160 Hz (since Nyquist < 80 Hz).

    Parameters
    ----------
    data : np.ndarray
        Signal array of shape (n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    cutoff_hz : float
        High-frequency cutoff boundary in Hz (default 80 Hz).

    Returns
    -------
    HighFreqNoiseProxy
        Proxy result with calculation status and skip reason if applicable.
    """
    if sfreq < 160.0:
        nyquist = sfreq / 2.0
        reason = (
            f"fs={sfreq:.1f} Hz < 160.0 Hz threshold: Nyquist frequency ({nyquist:.1f} Hz) "
            f"is below the {cutoff_hz:.1f} Hz cutoff; HF noise proxy skipped to avoid aliasing."
        )
        return HighFreqNoiseProxy(power_ratio=None, calculated=False, skip_reason=reason)

    freqs, mean_psd = _compute_welch_psd(data, sfreq)
    total_mask = (freqs >= 1.0) & (freqs <= sfreq / 2.0)
    hf_mask = (freqs >= cutoff_hz) & (freqs <= sfreq / 2.0)

    # NumPy 2.x compatibility: np.trapezoid replaced np.trapz
    integrate_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if integrate_trapz is None:
        import scipy.integrate
        integrate_trapz = scipy.integrate.trapezoid

    total_power = float(integrate_trapz(mean_psd[total_mask], freqs[total_mask]))
    hf_power = (
        float(integrate_trapz(mean_psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    )

    if total_power <= 1e-18:
        ratio = 0.0
    else:
        ratio = float(hf_power / total_power)

    return HighFreqNoiseProxy(
        power_ratio=round(ratio, 6),
        calculated=True,
        skip_reason=None,
    )


def compute_amplitude_stats(data: np.ndarray) -> AmplitudeStats:
    """Compute amplitude summary statistics across all channels and samples.

    Parameters
    ----------
    data : np.ndarray
        EEG data array in Volts.

    Returns
    -------
    AmplitudeStats
        Distribution stats including min, max, mean, std, median, and IQR.
    """
    flat = data.ravel()
    # Downsample if exceedingly large for instant computation
    if len(flat) > 500_000:
        sample_indices = np.linspace(0, len(flat) - 1, 500_000, dtype=int)
        sample = flat[sample_indices]
    else:
        sample = flat

    q25, q50, q75 = np.percentile(sample, [25.0, 50.0, 75.0])
    return AmplitudeStats(
        min_val=float(np.min(sample)),
        max_val=float(np.max(sample)),
        mean=float(np.mean(sample)),
        std=float(np.std(sample)),
        median=float(q50),
        iqr=float(q75 - q25),
        unit="V",  # MNE standard representation
    )


def fingerprint_dataset(
    source: str | Path | mne.io.BaseRaw,
) -> DatasetFingerprint:
    """Extract a comprehensive fingerprint from an EEG dataset or in-memory Raw object.

    Parameters
    ----------
    source : str | Path | mne.io.BaseRaw
        File path (.edf, .bdf, .vhdr, .set, .fif), directory path containing EEG files,
        or an MNE BaseRaw instance.

    Returns
    -------
    DatasetFingerprint
        Full dataset fingerprint with sampling rates, montage, line noise, and stats.
    """
    if isinstance(source, mne.io.BaseRaw):
        raws = [source]
        file_formats = ["in_memory_raw"]
    else:
        paths = discover_eeg_files(source)
        if not paths:
            raise FileNotFoundError(f"No supported EEG files found at: {source}")
        raws = [load_raw_file(p, preload=True) for p in paths]
        file_formats = sorted({p.suffix.lower() for p in paths})

    # Collect sampling rates and durations
    sampling_rates = [float(r.info["sfreq"]) for r in raws]
    durations = [float(r.n_times / r.info["sfreq"]) for r in raws]
    total_duration = sum(durations)

    # Use first raw (or representative raw) for channel and spectral analysis
    primary_raw = raws[0]
    primary_fs = float(primary_raw.info["sfreq"])
    raw_channels = list(primary_raw.ch_names)
    standardized = [clean_channel_name(ch) for ch in raw_channels]

    # Combine data across raws (or sample from primary) for stats & Welch PSD
    # To keep computation fast and deterministic, sample up to 60 seconds from each recording
    sampled_segments: list[np.ndarray] = []
    for r in raws:
        max_samples = min(r.n_times, int(60 * r.info["sfreq"]))
        d = r.get_data(start=0, stop=max_samples)
        sampled_segments.append(d)

    combined_data = np.concatenate(sampled_segments, axis=-1)

    line_noise = detect_line_noise(combined_data, primary_fs)
    hf_proxy = compute_hf_noise_proxy(combined_data, primary_fs)
    amp_stats = compute_amplitude_stats(combined_data)

    return DatasetFingerprint(
        sampling_rate=primary_fs,
        all_sampling_rates=sorted(list(set(sampling_rates))),
        channel_names=raw_channels,
        standardized_channels=standardized,
        n_channels=len(raw_channels),
        duration_sec=round(total_duration, 2),
        file_count=len(raws),
        line_noise=line_noise,
        amplitude_stats=amp_stats,
        high_freq_noise=hf_proxy,
        raw_file_formats=file_formats,
    )
