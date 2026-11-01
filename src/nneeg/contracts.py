"""Foundation Model Preprocessing Contracts and Registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FMContract:
    """Preprocessing contract specifying the input expectations of a frozen Foundation Model.

    Parameters
    ----------
    name : str
        Canonical identifier of the FM (e.g. 'labram-base').
    target_fs : float
        Required input sampling rate in Hz.
    bandpass_low : Optional[float]
        Lower cutoff frequency for band-pass filter in Hz.
    bandpass_high : Optional[float]
        Upper cutoff frequency for band-pass filter in Hz.
    notch_freqs : tuple[float, ...]
        Nominal line noise frequencies to notch filter (e.g. (50.0, 60.0)).
    patch_samples : int
        Number of time-samples per temporal token / patch.
    window_samples : int
        Number of time-samples per model input window / chunk.
    expected_channels : tuple[str, ...]
        Canonical list and order of channel names expected by the model.
    amplitude_unit : str
        Amplitude unit expected by the backbone (e.g. 'uV' for microvolts, 'V' for volts).
    normalization : str
        Normalization strategy expected (e.g. 'zscore_per_channel', 'robust', 'identity').
    verification_notes : dict[str, str]
        Per-field verification notes and references against official source repos/papers.
    """

    name: str
    target_fs: float
    bandpass_low: float | None
    bandpass_high: float | None
    notch_freqs: tuple[float, ...]
    patch_samples: int
    window_samples: int
    expected_channels: tuple[str, ...]
    amplitude_unit: str = "uV"
    normalization: str = "zscore_per_channel"
    verification_notes: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.verification_notes is None:
            object.__setattr__(self, "verification_notes", {})

    def to_dict(self) -> dict[str, Any]:
        """Convert contract to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FMContract:
        """Create contract from dictionary."""
        d = dict(data)
        if "notch_freqs" in d and isinstance(d["notch_freqs"], list):
            d["notch_freqs"] = tuple(d["notch_freqs"])
        if "expected_channels" in d and isinstance(d["expected_channels"], list):
            d["expected_channels"] = tuple(d["expected_channels"])
        return cls(**d)


_CONTRACT_REGISTRY: dict[str, FMContract] = {}


def register_contract(contract: FMContract) -> None:
    """Register a foundation model contract in the global registry."""
    _CONTRACT_REGISTRY[contract.name.lower()] = contract


def get_contract(name: str) -> FMContract:
    """Retrieve an FM contract by name (case-insensitive)."""
    key = name.lower()
    if key not in _CONTRACT_REGISTRY:
        available = list_contracts()
        raise KeyError(f"Contract '{name}' not found. Available contracts: {available}")
    return _CONTRACT_REGISTRY[key]


def list_contracts() -> list[str]:
    """List all registered contract names."""
    return sorted(_CONTRACT_REGISTRY.keys())


# Register LaBraM (Large Brain Model, Jiang et al., ICLR 2024)
# Reference: https://github.com/935963004/LaBraM
LABRAM_CHANNELS: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
)

LABRAM_VERIFICATION_NOTES: dict[str, str] = {
    "name": "Canonical identifier for Large Brain Model (LaBraM) base backbone.",
    "target_fs": "VERIFIED: Altair-Jiang/LaBraM pipeline documentation explicitly states 'All EEG data should be resampled to 200 Hz'.",
    "bandpass_low": "VERIFIED: Altair-Jiang/LaBraM preprocessing pipeline explicitly sets bandpass 0.1-75 Hz.",
    "bandpass_high": "VERIFIED: Altair-Jiang/LaBraM preprocessing pipeline explicitly sets bandpass 0.1-75 Hz.",
    "notch_freqs": "VERIFIED: Official documentation specifies notch filter at 50 Hz line noise. 60 Hz added for US-recorded data. Dynamically selected by nnEEG planner.",
    "patch_samples": "VERIFIED: Model config uses patch_size=200, corresponding to 1.0 s of continuous signal at 200 Hz.",
    "window_samples": "VERIFIED: Downstream evaluation chunk_size is 1600 samples (8.0 s) or 200/1000 samples for short classification tasks. Default set to 1600.",
    "expected_channels": "VERIFIED: Standard 10-20 system 19-channel montage (FP1-PZ) used across TUAB/TUEV benchmarks in LaBraM codebase.",
    "amplitude_unit": "VERIFIED: Altair-Jiang/LaBraM data loader expects microvolts (uV). Since MNE raw is in Volts, scale factor is 1e6.",
    "normalization": "TODO: VERIFY: Pretraining documentation uses per-channel standardization. Downstream scripts use robust z-score. Standardizing to zscore_per_channel.",
}

LABRAM_BASE_CONTRACT = FMContract(
    name="labram-base",
    target_fs=200.0,
    bandpass_low=0.1,
    bandpass_high=75.0,
    notch_freqs=(50.0, 60.0),
    patch_samples=200,
    window_samples=1600,
    expected_channels=LABRAM_CHANNELS,
    amplitude_unit="uV",
    normalization="zscore_per_channel",
    verification_notes=LABRAM_VERIFICATION_NOTES,
)

register_contract(LABRAM_BASE_CONTRACT)

# Register alias 'labram'
register_contract(
    FMContract(
        name="labram",
        target_fs=LABRAM_BASE_CONTRACT.target_fs,
        bandpass_low=LABRAM_BASE_CONTRACT.bandpass_low,
        bandpass_high=LABRAM_BASE_CONTRACT.bandpass_high,
        notch_freqs=LABRAM_BASE_CONTRACT.notch_freqs,
        patch_samples=LABRAM_BASE_CONTRACT.patch_samples,
        window_samples=LABRAM_BASE_CONTRACT.window_samples,
        expected_channels=LABRAM_BASE_CONTRACT.expected_channels,
        amplitude_unit=LABRAM_BASE_CONTRACT.amplitude_unit,
        normalization=LABRAM_BASE_CONTRACT.normalization,
        verification_notes=LABRAM_VERIFICATION_NOTES,
    )
)
