"""Channel cleaning, alias mapping, and montage interpolation utilities."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import mne
import numpy as np

# Standard 10-20 bidirectional aliases between legacy 10-20 and modified combinatorial 10-20
STANDARD_1020_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T7": "T3",
    "T4": "T8",
    "T8": "T4",
    "T5": "P7",
    "P7": "T5",
    "T6": "P8",
    "P8": "T6",
}

# Canonical 10-20 names (uppercase representation)
CANONICAL_1020_UPPER: set[str] = {
    "FP1",
    "FP2",
    "FPZ",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "AFZ",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "T7",
    "T3",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T4",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "T5",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "T6",
    "PO7",
    "PO3",
    "POZ",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2",
    "A1",
    "A2",
    "M1",
    "M2",
}


def clean_channel_name(name: str) -> str:
    """Standardize an EEG channel name.

    Strips common prefixes (e.g. 'EEG ', 'EEG-'), suffixes (e.g. '-REF', '-LE', '-AVG'),
    and whitespace, returning uppercase alphanumeric representation.

    Parameters
    ----------
    name : str
        Raw channel name from recording.

    Returns
    -------
    str
        Cleaned, uppercase standard channel label.
    """
    clean = name.strip()
    # Remove common prefixes
    clean = re.sub(r"^(?:EEG|Eeg|eeg)[\s_\-:]*", "", clean)
    # Remove common reference suffixes
    clean = re.sub(r"[\s_\-]*(?:REF|Ref|ref|LE|Le|le|AVG|Avg|avg|A1|A2|M1|M2)$", "", clean)
    clean = clean.strip()

    # Uppercase
    clean_upper = clean.upper()

    # Normalize Fp1/FP1, Fpz/FPZ
    if clean_upper.startswith("FP"):
        return clean_upper
    return clean_upper


def get_standard_montage() -> mne.channels.DigMontage:
    """Load standard 10-20 montage without deprecation noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            return mne.channels.make_standard_montage("standard_1020")
        except Exception:
            return mne.channels.make_standard_montage("colin27_1020")


def match_channel_to_target(
    channel: str,
    target_channels: Sequence[str],
) -> str | None:
    """Match a cleaned channel name against target channels considering aliases.

    Parameters
    ----------
    channel : str
        Cleaned channel name (e.g. 'T3', 'FP1').
    target_channels : Sequence[str]
        List of expected target channel names (e.g. ['T7', 'FP1', ...]).

    Returns
    -------
    str | None
        Matching target channel name, or None if no match.
    """
    ch_upper = clean_channel_name(channel)
    target_map = {t.upper(): t for t in target_channels}

    # Direct match
    if ch_upper in target_map:
        return target_map[ch_upper]

    # Check alias
    if ch_upper in STANDARD_1020_ALIASES:
        alias = STANDARD_1020_ALIASES[ch_upper].upper()
        if alias in target_map:
            return target_map[alias]

    return None


def interpolate_missing_channels(
    raw: mne.io.BaseRaw,
    target_channels: Sequence[str],
) -> mne.io.BaseRaw:
    """Interpolate missing channels using spherical spline interpolation.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw instance with existing channels.
    target_channels : Sequence[str]
        Full list of channels required.

    Returns
    -------
    mne.io.BaseRaw
        Raw instance containing all target channels interpolated if missing.
    """
    raw_copy = raw.copy()
    montage = get_standard_montage()

    # First clean and rename existing channels if possible
    rename_dict: dict[str, str] = {}
    for ch in raw_copy.ch_names:
        matched = match_channel_to_target(ch, target_channels)
        if matched and matched != ch:
            rename_dict[ch] = matched

    if rename_dict:
        raw_copy.rename_channels(rename_dict)

    # Set montage for existing channels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        raw_copy.set_montage(montage, match_case=False, on_missing="ignore")

    # Find missing channels
    existing_upper = {ch.upper(): ch for ch in raw_copy.ch_names}
    missing_targets: list[str] = []
    for tgt in target_channels:
        if tgt.upper() not in existing_upper:
            # Check alias
            alias = STANDARD_1020_ALIASES.get(tgt.upper(), "").upper()
            if alias not in existing_upper:
                missing_targets.append(tgt)

    if not missing_targets:
        # All target channels already exist
        return raw_copy

    # Add zero/flat data for missing channels
    for ch in missing_targets:
        dummy_info = mne.create_info([ch], sfreq=raw_copy.info["sfreq"], ch_types="eeg")
        dummy_data = np.zeros((1, raw_copy.n_times), dtype=np.float64)
        dummy_raw = mne.io.RawArray(dummy_data, dummy_info, verbose=False)
        raw_copy.add_channels([dummy_raw])

    # Re-apply montage to include the new channels
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        raw_copy.set_montage(montage, match_case=False, on_missing="ignore")

    # Mark added channels as bads for interpolation
    raw_copy.info["bads"] = list(missing_targets)

    # Perform spherical spline interpolation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        raw_copy.interpolate_bads(reset_bads=True, method="spline", verbose=False)

    return raw_copy
