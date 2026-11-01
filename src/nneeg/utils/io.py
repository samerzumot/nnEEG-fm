"""MNE-backed EEG file I/O utilities.

Strictly uses MNE's official readers for .edf, .bdf, .vhdr, .set, and .fif files.
Never reimplements binary file readers.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne

SUPPORTED_EXTENSIONS: set[str] = {".edf", ".bdf", ".vhdr", ".set", ".fif", ".fif.gz"}


def discover_eeg_files(path: str | Path) -> list[Path]:
    """Find all supported EEG files in a directory or validate a single file.

    Parameters
    ----------
    path : str | Path
        Path to file or directory.

    Returns
    -------
    list[Path]
        Sorted list of resolved Path objects.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        ext = "".join(p.suffixes).lower()
        # Handle .fif.gz
        if ext.endswith(".gz") and len(p.suffixes) >= 2:
            ext = "".join(p.suffixes[-2:]).lower()
        if any(ext.endswith(supported) for supported in SUPPORTED_EXTENSIONS):
            return [p]
        raise ValueError(
            f"Unsupported file format: {p.name}. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Search directory recursively and sort deterministically
    files: list[Path] = []
    for root, _, filenames in os.walk(p):
        for fname in sorted(filenames):
            fp = Path(root) / fname
            # Ignore hidden files
            if fname.startswith("."):
                continue
            ext = "".join(fp.suffixes).lower()
            if ext.endswith(".gz") and len(fp.suffixes) >= 2:
                ext = "".join(fp.suffixes[-2:]).lower()
            if any(ext.endswith(supported) for supported in SUPPORTED_EXTENSIONS):
                files.append(fp)

    return sorted(files)


def load_raw_file(path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Read a raw EEG file using MNE's format-specific readers.

    Parameters
    ----------
    path : str | Path
        Path to .edf, .bdf, .vhdr, .set, or .fif file.
    preload : bool
        Whether to load data into memory immediately.

    Returns
    -------
    mne.io.BaseRaw
        Loaded MNE Raw instance.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    fname_lower = p.name.lower()
    if fname_lower.endswith(".edf"):
        return mne.io.read_raw_edf(p, preload=preload, verbose=False)
    elif fname_lower.endswith(".bdf"):
        return mne.io.read_raw_bdf(p, preload=preload, verbose=False)
    elif fname_lower.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(p, preload=preload, verbose=False)
    elif fname_lower.endswith(".set"):
        return mne.io.read_raw_eeglab(p, preload=preload, verbose=False)
    elif fname_lower.endswith(".fif") or fname_lower.endswith(".fif.gz"):
        return mne.io.read_raw_fif(p, preload=preload, verbose=False)
    else:
        raise ValueError(
            f"Unsupported EEG extension for file '{p.name}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
