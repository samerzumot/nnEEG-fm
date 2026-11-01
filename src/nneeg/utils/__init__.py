"""Internal utility functions for nnEEG."""

from nneeg.utils.channels import (
    clean_channel_name,
    get_standard_montage,
    interpolate_missing_channels,
    match_channel_to_target,
)
from nneeg.utils.determinism import is_deterministic_env, set_seed
from nneeg.utils.io import discover_eeg_files, load_raw_file

__all__ = [
    "clean_channel_name",
    "discover_eeg_files",
    "get_standard_montage",
    "interpolate_missing_channels",
    "is_deterministic_env",
    "load_raw_file",
    "match_channel_to_target",
    "set_seed",
]
