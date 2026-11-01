"""Deterministic execution utilities for nnEEG."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution across libraries.

    Parameters
    ----------
    seed : int
        Seed value for random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms in PyTorch where available
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some operations on certain backends may raise an error if unsupported
        pass


def is_deterministic_env() -> bool:
    """Check if deterministic algorithms are enabled in PyTorch."""
    return torch.are_deterministic_algorithms_enabled()
