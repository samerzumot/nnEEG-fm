"""Tests for Foundation Model Preprocessing Contracts and Registry."""

from __future__ import annotations

import pytest

from nneeg.contracts import (
    FMContract,
    get_contract,
    list_contracts,
    register_contract,
)


def test_labram_base_contract_registered() -> None:
    """Verify labram-base and alias labram are registered with expected contract fields."""
    contracts = list_contracts()
    assert "labram-base" in contracts
    assert "labram" in contracts

    c = get_contract("labram-base")
    assert isinstance(c, FMContract)
    assert c.target_fs == 200.0
    assert c.bandpass_low == 0.1
    assert c.bandpass_high == 75.0
    assert c.patch_samples == 200
    assert c.window_samples == 1600
    assert c.amplitude_unit == "uV"
    assert len(c.expected_channels) == 19
    assert c.normalization == "zscore_per_channel"


def test_contract_verification_notes() -> None:
    """Verify that every contract field has an explicit verification note."""
    c = get_contract("labram-base")
    assert c.verification_notes is not None
    required_keys = [
        "name",
        "target_fs",
        "bandpass_low",
        "bandpass_high",
        "notch_freqs",
        "patch_samples",
        "window_samples",
        "expected_channels",
        "amplitude_unit",
        "normalization",
    ]
    for key in required_keys:
        assert key in c.verification_notes, f"Missing verification note for field: {key}"
        assert len(c.verification_notes[key]) > 10


def test_custom_contract_registration() -> None:
    """Verify registering and retrieving a custom FM contract."""
    custom = FMContract(
        name="test-fm",
        target_fs=128.0,
        bandpass_low=0.5,
        bandpass_high=50.0,
        notch_freqs=(50.0,),
        patch_samples=128,
        window_samples=1024,
        expected_channels=("C3", "C4", "CZ"),
        amplitude_unit="uV",
        verification_notes={"test": "Custom test model."},
    )
    register_contract(custom)
    retrieved = get_contract("test-fm")
    assert retrieved.name == "test-fm"
    assert retrieved.target_fs == 128.0
    assert retrieved.expected_channels == ("C3", "C4", "CZ")


def test_contract_unknown_raises() -> None:
    """Verify requesting an unknown contract raises KeyError with available contracts."""
    with pytest.raises(KeyError, match="not found"):
        get_contract("non-existent-fm")


def test_contract_serialization() -> None:
    """Verify serialization to and from dictionary."""
    c = get_contract("labram-base")
    d = c.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "labram-base"

    reconstructed = FMContract.from_dict(d)
    assert reconstructed.name == c.name
    assert reconstructed.target_fs == c.target_fs
    assert reconstructed.expected_channels == c.expected_channels
