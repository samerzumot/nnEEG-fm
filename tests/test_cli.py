"""Tests for the nnEEG Command Line Interface (CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import mne

from nneeg.cli import main


def test_cli_fingerprint(
    tmp_path: Path,
    synthetic_clean_raw: mne.io.BaseRaw,
    capsys,
) -> None:
    """Verify nneeg fingerprint command outputs valid JSON."""
    raw_path = tmp_path / "sample_raw.fif"
    synthetic_clean_raw.save(raw_path, overwrite=True, verbose=False)

    exit_code = main(["fingerprint", str(raw_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "sampling_rate" in captured.out
    data = json.loads(captured.out)
    assert data["sampling_rate"] == 200.0
    assert data["n_channels"] == 19


def test_cli_fingerprint_with_output_file(
    tmp_path: Path,
    synthetic_clean_raw: mne.io.BaseRaw,
    capsys,
) -> None:
    """Verify nneeg fingerprint saves artifact when --output is provided."""
    raw_path = tmp_path / "sample_raw.fif"
    out_file = tmp_path / "fingerprint.json"
    synthetic_clean_raw.save(raw_path, overwrite=True, verbose=False)

    exit_code = main(["fingerprint", str(raw_path), "-o", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()

    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["sampling_rate"] == 200.0


def test_cli_plan(
    tmp_path: Path,
    synthetic_50hz_noise_raw: mne.io.BaseRaw,
    capsys,
) -> None:
    """Verify nneeg plan command generates pipeline plan JSON."""
    raw_path = tmp_path / "sample_noise_raw.fif"
    out_file = tmp_path / "plan.json"
    synthetic_50hz_noise_raw.save(raw_path, overwrite=True, verbose=False)

    exit_code = main(["plan", str(raw_path), "--fm", "labram-base", "-o", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()

    plan_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert plan_data["contract_name"] == "labram-base"
    assert plan_data["target_fs"] == 200.0
    assert plan_data["resample_step"]["action"] == "resample_poly"
    assert plan_data["notch_step"]["action"] == "apply_notch"
