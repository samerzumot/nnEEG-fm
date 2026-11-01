# nnEEG: Automated EEG Foundation Model Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

`nneeg` is an open-source library that automatically adapts a **FROZEN** pretrained EEG foundation model (FM) to any new EEG dataset using dataset fingerprinting, rule-based configuration, and backprop-free deterministic alignment.

---

## Key Design Principles

1. **Frozen Backbone Guarantee**: Foundation model weights are **never** modified. Adaptation is strictly offline, deterministic, and backprop-free.
2. **Rule-Based Contract Matching**: Matches raw EEG dataset characteristics against the FM's verified "preprocessing contract" (sampling rate, bandpass, notch, channels, amplitude units).
3. **MNE-Native File I/O**: Full support for `.edf`, `.bdf`, `.vhdr`, `.set`, and `.fif` files via official MNE readers.
4. **Channel Handling Ladder**: Automatically handles channel discrepancies:
   - **Native Subset**: Directly extracts and reorders channels if present natively.
   - **Spherical-Spline Interpolation**: Reconstructs missing channels using 3D spherical splines when standard electrode coordinates are available.
   - **Explicit Error Reporting**: Flags datasets with insufficient spatial sensors (< 4 standard channels).
5. **Determinism**: Fully reproducible execution on identical hardware using fixed seeds and deterministic PyTorch execution.

---

## Milestone 1 Implementation

- **`DatasetFingerprint`** (`nneeg.fingerprint`):
  - Detects sampling rates across files.
  - Automatically identifies line noise frequencies (50 Hz / 60 Hz and harmonics) via Welch peak SNR analysis.
  - Calculates high-frequency noise proxies (>80 Hz power ratio), gracefully handling $f_s < 160\text{ Hz}$ by recording explicit skip explanations.
  - Computes amplitude distribution statistics (min, max, mean, std, median, IQR).
- **`FMContract`** (`nneeg.contracts`):
  - Preprocessing contracts for EEG foundation models.
  - Built-in registry with verified `labram-base` (Large Brain Model, Jiang et al., ICLR 2024).
  - Explicit per-field verification notes documented in [VERIFY.md](VERIFY.md).
- **`PipelinePlan`** (`nneeg.planner`):
  - Deterministic rule engine generating a serializable `PipelinePlan` with human-readable reason strings for every decision.
  - Resampling via rational polyphase filters (`resample_poly`).
  - Selective notch filtering applied only when line noise peaks are detected.
  - Bandpass filtering matching pretraining contracts (independent of downstream task).
- **`preprocess`** (`nneeg.preprocess`):
  - Preprocesses raw EEG data into $(N, C, T)$ PyTorch tensors (epochs $\times$ channels $\times$ time samples).
- **CLI** (`nneeg`):
  - `nneeg fingerprint <path> [-o output.json]`
  - `nneeg plan <path> --fm <contract_name> [-o output.json]`

---

## Installation

```bash
git clone https://github.com/samerzumot/nnEEG-fm.git
cd nnEEG-fm
pip install -e .
```

---

## Quickstart

### 1. Command Line Interface

```bash
# Fingerprint an EEG recording or directory
nneeg fingerprint /path/to/recording.edf -o fingerprint.json

# Generate an adaptation plan matching the dataset to LaBraM
nneeg plan /path/to/recording.edf --fm labram-base -o plan.json
```

### 2. Python API

```python
import mne
from nneeg import fingerprint_dataset, get_contract, generate_plan, preprocess_raw

# 1. Load raw EEG file with MNE
raw = mne.io.read_raw_edf("recording.edf", preload=True)

# 2. Extract dataset fingerprint
fingerprint = fingerprint_dataset(raw)
print(f"Sampling rate: {fingerprint.sampling_rate} Hz")
print(f"Detected line noise: {fingerprint.line_noise.detected_freq} Hz")

# 3. Retrieve Foundation Model contract
contract = get_contract("labram-base")

# 4. Generate deterministic PipelinePlan
plan = generate_plan(fingerprint, contract)
for step in plan.steps:
    print(f"[{step.step_name}] {step.action}: {step.reason}")

# 5. Execute preprocessing to obtain (N, C, T) tensor
tensor = preprocess_raw(raw, plan, contract=contract)
print("Output tensor shape:", tensor.shape)  # e.g., torch.Size([N, 19, 1600])
```

---

## Preprocessing Verification

For complete transparency, all foundation model parameters are verified against official repositories and publications. See [VERIFY.md](VERIFY.md) for detailed notes on `labram-base` and upcoming models.

---

## Roadmap

- **Milestone 1 (Current)**: Skeleton, Fingerprinting, FM Contracts, Pipeline Planner, MNE Preprocessing, CLI, Tests.
- **Milestone 2**: Feature-space alignment (mean/covariance recentering with shrinkage) and closed-form heads (Ridge / LDA / Class prototypes).
- **Milestone 3**: Evaluation harness comparing against strong baselines (pyRiemann tangent-space, per-recording z-score, EA+FM, LoRA upper bound).
- **Milestone 4**: Additional foundation model backbones (e.g. CBraMod).
- **Milestone 5**: Packaging, documentation, and locked-algorithm submission support.
