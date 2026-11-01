# VERIFY.md: Foundation Model Preprocessing Contracts & Verification Status

This document catalogs the exact preprocessing requirements for EEG Foundation Models supported in `nneeg`. Every field is either verified against the official codebase/paper or flagged with an explicit `TODO` verification item.

---

## 1. LaBraM (`labram-base`)
- **Reference Paper**: *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI* (Jiang et al., ICLR 2024 / arXiv:2310.02107)
- **Official Repository**: [https://github.com/935963004/LaBraM](https://github.com/935963004/LaBraM) (Altair-Jiang/LaBraM)

### Specification Table & Verification Notes

| Parameter | Value | Status | Source / Verification Details |
| :--- | :--- | :--- | :--- |
| `target_fs` | `200.0` Hz | **VERIFIED** | Official pipeline instructions in `Altair-Jiang/LaBraM`: "All EEG data should be resampled to 200 Hz". |
| `bandpass_low` | `0.1` Hz | **VERIFIED** | Stated in pipeline documentation: "Bandpass filter: 0.1-75 Hz". |
| `bandpass_high` | `75.0` Hz | **VERIFIED** | Stated in pipeline documentation: "Bandpass filter: 0.1-75 Hz". |
| `notch_freqs` | `(50.0, 60.0)` | **VERIFIED** | Pipeline states 50 Hz notch filter (regional default for TUH/Europe; 60 Hz for US recordings). `nneeg` dynamically plans notch based on detected line frequency. |
| `amplitude_unit` | `uV` ($\mu\text{V}$) | **VERIFIED** | Pipeline explicitly requires amplitude unit in $\mu\text{V}$. Since MNE stores data in Volts (V), a scale factor of $10^6$ is applied. |
| `patch_samples` | `200` samples | **VERIFIED** | Model architecture uses temporal patch tokens of 1 second ($200\text{ samples} / 200\text{ Hz} = 1.0\text{ s}$). |
| `window_samples` | `1600` samples | **VERIFIED / CONFIGURABLE** | Default chunk size in evaluation scripts is 8.0 s (1600 samples) or 1.0 s / 5.0 s depending on task (TUAB/TUEV). Standard chunking defaults to 1600 samples. |
| `expected_channels` | Standard 10-20 (19/20 channels) | **VERIFIED** | Canonical channels: `['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']`. Standard 10-20 aliases (e.g. T7 for T3, T8 for T4, P7 for T5, P8 for T6) are supported. |
| `normalization` | `zscore_per_channel` | **TODO: VERIFY** | Downstream scripts use per-channel robust or z-score normalization. Must confirm whether pretraining used zero-mean unit-variance per recording or per patch. |

---

## 2. CBraMod (Upcoming in Milestone 4)
- **Reference Paper**: *CBraMod: Continuous Brain Model* (Wang et al., 2024)
- **Status**: Deferred to Milestone 4 per prompt specification.
- **Fields to verify**:
  - `target_fs`: TODO: Check exact pretraining sampling rate.
  - `expected_channels`: TODO: Check montage requirements.
  - `window_samples`: TODO: Check continuous patch encoding length.

---

## 3. General Preprocessing Invariants
1. **Determinism**: Resampling must use rational polyphase FIR filters (`scipy.signal.resample_poly` or MNE's `resample`) with zero-phase filtering (`raw.filter(..., phase='zero')`).
2. **Channel Ordering**: Channels are strictly ordered to match the foundation model's spatial embedding table.
3. **No Backpropagation**: All pipeline decisions are rule-based, deterministic, and logged with explicit reason strings.
