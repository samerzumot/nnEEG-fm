"""nnEEG: Automated EEG dataset fingerprinting, rule-based config, and deterministic alignment."""

from nneeg.contracts import (
    FMContract,
    get_contract,
    list_contracts,
    register_contract,
)
from nneeg.fingerprint import (
    AmplitudeStats,
    DatasetFingerprint,
    HighFreqNoiseProxy,
    LineNoiseDetection,
    fingerprint_dataset,
)
from nneeg.planner import (
    PipelinePlan,
    PlanStep,
    generate_plan,
)
from nneeg.preprocess import (
    preprocess_dataset,
    preprocess_raw,
)
from nneeg.utils.determinism import (
    is_deterministic_env,
    set_seed,
)

__version__ = "0.1.0"

__all__ = [
    "AmplitudeStats",
    "DatasetFingerprint",
    "FMContract",
    "HighFreqNoiseProxy",
    "LineNoiseDetection",
    "PipelinePlan",
    "PlanStep",
    "fingerprint_dataset",
    "generate_plan",
    "get_contract",
    "is_deterministic_env",
    "list_contracts",
    "preprocess_dataset",
    "preprocess_raw",
    "register_contract",
    "set_seed",
]
