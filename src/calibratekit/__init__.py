"""calibratekit - probability-calibration toolkit for scikit-learn classifiers."""

from __future__ import annotations

from .metrics import (
    BinStat,
    bin_stats,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from .recalibrate import (
    CalibrationResult,
    IsotonicRecalibrator,
    PlattRecalibrator,
    evaluate_recalibration,
)
from .reliability import reliability_diagram

__version__ = "0.1.0"

__all__ = [
    "BinStat",
    "CalibrationResult",
    "IsotonicRecalibrator",
    "PlattRecalibrator",
    "__version__",
    "bin_stats",
    "brier_score",
    "evaluate_recalibration",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_diagram",
]
