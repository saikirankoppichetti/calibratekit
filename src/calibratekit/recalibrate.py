"""Post-hoc probability recalibrators for binary classifiers.

Two classic methods are provided:

* :class:`IsotonicRecalibrator` - non-parametric monotone step function
  (scikit-learn ``IsotonicRegression``). Flexible, needs more data.
* :class:`PlattRecalibrator` - parametric sigmoid (a.k.a. Platt scaling),
  a 1-D logistic regression fit on the classifier scores. Robust on small
  validation sets.

Both take a *held-out* set of predicted probabilities and true labels, and
learn a mapping ``p_raw -> p_calibrated`` that you apply to fresh scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .metrics import brier_score, expected_calibration_error

_EPS = 1e-12


def _prep(
    y_prob: NDArray[np.float64] | list[float],
    y_true: NDArray[np.float64] | list[float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    yp = np.asarray(y_prob, dtype=np.float64).ravel()
    if (yp < 0.0).any() or (yp > 1.0).any():
        raise ValueError("y_prob must lie in [0, 1]")
    yt = None
    if y_true is not None:
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        if yt.shape != yp.shape:
            raise ValueError("y_true and y_prob must have the same shape")
    return yp, yt


class Recalibrator(Protocol):
    """Common interface for recalibrators."""

    def fit(
        self,
        y_prob: NDArray[np.float64] | list[float],
        y_true: NDArray[np.float64] | list[float],
    ) -> Recalibrator: ...

    def predict(
        self, y_prob: NDArray[np.float64] | list[float]
    ) -> NDArray[np.float64]: ...


class IsotonicRecalibrator:
    """Isotonic (monotone, non-parametric) recalibration."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(
        self,
        y_prob: NDArray[np.float64] | list[float],
        y_true: NDArray[np.float64] | list[float],
    ) -> IsotonicRecalibrator:
        yp, yt = _prep(y_prob, y_true)
        assert yt is not None
        self._model.fit(yp, yt)
        self._fitted = True
        return self

    def predict(
        self, y_prob: NDArray[np.float64] | list[float]
    ) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        yp, _ = _prep(y_prob)
        out = self._model.predict(yp)
        return np.asarray(out, dtype=np.float64)


class PlattRecalibrator:
    """Platt scaling: a 1-D logistic sigmoid fit on the classifier scores.

    The raw probability is mapped through its logit and a logistic regression
    learns the sigmoid ``1 / (1 + exp(-(a*logit(p) + b)))``.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1e10, solver="lbfgs")
        self._fitted = False

    @staticmethod
    def _logit(p: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    def fit(
        self,
        y_prob: NDArray[np.float64] | list[float],
        y_true: NDArray[np.float64] | list[float],
    ) -> PlattRecalibrator:
        yp, yt = _prep(y_prob, y_true)
        assert yt is not None
        x = self._logit(yp).reshape(-1, 1)
        self._model.fit(x, yt)
        self._fitted = True
        return self

    def predict(
        self, y_prob: NDArray[np.float64] | list[float]
    ) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        yp, _ = _prep(y_prob)
        x = self._logit(yp).reshape(-1, 1)
        proba = self._model.predict_proba(x)[:, 1]
        return np.asarray(proba, dtype=np.float64)


@dataclass(frozen=True)
class CalibrationResult:
    """Before/after calibration metrics for one recalibration method."""

    method: str
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float

    @property
    def ece_improvement(self) -> float:
        """Absolute ECE reduction (positive means better calibration)."""
        return self.ece_before - self.ece_after

    @property
    def ece_improvement_pct(self) -> float:
        """Relative ECE reduction as a percentage of the original ECE."""
        if self.ece_before <= 0.0:
            return 0.0
        return 100.0 * self.ece_improvement / self.ece_before


def _method(name: str) -> Recalibrator:
    if name == "isotonic":
        return IsotonicRecalibrator()
    if name in ("platt", "sigmoid"):
        return PlattRecalibrator()
    raise ValueError(f"unknown method {name!r}; expected 'isotonic' or 'platt'")


def evaluate_recalibration(
    y_prob_valid: NDArray[np.float64] | list[float],
    y_true_valid: NDArray[np.float64] | list[float],
    y_prob_test: NDArray[np.float64] | list[float],
    y_true_test: NDArray[np.float64] | list[float],
    method: str = "isotonic",
    n_bins: int = 10,
) -> CalibrationResult:
    """Fit a recalibrator on the validation split and score it on the test split.

    Metrics are always reported on the *test* split, which the recalibrator
    never saw during fitting - this is the honest measure of improvement.
    """
    recal = _method(method)
    recal.fit(y_prob_valid, y_true_valid)
    calibrated = recal.predict(y_prob_test)

    return CalibrationResult(
        method=method,
        ece_before=expected_calibration_error(y_true_test, y_prob_test, n_bins=n_bins),
        ece_after=expected_calibration_error(y_true_test, calibrated, n_bins=n_bins),
        brier_before=brier_score(y_true_test, y_prob_test),
        brier_after=brier_score(y_true_test, calibrated),
    )
