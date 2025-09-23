"""Calibration metrics for binary probabilistic classifiers.

All functions operate on 1-D arrays of predicted probabilities for the
positive class (``y_prob``) and binary ground-truth labels (``y_true`` in
``{0, 1}``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _validate(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> None:
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true{y_true.shape} vs y_prob{y_prob.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {y_true.ndim}-D")
    if y_true.size == 0:
        raise ValueError("empty input")
    uniq = np.unique(y_true)
    if not np.isin(uniq, (0.0, 1.0)).all():
        raise ValueError(f"y_true must be binary in {{0, 1}}, got values {uniq.tolist()}")
    if (y_prob < 0.0).any() or (y_prob > 1.0).any():
        raise ValueError("y_prob must lie in [0, 1]")


def _as_arrays(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    _validate(yt, yp)
    return yt, yp


def brier_score(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
) -> float:
    """Mean squared error between predicted probabilities and labels.

    Lower is better; a perfectly calibrated *and* confident model scores 0.
    """
    yt, yp = _as_arrays(y_true, y_prob)
    return float(np.mean((yp - yt) ** 2))


@dataclass(frozen=True)
class BinStat:
    """Summary of a single reliability-diagram bin."""

    lo: float
    hi: float
    count: int
    confidence: float  # mean predicted probability in the bin
    accuracy: float  # observed positive rate in the bin

    @property
    def gap(self) -> float:
        """Signed calibration gap (accuracy - confidence)."""
        return self.accuracy - self.confidence


def bin_stats(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
    n_bins: int = 10,
) -> list[BinStat]:
    """Partition predictions into ``n_bins`` equal-width bins over [0, 1].

    Empty bins are returned with zero count and NaN confidence/accuracy so
    callers can render them faithfully.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    yt, yp = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins; clip so prob == 1.0 lands in the final bin.
    idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins - 1)

    stats: list[BinStat] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            conf = float("nan")
            acc = float("nan")
        else:
            conf = float(yp[mask].mean())
            acc = float(yt[mask].mean())
        stats.append(BinStat(lo=float(edges[b]), hi=float(edges[b + 1]),
                             count=count, confidence=conf, accuracy=acc))
    return stats


def expected_calibration_error(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of ``|accuracy - confidence|`` across equal-width bins,
    weighted by the fraction of samples in each bin. Range ``[0, 1]``; 0 is
    perfectly calibrated.
    """
    yt, _ = _as_arrays(y_true, y_prob)
    total = yt.size
    ece = 0.0
    for st in bin_stats(y_true, y_prob, n_bins=n_bins):
        if st.count == 0:
            continue
        ece += (st.count / total) * abs(st.gap)
    return ece


def maximum_calibration_error(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE): the worst per-bin calibration gap."""
    gaps = [
        abs(st.gap)
        for st in bin_stats(y_true, y_prob, n_bins=n_bins)
        if st.count > 0
    ]
    return max(gaps) if gaps else 0.0
