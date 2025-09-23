from __future__ import annotations

import numpy as np
import pytest

from calibratekit.metrics import (
    bin_stats,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)


def test_brier_perfect_predictions() -> None:
    y = [0.0, 1.0, 1.0, 0.0]
    assert brier_score(y, y) == 0.0


def test_brier_worst_predictions() -> None:
    # Always predict the opposite with full confidence -> Brier = 1.
    assert brier_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_matches_definition() -> None:
    y_true = [1.0, 0.0, 1.0]
    y_prob = [0.8, 0.3, 0.6]
    expected = np.mean([(0.8 - 1) ** 2, (0.3 - 0) ** 2, (0.6 - 1) ** 2])
    assert brier_score(y_true, y_prob) == pytest.approx(expected)


def test_ece_perfectly_calibrated_is_zero() -> None:
    # Each prob bucket has an empirical rate exactly matching its confidence.
    y_prob = [0.1] * 10 + [0.9] * 10
    y_true = [1.0] + [0.0] * 9 + [1.0] * 9 + [0.0]
    assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    # Model says 0.99 but is right only half the time -> large ECE.
    y_prob = [0.99] * 100
    y_true = [1.0, 0.0] * 50
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.49, abs=1e-9)


def test_mce_is_worst_bin_gap() -> None:
    y_prob = [0.05] * 10 + [0.95] * 10
    y_true = [0.0] * 10 + [0.0] * 10  # second bin totally wrong: gap 0.95
    mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
    assert mce == pytest.approx(0.95, abs=1e-9)


def test_bin_stats_partition_and_counts() -> None:
    y_prob = [0.05, 0.15, 0.95]
    y_true = [0.0, 1.0, 1.0]
    stats = bin_stats(y_true, y_prob, n_bins=10)
    assert len(stats) == 10
    assert sum(s.count for s in stats) == 3
    assert stats[0].count == 1
    assert stats[1].count == 1
    assert stats[9].count == 1
    # Empty bins report NaN, not a fabricated 0.
    assert np.isnan(stats[5].confidence)


def test_bin_stats_gap_sign() -> None:
    # Bin confidence 0.85, accuracy 1.0 -> positive gap (under-confident).
    st = bin_stats([1.0], [0.85], n_bins=10)[8]
    assert st.gap == pytest.approx(0.15)


def test_prob_one_lands_in_last_bin() -> None:
    stats = bin_stats([1.0], [1.0], n_bins=10)
    assert stats[9].count == 1


@pytest.mark.parametrize(
    "y_true, y_prob",
    [
        ([0.0, 1.0], [0.5]),  # shape mismatch
        ([2.0], [0.5]),  # non-binary label
        ([0.0], [1.5]),  # prob out of range
        ([], []),  # empty
    ],
)
def test_metrics_reject_bad_input(y_true: list[float], y_prob: list[float]) -> None:
    with pytest.raises(ValueError):
        brier_score(y_true, y_prob)


def test_bin_stats_rejects_zero_bins() -> None:
    with pytest.raises(ValueError):
        bin_stats([0.0, 1.0], [0.2, 0.8], n_bins=0)
