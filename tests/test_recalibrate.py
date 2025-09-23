from __future__ import annotations

import numpy as np
import pytest

from calibratekit.demo import make_demo_splits
from calibratekit.recalibrate import (
    IsotonicRecalibrator,
    PlattRecalibrator,
    evaluate_recalibration,
)


def _overconfident() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 2000
    # True event rate is 0.5 in the top bucket but the model claims ~0.95.
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_prob = np.where(y_true == 1, 0.97, 0.9) + rng.normal(0, 0.01, size=n)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return y_prob, y_true


@pytest.mark.parametrize("cls", [IsotonicRecalibrator, PlattRecalibrator])
def test_predict_stays_in_unit_interval(cls: type) -> None:
    y_prob, y_true = _overconfident()
    recal = cls().fit(y_prob, y_true)
    out = recal.predict(np.linspace(0, 1, 50))
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.shape == (50,)


@pytest.mark.parametrize("cls", [IsotonicRecalibrator, PlattRecalibrator])
def test_recalibration_is_monotone(cls: type) -> None:
    y_prob, y_true = _overconfident()
    recal = cls().fit(y_prob, y_true)
    grid = np.linspace(0.01, 0.99, 100)
    out = recal.predict(grid)
    assert np.all(np.diff(out) >= -1e-9)


@pytest.mark.parametrize("cls", [IsotonicRecalibrator, PlattRecalibrator])
def test_predict_before_fit_raises(cls: type) -> None:
    with pytest.raises(RuntimeError):
        cls().predict([0.5])


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_recalibration([0.5], [1.0], [0.5], [1.0], method="bogus")


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_recalibration_improves_ece_on_demo(method: str) -> None:
    s = make_demo_splits()
    r = evaluate_recalibration(
        s.valid_prob, s.valid_true, s.test_prob, s.test_true, method=method
    )
    # The GaussianNB demo is badly over-confident; both methods should cut ECE.
    assert r.ece_before > 0.15
    assert r.ece_after < r.ece_before
    assert r.ece_improvement > 0.0
    assert r.ece_improvement_pct > 50.0


def test_platt_and_sigmoid_are_aliases() -> None:
    s = make_demo_splits()
    a = evaluate_recalibration(
        s.valid_prob, s.valid_true, s.test_prob, s.test_true, method="platt"
    )
    b = evaluate_recalibration(
        s.valid_prob, s.valid_true, s.test_prob, s.test_true, method="sigmoid"
    )
    assert a.ece_after == pytest.approx(b.ece_after)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        IsotonicRecalibrator().fit([0.1, 0.2], [1.0])
