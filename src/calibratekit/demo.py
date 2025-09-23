"""Deterministic, fully-offline demo.

Builds a synthetic binary-classification problem, trains a deliberately
mis-calibrated scikit-learn classifier (Gaussian Naive Bayes, which is
notoriously over-confident), and recalibrates it. Everything is seeded so the
numbers quoted in the README are reproducible with ``calibratekit demo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

from .recalibrate import CalibrationResult, evaluate_recalibration

SEED = 20260713


@dataclass(frozen=True)
class DemoSplits:
    """Predicted positive-class probabilities and labels for each split."""

    valid_prob: NDArray[np.float64]
    valid_true: NDArray[np.float64]
    test_prob: NDArray[np.float64]
    test_true: NDArray[np.float64]


def make_demo_splits(seed: int = SEED) -> DemoSplits:
    """Train an over-confident classifier and return validation/test scores.

    The classifier is trained on a ``train`` split; its raw probabilities on
    the disjoint ``valid`` and ``test`` splits are what we calibrate and score.
    """
    x, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=6,
        n_redundant=10,
        n_repeated=4,
        n_clusters_per_class=2,
        class_sep=0.5,
        flip_y=0.05,
        random_state=seed,
    )
    y = y.astype(np.float64)

    # Deterministic 3-way split: 2000 train / 2000 valid / 2000 test.
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    tr, va, te = order[:2000], order[2000:4000], order[4000:6000]

    clf = GaussianNB()
    clf.fit(x[tr], y[tr])

    valid_prob = clf.predict_proba(x[va])[:, 1].astype(np.float64)
    test_prob = clf.predict_proba(x[te])[:, 1].astype(np.float64)

    return DemoSplits(
        valid_prob=valid_prob,
        valid_true=y[va],
        test_prob=test_prob,
        test_true=y[te],
    )


def run_demo(seed: int = SEED, n_bins: int = 10) -> dict[str, CalibrationResult]:
    """Return before/after results for both recalibration methods."""
    splits = make_demo_splits(seed=seed)
    return {
        method: evaluate_recalibration(
            splits.valid_prob,
            splits.valid_true,
            splits.test_prob,
            splits.test_true,
            method=method,
            n_bins=n_bins,
        )
        for method in ("isotonic", "platt")
    }
