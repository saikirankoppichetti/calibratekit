# calibratekit

**A tiny, offline toolkit that tells you whether your classifier's probabilities can be trusted - and fixes them when they can't.**

## The problem

A scikit-learn model that reports `predict_proba([...]) = 0.95` is only useful
if events it calls "95% likely" actually happen about 95% of the time. Many
strong classifiers (Naive Bayes, SVMs, boosted trees, deep nets) are
**mis-calibrated** - usually over-confident - so those numbers lie. That
matters the moment a downstream decision depends on the probability itself:
thresholding, expected-value ranking, risk scoring, abstention.

`calibratekit` gives you the two things you need:

1. **Measurement** - Expected Calibration Error (ECE), Maximum Calibration
   Error (MCE), the Brier score, and a readable ASCII reliability diagram.
2. **Repair** - post-hoc **isotonic** and **Platt (sigmoid)** recalibrators,
   fit on a held-out split, with an honest before/after ECE report measured on
   a separate test split.

No Docker, no services, no network, no downloads. Pure Python on top of
`numpy` + `scikit-learn`.

## Quickstart

From a fresh clone (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run calibratekit demo
```

### The demo, and the numbers it prints

The demo trains a deliberately over-confident **Gaussian Naive Bayes** on a
seeded synthetic dataset (`seed = 20260713`), then recalibrates it. It is fully
deterministic - you get exactly these numbers:

```
calibratekit demo - GaussianNB on synthetic data (seed fixed)

method      ECE before   ECE after   improvement   Brier after
--------------------------------------------------------------
isotonic        0.2219      0.0287   +0.1932 (  87%)        0.2041
platt           0.2219      0.0230   +0.1989 (  90%)        0.2031
```

The raw model is badly mis-calibrated (**ECE 0.2219**). Platt scaling cuts that
to **0.0230 - a 90% reduction** on the untouched test split; isotonic reaches
**0.0287 (87%)**. Both also lower the Brier score, so the fix improves accuracy
and honesty, not just one at the expense of the other.

### Inspect any scores you have

Point the `report` command at a CSV of `y_true,y_prob` (header optional):

```bash
uv run calibratekit report tests/fixtures/sample_scores.csv
```

```
samples : 200
ECE     : 0.2433
MCE     : 0.4548
Brier   : 0.2633

Reliability diagram (10 bins)  |  ECE = 0.2433
======================================================================
      range  diagram                                       n      gap
----------------------------------------------------------------------
    0.0-0.1  #+###########...........................     82  +0.277
    0.1-0.2  ######+################.................     16  +0.417
    0.2-0.3  ##########+.............................      8  -0.010
    0.3-0.4  ##############+######...................      4  +0.142
    0.4-0.5  ###########......|......................      4  -0.191
    0.5-0.6  ##############.......|..................      6  -0.208
    0.6-0.7  #########.................|.............      5  -0.455
    0.7-0.8  #############################|..........      7  -0.037
    0.8-0.9  #######################...........|.....      7  -0.293
    0.9-1.0  ################################......|.     61  -0.197
----------------------------------------------------------------------
legend: '#' observed accuracy, '|' predicted confidence
```

Read it like this: `#` marks the **observed** accuracy in each probability bin,
`|` marks the **predicted** confidence. When `|` sits to the right of where the
`#` bar ends, the model is over-confident for that bin (a negative gap); to
the left, under-confident. The `0.9-1.0` row above says "90%+ confident" but is
only right ~70% of the time - a textbook over-confidence signature.

## Library API

```python
import numpy as np
from calibratekit import (
    expected_calibration_error,
    brier_score,
    reliability_diagram,
    IsotonicRecalibrator,
    evaluate_recalibration,
)

y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.95])   # over-confident scores

expected_calibration_error(y_true, y_prob, n_bins=10)  # -> float
brier_score(y_true, y_prob)                            # -> float
print(reliability_diagram(y_true, y_prob))             # -> ASCII string

# Fit on a validation split, apply to fresh scores:
recal = IsotonicRecalibrator().fit(valid_prob, valid_true)
calibrated = recal.predict(test_prob)

# Or get a full before/after report in one call:
result = evaluate_recalibration(
    valid_prob, valid_true, test_prob, test_true, method="platt"
)
print(result.ece_before, result.ece_after, result.ece_improvement_pct)
```

Recalibrators are always fit on one split and scored on a **disjoint** split,
so the reported improvement reflects generalization, not the model memorizing
its own validation set.

## Development

```bash
uv sync
uv run pytest      # 35 tests
uv run ruff check .
uv run mypy src tests
```

## What I'd build next

- **Multiclass calibration** - one-vs-rest ECE and temperature scaling for
  softmax outputs (currently binary-only).
- **Temperature scaling** - the single-parameter logit rescaling that is the
  default fix for neural nets, alongside isotonic/Platt.
- **Adaptive (equal-mass) binning** - ECE with quantile bins so sparse
  high-confidence regions aren't dominated by a few points.
- **Confidence intervals on ECE** via bootstrap, so "0.028 vs 0.023" comes with
  an honest error bar.
- **Matplotlib export** of the reliability diagram for notebooks/reports.

## Maintainer

Sai Kiran Koppichetti is an AI/ML Engineer specializing in machine learning, risk modeling, and evaluation pipelines. He maintains calibratekit as part of a focus on ensuring model reliability and trust in production environments.

- GitHub: https://github.com/saikirankoppichetti
- LinkedIn: https://www.linkedin.com/in/saikirankoppichetti97/
- Email: koppichettisaikiran97@gmail.com