"""Command-line interface for calibratekit.

Subcommands
-----------
* ``demo``   - run the deterministic synthetic demo and print a report.
* ``report`` - read a CSV of ``y_true,y_prob`` and print calibration metrics
  plus an ASCII reliability diagram.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .demo import run_demo
from .metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from .reliability import reliability_diagram


def _read_csv(path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true: list[float] = []
    y_prob: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty file")
    start = 0
    first = rows[0]
    # Skip a header row if the first cell is not numeric.
    try:
        float(first[0])
    except ValueError:
        start = 1
    for i, row in enumerate(rows[start:], start=start + 1):
        if len(row) < 2:
            raise ValueError(f"{path}:{i}: expected 2 columns 'y_true,y_prob'")
        y_true.append(float(row[0]))
        y_prob.append(float(row[1]))
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_prob, dtype=np.float64)


def _print_metrics(
    y_true: NDArray[np.float64], y_prob: NDArray[np.float64], n_bins: int
) -> None:
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
    brier = brier_score(y_true, y_prob)
    print(f"samples : {y_true.size}")
    print(f"ECE     : {ece:.4f}")
    print(f"MCE     : {mce:.4f}")
    print(f"Brier   : {brier:.4f}")
    print()
    print(reliability_diagram(y_true, y_prob, n_bins=n_bins))


def _cmd_report(args: argparse.Namespace) -> int:
    y_true, y_prob = _read_csv(args.csv)
    _print_metrics(y_true, y_prob, args.bins)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    results = run_demo(n_bins=args.bins)
    print("calibratekit demo - GaussianNB on synthetic data (seed fixed)\n")
    header = (
        f"{'method':<10}{'ECE before':>12}{'ECE after':>12}"
        f"{'improvement':>14}{'Brier after':>14}"
    )
    print(header)
    print("-" * len(header))
    for name in ("isotonic", "platt"):
        r = results[name]
        print(
            f"{r.method:<10}"
            f"{r.ece_before:>12.4f}"
            f"{r.ece_after:>12.4f}"
            f"{r.ece_improvement:>+10.4f} ({r.ece_improvement_pct:>4.0f}%)"
            f"{r.brier_after:>14.4f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibratekit",
        description="Probability-calibration toolkit for scikit-learn classifiers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="run the deterministic synthetic demo")
    p_demo.add_argument("--bins", type=int, default=10, help="number of calibration bins")
    p_demo.set_defaults(func=_cmd_demo)

    p_report = sub.add_parser("report", help="metrics + diagram from a CSV of y_true,y_prob")
    p_report.add_argument("csv", help="path to CSV with columns y_true,y_prob")
    p_report.add_argument("--bins", type=int, default=10, help="number of calibration bins")
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
