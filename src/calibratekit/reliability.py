"""ASCII reliability diagram rendering.

A reliability diagram plots observed accuracy against predicted confidence
per bin. A perfectly calibrated model lies on the diagonal (accuracy ==
confidence); bars are drawn relative to that diagonal so miscalibration is
visible at a glance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .metrics import BinStat, bin_stats, expected_calibration_error


def reliability_diagram(
    y_true: NDArray[np.float64] | list[float],
    y_prob: NDArray[np.float64] | list[float],
    n_bins: int = 10,
    width: int = 40,
) -> str:
    """Return a multi-line ASCII reliability diagram.

    Each row shows a bin's probability range, a bar whose length encodes the
    observed accuracy, the perfectly-calibrated marker ``|`` at the bin's
    confidence, the sample count, and the signed gap.
    """
    stats = bin_stats(y_true, y_prob, n_bins=n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    lines: list[str] = []
    header = f"Reliability diagram ({n_bins} bins)  |  ECE = {ece:.4f}"
    lines.append(header)
    lines.append("=" * max(len(header), width + 30))
    lines.append(f"{'range':>11}  {'diagram':<{width}}  {'n':>5}  {'gap':>7}")
    lines.append("-" * max(len(header), width + 30))

    for st in stats:
        rng = f"{st.lo:.1f}-{st.hi:.1f}"
        lines.append(f"{rng:>11}  {_bar(st, width)}  {st.count:>5}  {_gap_str(st)}")

    lines.append("-" * max(len(header), width + 30))
    lines.append("legend: '#' observed accuracy, '|' predicted confidence")
    return "\n".join(lines)


def _bar(st: BinStat, width: int) -> str:
    if st.count == 0:
        return "." * width
    cells = ["."] * width
    acc_cells = round(st.accuracy * (width - 1))
    for i in range(acc_cells + 1):
        cells[i] = "#"
    conf_pos = round(st.confidence * (width - 1))
    conf_pos = min(max(conf_pos, 0), width - 1)
    # Confidence marker overlays the bar (shows over/under-confidence).
    cells[conf_pos] = "|" if cells[conf_pos] == "." else "+"
    return "".join(cells)


def _gap_str(st: BinStat) -> str:
    if st.count == 0 or np.isnan(st.gap):
        return "     --"
    return f"{st.gap:+.3f}"
