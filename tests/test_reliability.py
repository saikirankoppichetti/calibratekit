from __future__ import annotations

from calibratekit.reliability import reliability_diagram


def test_diagram_has_row_per_bin() -> None:
    y_prob = [0.05, 0.15, 0.25, 0.95]
    y_true = [0.0, 0.0, 1.0, 1.0]
    out = reliability_diagram(y_true, y_prob, n_bins=10, width=20)
    # 10 bin rows, each shows a range like "0.0-0.1".
    assert out.count("-0.") + out.count("-1.0") >= 10
    assert "ECE =" in out
    assert "legend:" in out


def test_empty_bins_render_as_dots() -> None:
    out = reliability_diagram([0.0, 1.0], [0.05, 0.95], n_bins=10, width=20)
    lines = out.splitlines()
    # A middle bin (0.4-0.5) is empty and its bar is all dots.
    empty_rows = [ln for ln in lines if ln.strip().startswith("0.4-0.5")]
    assert len(empty_rows) == 1
    assert "." * 20 in empty_rows[0]
    assert empty_rows[0].strip().endswith("--")


def test_confidence_marker_present() -> None:
    out = reliability_diagram([1.0] * 5, [0.6] * 5, n_bins=10, width=30)
    assert "|" in out or "+" in out
