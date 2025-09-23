from __future__ import annotations

import numpy as np
import pytest

from calibratekit.cli import main
from calibratekit.demo import make_demo_splits, run_demo


def test_demo_is_deterministic() -> None:
    a = make_demo_splits()
    b = make_demo_splits()
    np.testing.assert_array_equal(a.test_prob, b.test_prob)
    np.testing.assert_array_equal(a.valid_true, b.valid_true)


def test_run_demo_reports_both_methods() -> None:
    results = run_demo()
    assert set(results) == {"isotonic", "platt"}
    for r in results.values():
        assert r.ece_after < r.ece_before


def test_demo_splits_disjoint_sizes() -> None:
    s = make_demo_splits()
    assert s.valid_prob.shape == (2000,)
    assert s.test_prob.shape == (2000,)
    assert s.valid_prob.min() >= 0.0 and s.valid_prob.max() <= 1.0


def test_cli_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["demo", "--bins", "10"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "isotonic" in out
    assert "platt" in out


def test_cli_report_reads_csv(
    tmp_path: object, capsys: pytest.CaptureFixture[str]
) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "scores.csv"
    p.write_text("y_true,y_prob\n1,0.9\n0,0.2\n1,0.8\n0,0.1\n")
    rc = main(["report", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ECE" in out
    assert "Reliability diagram" in out
    assert "samples : 4" in out


def test_cli_report_handles_headerless_csv(
    tmp_path: object, capsys: pytest.CaptureFixture[str]
) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "raw.csv"
    p.write_text("1,0.9\n0,0.2\n")
    rc = main(["report", str(p)])
    assert rc == 0
    assert "samples : 2" in capsys.readouterr().out


def test_cli_report_bad_row_raises(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "bad.csv"
    p.write_text("y_true,y_prob\n1\n")
    with pytest.raises(ValueError):
        main(["report", str(p)])
