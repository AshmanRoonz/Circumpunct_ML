"""Tests for the full prediction suite."""

import pytest
from circumpunct_ml import run_all_predictions


def test_all_predictions_run():
    """All predictions should compute without error."""
    results = run_all_predictions()
    assert results.total >= 25, f"Expected 25+ predictions, got {results.total}"


def test_all_predictions_have_finite_error():
    """No prediction should have infinite error."""
    results = run_all_predictions()
    for p in results.predictions:
        assert p.error_pct < float('inf'), f"{p.name} has infinite error"


def test_derived_predictions_within_2pct():
    """All DERIVED predictions should be within 2% of measured values."""
    results = run_all_predictions()
    by_status = results.by_status()
    for p in by_status.get("DERIVED", []):
        assert p.error_pct <= 2.0, (
            f"{p.name}: error {p.error_pct:.4f}% exceeds 2% bound"
        )


def test_all_predictions_within_bounds():
    """All predictions should pass their falsification bounds."""
    from circumpunct_ml.falsification import FalsificationSuite
    suite = FalsificationSuite()
    report = suite.run()
    assert report.framework_survives, (
        f"{report.failed} predictions exceeded their falsification bounds"
    )


def test_summary_output():
    """Summary should produce non-empty output."""
    results = run_all_predictions()
    output = results.summary()
    assert len(output) > 100
    assert "Circumpunct" in output


def test_csv_export():
    """CSV export should have header + data rows."""
    results = run_all_predictions()
    csv = results.csv()
    lines = csv.strip().split("\n")
    assert len(lines) == results.total + 1  # header + data


def test_top_predictions():
    """Top predictions should be sorted by error."""
    results = run_all_predictions()
    top = results.top_predictions(5)
    errors = [p.error_pct for p in top]
    assert errors == sorted(errors)
