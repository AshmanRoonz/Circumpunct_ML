"""
Falsification testing for the Circumpunct Framework.

Each prediction specifies conditions under which it fails.
If ANY prediction exceeds its error bound, the framework
fails that specific test.

This is the scientific core: the framework is falsifiable.
"""

from dataclasses import dataclass
from typing import Optional

from .derivations import compute_all, Prediction


@dataclass
class FalsificationResult:
    """Result of a single falsification test."""
    prediction: Prediction
    bound_pct: float
    passed: bool
    margin: float           # How far from the bound (negative = exceeded)

    def __repr__(self):
        if self.passed:
            return (
                f"  SURVIVES: {self.prediction.name} — "
                f"error {self.prediction.error_pct:.4f}% "
                f"within bound {self.bound_pct}% "
                f"(margin: {self.margin:.4f}%)"
            )
        else:
            return (
                f"  *** FALSIFIED: {self.prediction.name} — "
                f"predicted {self.prediction.predicted:.6g}, "
                f"measured {self.prediction.measured:.6g}, "
                f"error {self.prediction.error_pct:.4f}% "
                f"EXCEEDS bound of {self.bound_pct}%"
            )


@dataclass
class FalsificationReport:
    """Complete falsification report."""
    results: list[FalsificationResult]

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def framework_survives(self) -> bool:
        """Framework survives only if ALL tests pass."""
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = []
        lines.append("")
        lines.append("Circumpunct Framework — Falsification Report")
        lines.append("=" * 52)

        if self.framework_survives:
            lines.append(
                f"STATUS: SURVIVES ({self.passed}/{self.total_tests} tests passed)"
            )
        else:
            lines.append(
                f"STATUS: FALSIFIED ({self.failed}/{self.total_tests} tests FAILED)"
            )

        lines.append("")

        # Show failures first
        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append("FAILURES:")
            for r in failures:
                lines.append(repr(r))
            lines.append("")

        # Show passes grouped by category
        passes = [r for r in self.results if r.passed]
        if passes:
            lines.append(f"SURVIVING ({len(passes)} predictions):")
            for r in sorted(passes, key=lambda x: x.prediction.error_pct):
                lines.append(repr(r))

        lines.append("")
        output = "\n".join(lines)
        print(output)
        return output


# Default falsification bounds by prediction status
DEFAULT_BOUNDS = {
    "DERIVED": 2.0,             # Derived predictions must be within 2%
    "PHENOMENOLOGICAL": 5.0,    # Phenomenological: 5% (awaiting derivation)
    "FITTED": 5.0,              # Fitted: 5% (building block expressions)
    "OBSERVATION": 10.0,        # Mathematical observations: 10%
}


class FalsificationSuite:
    """
    Run falsification tests against all framework predictions.

    Each prediction is tested against its error bound. If the prediction's
    error exceeds the bound, the framework is FALSIFIED for that prediction.

    Custom bounds can override defaults per prediction name.
    """

    def __init__(self, custom_bounds: Optional[dict[str, float]] = None):
        self.custom_bounds = custom_bounds or {}

    def _get_bound(self, prediction: Prediction) -> float:
        """Get falsification bound for a prediction."""
        if prediction.name in self.custom_bounds:
            return self.custom_bounds[prediction.name]
        return DEFAULT_BOUNDS.get(prediction.status, 5.0)

    def run(self) -> FalsificationReport:
        """Run all falsification tests."""
        predictions = compute_all()
        results = []

        for p in predictions:
            bound = self._get_bound(p)
            margin = bound - p.error_pct
            passed = p.error_pct <= bound

            results.append(FalsificationResult(
                prediction=p,
                bound_pct=bound,
                passed=passed,
                margin=margin,
            ))

        return FalsificationReport(results=results)

    def run_strict(self) -> FalsificationReport:
        """Run with strict bounds (1% for derived, 2% for others)."""
        predictions = compute_all()
        strict_bounds = {
            "DERIVED": 1.0,
            "PHENOMENOLOGICAL": 2.0,
            "FITTED": 2.0,
            "OBSERVATION": 5.0,
        }
        results = []

        for p in predictions:
            bound = strict_bounds.get(p.status, 2.0)
            margin = bound - p.error_pct
            results.append(FalsificationResult(
                prediction=p,
                bound_pct=bound,
                passed=p.error_pct <= bound,
                margin=margin,
            ))

        return FalsificationReport(results=results)
