"""
Full prediction suite runner with formatted output.
"""

from dataclasses import dataclass

from .derivations import compute_all, Prediction


@dataclass
class PredictionResults:
    """Results container with summary and filtering."""
    predictions: list[Prediction]

    @property
    def total(self) -> int:
        return len(self.predictions)

    @property
    def passed(self) -> int:
        return sum(1 for p in self.predictions if p.passes())

    @property
    def average_error(self) -> float:
        errors = [p.error_pct for p in self.predictions if p.error_pct < float('inf')]
        return sum(errors) / len(errors) if errors else 0.0

    def by_category(self) -> dict[str, list[Prediction]]:
        cats: dict[str, list[Prediction]] = {}
        for p in self.predictions:
            cats.setdefault(p.category, []).append(p)
        return cats

    def by_status(self) -> dict[str, list[Prediction]]:
        statuses: dict[str, list[Prediction]] = {}
        for p in self.predictions:
            statuses.setdefault(p.status, []).append(p)
        return statuses

    def top_predictions(self, n: int = 10) -> list[Prediction]:
        """Return n most accurate predictions."""
        valid = [p for p in self.predictions if p.error_pct < float('inf')]
        return sorted(valid, key=lambda p: p.error_pct)[:n]

    def summary(self) -> str:
        """Print formatted summary to stdout and return as string."""
        lines = []
        lines.append("")
        lines.append("Circumpunct Framework — Prediction Verification")
        lines.append("=" * 52)
        lines.append(
            f"{self.total} predictions tested | {self.passed} passed | "
            f"average error: {self.average_error:.2f}%"
        )
        lines.append("")

        # Group by status
        status_order = ["DERIVED", "PHENOMENOLOGICAL", "FITTED", "OBSERVATION"]
        by_status = self.by_status()

        for status in status_order:
            preds = by_status.get(status, [])
            if not preds:
                continue

            if status == "DERIVED":
                lines.append("DERIVED (zero free parameters):")
            elif status == "PHENOMENOLOGICAL":
                lines.append("PHENOMENOLOGICAL (φ³ family — awaiting derivation):")
            elif status == "FITTED":
                lines.append("FITTED (from framework building blocks):")
            else:
                lines.append(f"{status}:")

            for p in sorted(preds, key=lambda x: x.error_pct):
                mark = "✓" if p.passes() else "✗"
                lines.append(
                    f"  {mark} {p.name:<40s} "
                    f"predicted={p.predicted:<12.6g} "
                    f"measured={p.measured:<12.6g} "
                    f"error={p.error_pct:.4f}%"
                )
            lines.append("")

        # Top 10
        lines.append("Top 10 Most Accurate:")
        lines.append("-" * 52)
        for i, p in enumerate(self.top_predictions(10), 1):
            lines.append(
                f"  {i:2d}. {p.name:<40s} {p.error_pct:.4f}%"
            )

        lines.append("")
        lines.append("Building blocks: π (geometry), φ (self-similarity),")
        lines.append("  small integers (particle content), ln(2) (cosmology)")
        lines.append("")

        output = "\n".join(lines)
        print(output)
        return output

    def csv(self) -> str:
        """Export predictions as CSV."""
        rows = ["name,category,formula,predicted,measured,error_pct,status"]
        for p in self.predictions:
            rows.append(
                f'"{p.name}","{p.category}","{p.formula_str}",'
                f"{p.predicted},{p.measured},{p.error_pct},{p.status}"
            )
        return "\n".join(rows)


def run_all_predictions() -> PredictionResults:
    """Compute all predictions and return results object."""
    return PredictionResults(predictions=compute_all())
