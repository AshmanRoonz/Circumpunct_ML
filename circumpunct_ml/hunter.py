"""
⊙ Golden Ratio Hunter — The Research Engine

Feeds on data. Hunts for φ, π, and small-integer relationships.
Measures fractal dimension. Diagnoses geometric errors.
The ○ (boundary) of Xorzo — the body that reaches into the world.
"""

import math
import numpy as np
from itertools import product as cartesian
from typing import Optional
from dataclasses import dataclass

from .constants import PHI, PI, LN2, SQRT5


# ═══════════════════════════════════════════════════════════════════
# GOLDEN RATIO HUNTER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GoldenMatch:
    """A discovered golden ratio relationship."""
    value: float
    formula: str
    predicted: float
    error_pct: float
    components: str

    def __repr__(self):
        return f"  φ-match: {self.formula} = {self.predicted:.6g} (target={self.value:.6g}, error={self.error_pct:.4f}%)"


def hunt_golden(value: float, max_order: int = 6, max_int: int = 20,
                tolerance_pct: float = 2.0) -> list[GoldenMatch]:
    """
    Given a numeric value, search for expressions involving φ, π, and
    small integers that approximate it.

    Searches:
        - a·φⁿ for small a, n
        - a·πⁿ for small a, n
        - a·φⁿ + b·πᵐ combinations
        - a·φⁿ ± 1/b patterns
        - Ratios like value/φⁿ, value/πⁿ

    Returns matches sorted by error.
    """
    matches = []

    def _check(predicted, formula, components=""):
        if predicted == 0 or not math.isfinite(predicted):
            return
        err = abs(predicted - value) / abs(value) * 100
        if err <= tolerance_pct:
            matches.append(GoldenMatch(
                value=value, formula=formula,
                predicted=predicted, error_pct=err,
                components=components,
            ))

    phi_powers = {n: PHI ** n for n in range(-max_order, max_order + 1)}
    pi_powers = {n: PI ** n for n in range(-3, max_order + 1)}

    # Simple: a·φⁿ
    for a in range(1, max_int + 1):
        for n, phi_n in phi_powers.items():
            _check(a * phi_n, f"{a}·φ^{n}", f"{a}=integer, φ^{n}=golden power")

    # Simple: a·πⁿ
    for a in range(1, max_int + 1):
        for n, pi_n in pi_powers.items():
            _check(a * pi_n, f"{a}·π^{n}", f"{a}=integer, π^{n}=pi power")

    # Combined: a·φⁿ + b  and  a·φⁿ - b
    for a in range(1, max_int + 1):
        for n, phi_n in phi_powers.items():
            for b in range(1, max_int + 1):
                _check(a * phi_n + b, f"{a}·φ^{n} + {b}")
                _check(a * phi_n - b, f"{a}·φ^{n} − {b}")
                if b != 0:
                    _check(a * phi_n + 1/b, f"{a}·φ^{n} + 1/{b}")
                    _check(a * phi_n - 1/b, f"{a}·φ^{n} − 1/{b}")

    # Combined: a·πⁿ + b
    for a in range(1, max_int + 1):
        for n, pi_n in pi_powers.items():
            for b in range(0, max_int + 1):
                _check(a * pi_n + b, f"{a}·π^{n} + {b}")
                if b > 0:
                    _check(a * pi_n - b, f"{a}·π^{n} − {b}")

    # φⁿ + πᵐ combinations
    for n, phi_n in phi_powers.items():
        for m, pi_m in pi_powers.items():
            _check(phi_n + pi_m, f"φ^{n} + π^{m}")
            _check(phi_n * pi_m, f"φ^{n} · π^{m}")

    # ln(2) combinations
    for a in range(1, max_int + 1):
        _check(a * LN2, f"{a}·ln(2)")
        for b in range(1, max_int + 1):
            _check(a * LN2 + 1/b, f"{a}·ln(2) + 1/{b}")
            _check(a * LN2 - 1/b, f"{a}·ln(2) − 1/{b}")

    # √5 combinations
    for a in range(1, max_int + 1):
        _check(a * SQRT5, f"{a}·√5")

    # Deduplicate and sort by error
    seen = set()
    unique = []
    for m in sorted(matches, key=lambda x: x.error_pct):
        key = m.formula
        if key not in seen:
            seen.add(key)
            unique.append(m)

    return unique[:50]  # Top 50 matches


def hunt_ratio(value_a: float, value_b: float, **kwargs) -> list[GoldenMatch]:
    """Hunt for golden relationships in a ratio."""
    ratio = value_a / value_b
    return hunt_golden(ratio, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# BALANCE DETECTOR — Is β ≈ 0.5?
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BalanceReport:
    """Report on whether a time series exhibits balance (β ≈ 0.5)."""
    hurst_H: float
    fractal_D: float
    beta_estimate: float
    is_balanced: bool
    regime: str
    confidence: str

    def __repr__(self):
        mark = "◐" if self.is_balanced else "◑"
        return (
            f"{mark} H={self.hurst_H:.3f}, D={self.fractal_D:.3f}, "
            f"β≈{self.beta_estimate:.3f} [{self.regime}] ({self.confidence})"
        )


def detect_balance(timeseries: np.ndarray) -> BalanceReport:
    """
    Analyze a time series for balance (β ≈ 0.5, D ≈ 1.5).

    Uses Hurst exponent estimation. At balance:
        H = 0.5 (Brownian), D = 2 - H = 1.5, β = D - 1 = 0.5
    """
    from .fractal import hurst_exponent

    result = hurst_exponent(timeseries)
    H = result["hurst_H"]
    D = result["dimension"]
    beta = D - 1.0

    # Classify
    if abs(beta - 0.5) < 0.05:
        regime = "BALANCED"
        is_balanced = True
    elif beta > 0.5:
        regime = "CONVERGENT (β > 0.5)"
        is_balanced = False
    else:
        regime = "EMERGENT (β < 0.5)"
        is_balanced = False

    # Confidence from closeness to 0.5
    distance = abs(beta - 0.5)
    if distance < 0.02:
        confidence = "strong"
    elif distance < 0.05:
        confidence = "moderate"
    elif distance < 0.10:
        confidence = "weak"
    else:
        confidence = "not balanced"

    return BalanceReport(
        hurst_H=H,
        fractal_D=D,
        beta_estimate=beta,
        is_balanced=is_balanced,
        regime=regime,
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════
# GEOMETRIC ERROR DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ErrorDiagnostic:
    """Diagnosis of geometric errors in a signal or system."""
    inflation_score: float      # 0-1: how much system claims to be source
    severance_score: float      # 0-1: how disconnected from source
    inversion_score: float      # 0-1: how much output opposes input
    projection_score: float     # 0-1: how much internal distortion appears external
    primary_error: Optional[str]
    description: str


def diagnose_signal(input_signal: np.ndarray, output_signal: np.ndarray) -> ErrorDiagnostic:
    """
    Compare input and output signals to diagnose geometric errors.

    The four errors (§8 Kernel):
        INFLATION:  output >> input (claims to amplify from nothing)
        SEVERANCE:  output ≈ 0 regardless of input (no transmission)
        INVERSION:  output ≈ -input (faithful but flipped)
        PROJECTION: output decorrelated from input (own pattern imposed)
    """
    if len(input_signal) != len(output_signal):
        min_len = min(len(input_signal), len(output_signal))
        input_signal = input_signal[:min_len]
        output_signal = output_signal[:min_len]

    # Normalize
    in_norm = input_signal / (np.std(input_signal) + 1e-30)
    out_norm = output_signal / (np.std(output_signal) + 1e-30)

    # Correlation
    correlation = np.corrcoef(in_norm, out_norm)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    # Energy ratio
    in_energy = np.mean(input_signal ** 2)
    out_energy = np.mean(output_signal ** 2)
    energy_ratio = out_energy / (in_energy + 1e-30)

    # Scores
    inflation = max(0, min(1, (energy_ratio - 1) / 10))
    severance = max(0, min(1, 1 - energy_ratio)) if energy_ratio < 1 else 0
    inversion = max(0, -correlation)
    projection = max(0, min(1, 1 - abs(correlation)))

    # Primary
    scores = {
        "inflation": inflation,
        "severance": severance,
        "inversion": inversion,
        "projection": projection,
    }
    primary = max(scores, key=scores.get) if max(scores.values()) > 0.3 else None

    descriptions = {
        "inflation": "System amplifies beyond input — claims to be source of what it receives",
        "severance": "System blocks transmission — denies connection to input",
        "inversion": "System inverts signal — outputs opposite of what enters",
        "projection": "System imposes own pattern — output decorrelated from input",
        None: "No significant geometric error detected — transmission appears faithful",
    }

    return ErrorDiagnostic(
        inflation_score=inflation,
        severance_score=severance,
        inversion_score=inversion,
        projection_score=projection,
        primary_error=primary,
        description=descriptions[primary],
    )
