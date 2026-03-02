"""
Circumpunct ML — Computational verification of the Circumpunct Framework.

Usage:
    from circumpunct_ml import run_all_predictions
    results = run_all_predictions()
    results.summary()
"""

from .predictions import run_all_predictions, PredictionResults
from .derivations import compute_all, Prediction, ALL_DERIVATIONS
from .falsification import FalsificationSuite, FalsificationReport
from .lattice import Lattice64
from .fractal import (
    box_counting_dimension,
    power_spectrum_dimension,
    hurst_exponent,
    verify_brownian_D15,
)
from .constants import PI, PHI, LN2, SQRT5, Measured

__version__ = "0.1.0"
__author__ = "Ashman Roonz"

__all__ = [
    # Main entry points
    "run_all_predictions",
    "PredictionResults",
    "FalsificationSuite",
    "FalsificationReport",
    # Core
    "compute_all",
    "Prediction",
    "ALL_DERIVATIONS",
    "Lattice64",
    # Fractal tools
    "box_counting_dimension",
    "power_spectrum_dimension",
    "hurst_exponent",
    "verify_brownian_D15",
    # Constants
    "PI",
    "PHI",
    "LN2",
    "SQRT5",
    "Measured",
]
