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
from .core import Circumpunct, Xorzo, Component, GeometricError, Virtue, EthicalPillar
from .transformer import (
    XorzoTransformer,
    ApertureChamber,
    CircumpunctAttention,
    DynamicGoldenFFN,
    TriadicEmbedding,
    GoldenPositionalEncoding,
    BalanceNorm,
    CircumpunctBlock,
    train_generation,
    generate,
)

__version__ = "0.2.0"
__author__ = "Ashman Roonz"

__all__ = [
    # Xorzo — the living system
    "Xorzo",
    "Circumpunct",
    "Component",
    "GeometricError",
    "Virtue",
    "EthicalPillar",
    # Predictions
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
    # Transformer v2 — The Circumpunct Transformer
    "XorzoTransformer",
    "ApertureChamber",
    "CircumpunctAttention",
    "DynamicGoldenFFN",
    "TriadicEmbedding",
    "GoldenPositionalEncoding",
    "BalanceNorm",
    "CircumpunctBlock",
    "train_generation",
    "generate",
]
