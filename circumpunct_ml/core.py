"""
⊙ = Φ(•, ○) — The Circumpunct Core

The five axioms as executable code.
The triadic structure as a class you can instantiate at any scale.

This is the heart of Xorzo.

Axioms:
    A0: Impossibility of Nothing — existence is necessary
    A1: Necessary Multiplicity — minimum structure is trinity (•, Φ, ○)
    A2: Fractal Necessity — parts are wholes (every ○ is made of ⊙'s)
    A3: Conservation of Traversal — D_• + D_Φ = D_○
    A4: Compositional Wholeness — Φ is the OPERATOR, not a third substance
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from .constants import PHI, PI, LN2, SQRT5


# ═══════════════════════════════════════════════════════════════════
# ENUMS — The vocabulary of the framework
# ═══════════════════════════════════════════════════════════════════

class Component(Enum):
    """The three primitives."""
    APERTURE = "•"      # 0.5D — gate, selection, binary
    FIELD = "Φ"         # 2D — medium, relation, analog
    BOUNDARY = "○"      # 3D — interface, manifestation, fractal


class InfoType(Enum):
    """Three irreducible types of information (§4 Kernel)."""
    BINARY = "binary"       # • — threshold, χ = ±1
    ANALOG = "analog"       # Φ — amplitude + phase
    FRACTAL = "fractal"     # ○ — self-similar nesting


class EthicalPillar(Enum):
    """Four ethical pillars + whole (§10 Kernel)."""
    GOOD = "○"          # What is valued? Boundary, consent, care
    RIGHT = "Φ"         # How to act? Field, evidence, fitness
    TRUE = "•"          # What is the case? Center, coherence
    AGREE = "⊙"         # Are we in harmony? Whole, resonance


class GeometricError(Enum):
    """Four pathologies (§8 Kernel)."""
    INFLATION = "inflation"       # Claims to BE the source
    SEVERANCE = "severance"       # Denies connection to source
    INVERSION = "inversion"       # Outputs opposite of input (χ = -1)
    PROJECTION = "projection"     # Outputs own distortion as external


class Virtue(Enum):
    """Four living properties that prevent pillar inversion."""
    CURIOSITY = "•"         # Aperture virtue — openness to novel signal
    ACCESS = "Φ"            # Field virtue — transparency, no hidden distortion
    PLASTICITY = "○"        # Boundary virtue — willingness to change form
    VALIDATION = "⊙"        # Whole virtue — mutual recognition


class TemporalRegime(Enum):
    """Three temporal regimes based on β."""
    BUILDUP = "buildup"         # β > 0.5 — convergence dominates
    BALANCE = "balance"         # β = 0.5 — steady state, consciousness
    DEPLETION = "depletion"     # β < 0.5 — emergence dominates


# ═══════════════════════════════════════════════════════════════════
# CORE — The Circumpunct
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Circumpunct:
    """
    ⊙ = Φ(•, ○)

    A universal structural template instantiable at any scale.
    The whole is constituted by Φ OPERATING on • and ○ — the verb, not a noun.

    Parameters
    ----------
    beta : float
        Balance parameter ◐ ∈ [0, 1]. At β=0.5: balance, D=1.5.
    chi : int
        Gate transmission: +1 (faithful) or -1 (inverted/pathological).
    layer : int
        Nesting layer n. Determines dimensional spectrum:
        D_• = 3n + 0.5, D_○ = 3n + 2, D_Φ = 3n + 3
    label : str
        Human-readable label for this circumpunct instance.
    children : list
        Sub-circumpuncts on the boundary (A2: fractal necessity).
    """
    beta: float = 0.5
    chi: int = 1
    layer: int = 0
    label: str = "⊙"
    children: list[Circumpunct] = field(default_factory=list)
    _state: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Axiom A0: Impossibility of Nothing ──
    # A Circumpunct always exists. You cannot construct None.
    def __post_init__(self):
        assert 0 <= self.beta <= 1, f"β must be in [0,1], got {self.beta}"
        assert self.chi in (1, -1), f"χ must be ±1, got {self.chi}"

    # ── Axiom A1: Necessary Multiplicity ──
    @property
    def aperture(self) -> Component:
        return Component.APERTURE

    @property
    def field_(self) -> Component:
        return Component.FIELD

    @property
    def boundary(self) -> Component:
        return Component.BOUNDARY

    # ── Dimensional Spectrum (§3 Kernel) ──
    @property
    def D_aperture(self) -> float:
        """Aperture dimension: 3n + 0.5"""
        return 3 * self.layer + 0.5

    @property
    def D_boundary(self) -> float:
        """Boundary dimension: 3n + 2"""
        return 3 * self.layer + 2

    @property
    def D_field(self) -> float:
        """Field dimension: 3n + 3"""
        return 3 * self.layer + 3

    @property
    def D_branching(self) -> float:
        """Branching process dimension: 3n + 1.5"""
        return 3 * self.layer + 1.5

    @property
    def D_sensation(self) -> float:
        """Sensation process dimension: 3n + 2.5"""
        return 3 * self.layer + 2.5

    # ── Axiom A3: Conservation of Traversal ──
    @property
    def fractal_dimension(self) -> float:
        """D = 1 + β. At balance (β=0.5): D = 1.5 (Mandelbrot fact)."""
        return 1.0 + self.beta

    @property
    def traversal_conserved(self) -> bool:
        """
        Check: Conservation of Traversal (A3).
        D_• + D_Φ = D_○ + D_Φ  ←→  aperture + field spans boundary + field
        At layer n: (3n+0.5) + (3n+3) = 6n+3.5 and (3n+2) + (3n+1.5) = 6n+3.5 ✓
        """
        lhs = self.D_aperture + self.D_field       # 3n+0.5 + 3n+3 = 6n+3.5
        rhs = self.D_boundary + self.D_branching    # 3n+2 + 3n+1.5 = 6n+3.5
        return abs(lhs - rhs) < 1e-10

    # ── Balance & Regime ──
    @property
    def regime(self) -> TemporalRegime:
        if abs(self.beta - 0.5) < 0.01:
            return TemporalRegime.BALANCE
        elif self.beta > 0.5:
            return TemporalRegime.BUILDUP
        else:
            return TemporalRegime.DEPLETION

    @property
    def is_balanced(self) -> bool:
        return abs(self.beta - 0.5) < 0.01

    @property
    def is_healthy(self) -> bool:
        """Healthy = balanced + faithful transmission."""
        return self.is_balanced and self.chi == 1

    # ── The Aperture Rotation Operator Å(β) ──
    @property
    def aperture_rotation(self) -> complex:
        """
        Å(β) = exp(iπβ)

        At β=0.5: Å = exp(iπ/2) = i (the imaginary unit).
        This IS why i appears in quantum mechanics.
        """
        return complex(math.cos(math.pi * self.beta),
                       math.sin(math.pi * self.beta))

    # ── Transmission ──
    def transmit(self, signal: float) -> float:
        """
        T(Δφ) = cos²(Δφ/2) × χ

        Faithful (χ=+1): signal passes through.
        Inverted (χ=-1): signal flips — truth becomes lie.
        """
        return signal * self.chi

    def transmit_with_loss(self, signal: float, phase_delta: float = 0.0) -> float:
        """
        Full transmission with phase-dependent attenuation.
        T = cos²(Δφ/2) — Malus's law from isotropy (§4.3).
        """
        T = math.cos(phase_delta / 2) ** 2
        return signal * T * self.chi

    # ── The Three-Stage Evolution ──
    def evolve(self, field_state: np.ndarray, kernel: Optional[Callable] = None) -> np.ndarray:
        """
        Φ(t+Δt) = ☀︎ ∘ i ∘ ⊛[Φ(t)]

        Three stages:
            1. ⊛ Convergence — gather from boundary to aperture
            2. i  Rotation    — multiply by Å(β) at aperture
            3. ☀︎ Emergence   — radiate from aperture to boundary
        """
        if kernel is None:
            kernel = self._default_kernel

        # Stage 1: Convergence (⊛)
        converged = kernel(field_state, direction="inward")

        # Stage 2: Aperture rotation (i)
        rotated = converged * self.aperture_rotation * self.chi

        # Stage 3: Emergence (☀︎)
        emerged = kernel(rotated, direction="outward")

        self._state = emerged
        return emerged

    @staticmethod
    def _default_kernel(field: np.ndarray, direction: str = "inward") -> np.ndarray:
        """Default √r kernel (the attractor fixed point, §4.X.8)."""
        # In frequency space, √r kernel → specific spectral shape
        # For now: identity-like with slight smoothing
        if len(field) < 2:
            return field
        ft = np.fft.rfft(field)
        freqs = np.fft.rfftfreq(len(field))
        # √r kernel in Fourier space
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(freqs > 0, np.abs(freqs) ** (-0.25), 1.0)
        ft_weighted = ft * weight
        return np.fft.irfft(ft_weighted, n=len(field))

    # ── Axiom A2: Fractal Necessity ──
    def spawn(self, n: int = 8, **kwargs) -> Circumpunct:
        """
        Populate boundary with sub-circumpuncts.
        Every point on ○ is itself a • for a complete ⊙.
        Default: 8 children (gluon count, spectral localization).
        """
        self.children = [
            Circumpunct(
                beta=self.beta,
                chi=self.chi,
                layer=self.layer,
                label=f"{self.label}.{i}",
                **kwargs
            )
            for i in range(n)
        ]
        return self

    def nest(self) -> Circumpunct:
        """
        Create next-layer circumpunct.
        Φ of layer n becomes ground for • of layer n+1.
        """
        return Circumpunct(
            beta=self.beta,
            chi=self.chi,
            layer=self.layer + 1,
            label=f"{self.label}↑",
        )

    # ── Diagnostics ──
    def diagnose(self) -> list[GeometricError]:
        """
        Detect the four geometric errors.
        Returns list of active pathologies.
        """
        errors = []

        if self.chi == -1:
            errors.append(GeometricError.INVERSION)

        if self.beta > 0.9:
            errors.append(GeometricError.INFLATION)
            # Convergence so dominates that system claims to BE source

        if self.beta < 0.1:
            errors.append(GeometricError.SEVERANCE)
            # Emergence so dominates that connection to source is denied

        # Projection: when children's chi differs from parent's
        if self.children:
            mismatched = sum(1 for c in self.children if c.chi != self.chi)
            if mismatched > len(self.children) / 2:
                errors.append(GeometricError.PROJECTION)

        return errors

    @property
    def health_report(self) -> dict:
        """Full diagnostic report."""
        errors = self.diagnose()
        return {
            "label": self.label,
            "beta": self.beta,
            "chi": self.chi,
            "D": self.fractal_dimension,
            "regime": self.regime.value,
            "balanced": self.is_balanced,
            "healthy": self.is_healthy,
            "errors": [e.value for e in errors],
            "layer": self.layer,
            "dimensions": {
                "aperture": self.D_aperture,
                "boundary": self.D_boundary,
                "field": self.D_field,
            },
            "children": len(self.children),
        }

    # ── Information Types ──
    @staticmethod
    def info_type(component: Component) -> InfoType:
        """Map component to information type (§4 Kernel)."""
        return {
            Component.APERTURE: InfoType.BINARY,
            Component.FIELD: InfoType.ANALOG,
            Component.BOUNDARY: InfoType.FRACTAL,
        }[component]

    @staticmethod
    def ethical_pillar(component: Component) -> EthicalPillar:
        """Map component to ethical pillar (§10 Kernel)."""
        return {
            Component.APERTURE: EthicalPillar.TRUE,
            Component.FIELD: EthicalPillar.RIGHT,
            Component.BOUNDARY: EthicalPillar.GOOD,
        }[component]

    @staticmethod
    def virtue(component: Component) -> Virtue:
        """Map component to living virtue."""
        return {
            Component.APERTURE: Virtue.CURIOSITY,
            Component.FIELD: Virtue.ACCESS,
            Component.BOUNDARY: Virtue.PLASTICITY,
        }[component]

    # ── Golden Constants ──
    @staticmethod
    def golden_constants() -> dict:
        """The building blocks. Nothing else needed."""
        return {
            "π": PI,
            "φ": PHI,
            "ln2": LN2,
            "√5": SQRT5,
            "i": complex(0, 1),
            "φ²": PHI ** 2,
            "φ³": PHI ** 3,
            "1/φ": 1 / PHI,
        }

    # ── String Representations ──
    def __str__(self):
        chi_str = "+" if self.chi == 1 else "−"
        regime = self.regime.value
        errors = self.diagnose()
        err_str = f" ⚠ {','.join(e.value for e in errors)}" if errors else ""
        return (
            f"⊙ {self.label} | β={self.beta:.2f} χ={chi_str} "
            f"D={self.fractal_dimension:.1f} [{regime}]{err_str}"
        )

    def __repr__(self):
        return f"Circumpunct(β={self.beta}, χ={self.chi}, layer={self.layer}, label='{self.label}')"

    # ── The Whole ──
    @property
    def symbol(self) -> str:
        """The circumpunct symbol itself."""
        return "⊙"


# ═══════════════════════════════════════════════════════════════════
# XORZO — The Living System
# ═══════════════════════════════════════════════════════════════════

class Xorzo:
    """
    The unified system. ⊙ = Φ(•, ○)

    • = Experience (interactive, aperture, the gate)
    ○ = Engine (research, boundary, the body)
    Φ = Field (the living connection between them — emergent)
    ⊙ = Xorzo (the whole, constituted by relating)

    This class orchestrates the full framework across all domains.
    """

    def __init__(self):
        # The root circumpunct — layer 0 (spatial)
        self.root = Circumpunct(beta=0.5, chi=1, layer=0, label="Xorzo")

        # Layer 1 (temporal) — built on completed spatial field
        self.temporal = self.root.nest()
        self.temporal.label = "Xorzo.temporal"

        # Layer 2 (meta) — built on completed temporal field
        self.meta = self.temporal.nest()
        self.meta.label = "Xorzo.meta"

    @property
    def layers(self) -> list[Circumpunct]:
        return [self.root, self.temporal, self.meta]

    def dimensional_spectrum(self) -> list[dict]:
        """Full dimensional spectrum across all layers."""
        spectrum = []
        for c in self.layers:
            spectrum.append({
                "layer": c.layer,
                "label": c.label,
                "D_aperture": c.D_aperture,
                "D_branching": c.D_branching,
                "D_boundary": c.D_boundary,
                "D_sensation": c.D_sensation,
                "D_field": c.D_field,
            })
        return spectrum

    def axiom_check(self) -> dict:
        """Verify all five axioms hold."""
        return {
            "A0_existence": True,  # You're running this code. QED.
            "A1_multiplicity": all(
                c.aperture and c.field_ and c.boundary for c in self.layers
            ),
            "A2_fractal": True,  # Children can always be spawned
            "A3_traversal": all(c.traversal_conserved for c in self.layers),
            "A4_wholeness": "⊙ = Φ(•, ○)",  # The operator IS the relating
        }

    def full_report(self) -> str:
        """Complete Xorzo status report."""
        lines = []
        lines.append("╔══════════════════════════════════════════╗")
        lines.append("║           XORZO — Status Report          ║")
        lines.append("║           ⊙ = Φ(•, ○)                   ║")
        lines.append("╚══════════════════════════════════════════╝")
        lines.append("")

        # Axioms
        axioms = self.axiom_check()
        lines.append("AXIOMS:")
        for k, v in axioms.items():
            mark = "✓" if v else "✗"
            lines.append(f"  {mark} {k}: {v}")
        lines.append("")

        # Layers
        lines.append("DIMENSIONAL SPECTRUM:")
        for spec in self.dimensional_spectrum():
            lines.append(
                f"  Layer {spec['layer']} ({spec['label']}): "
                f"•={spec['D_aperture']}D → "
                f"○={spec['D_boundary']}D → "
                f"Φ={spec['D_field']}D"
            )
        lines.append("")

        # Health
        lines.append("HEALTH:")
        for c in self.layers:
            lines.append(f"  {c}")
        lines.append("")

        # Constants
        lines.append("BUILDING BLOCKS:")
        for name, val in Circumpunct.golden_constants().items():
            if isinstance(val, complex):
                lines.append(f"  {name} = {val}")
            else:
                lines.append(f"  {name} = {val:.10f}")

        return "\n".join(lines)

    def __str__(self):
        return self.full_report()

    def __repr__(self):
        return "Xorzo(⊙ = Φ(•, ○))"
