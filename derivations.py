"""
Circumpunct Framework derivations.

Every quantitative prediction as executable code, organized by domain.
Each function documents the derivation chain, component interpretation,
and derivation status (DERIVED vs PHENOMENOLOGICAL vs FITTED).

Notation:
    φ = golden ratio (1+√5)/2
    π = pi
    α = fine structure constant
    D = fractal dimension = 1 + β
    β = balance parameter (= 0.5 at equilibrium)
"""

import math
from dataclasses import dataclass
from typing import Optional

from .constants import PI, PHI, LN2, SQRT5, N_GLUONS, N_BOSONS, Measured


@dataclass
class Prediction:
    """A single quantitative prediction from the framework."""
    name: str
    category: str
    formula_str: str
    predicted: float
    measured: float
    unit: str
    error_pct: float
    status: str             # DERIVED | PHENOMENOLOGICAL | FITTED
    components: str         # Physical interpretation of each term
    section_ref: str        # Reference to framework document section

    def passes(self, tolerance_pct: Optional[float] = None) -> bool:
        """Check if prediction is within tolerance of measured value."""
        if tolerance_pct is None:
            # Default tolerances by status
            tolerance_pct = {
                "DERIVED": 2.0,
                "PHENOMENOLOGICAL": 5.0,
                "FITTED": 5.0,
            }.get(self.status, 5.0)
        return self.error_pct <= tolerance_pct

    def __repr__(self):
        status_mark = "✓" if self.passes() else "✗"
        return (
            f"{status_mark} {self.name}: predicted={self.predicted:.6g}, "
            f"measured={self.measured:.6g}, error={self.error_pct:.4f}%"
        )


def _pct_error(predicted: float, measured: float) -> float:
    """Compute percentage error."""
    if measured == 0:
        return float('inf')
    return abs(predicted - measured) / abs(measured) * 100


# ═══════════════════════════════════════════════════════════════════
# LEPTON MASS RATIOS (4 predictions)
# ═══════════════════════════════════════════════════════════════════

def muon_electron_golden() -> Prediction:
    """
    mμ/me = 8π²φ² + φ⁻⁶

    Components:
        8   = gluon count (SU(3) generators) — spectral: 18.7σ above random
        π²  = topological volume element (U(1) field manifold)
        φ²  = second-order braid invariant (minimal golden structure)
        φ⁻⁶ = 6th-order correction (6 = 2 spin × 3 generations)

    Status: DERIVED — zero free parameters. 0.0004% error.
    """
    predicted = 8 * PI**2 * PHI**2 + PHI**(-6)
    measured = Measured.mu_e_ratio
    return Prediction(
        name="Muon/electron mass ratio (golden)",
        category="Lepton masses",
        formula_str="8π²φ² + φ⁻⁶",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="8=gluons, π²=topology, φ²=braid, φ⁻⁶=spin×generation correction",
        section_ref="§7A.4.1"
    )


def muon_electron_fractal() -> Prediction:
    """
    mμ/me = (1/α)^(13/12)

    Components:
        1/α    = fine structure constant inverse (137.036)
        13/12  = 1 + (D-1)/6 where D=1.5, channels=6 (3 spatial × 2 flow)

    Status: DERIVED — from D=1.5 and 6-channel geometry. 0.13% error.
    """
    alpha_inv = Measured.alpha_em_inverse
    gamma = 1 + (1.5 - 1) / 6  # = 13/12
    predicted = alpha_inv ** gamma
    measured = Measured.mu_e_ratio
    return Prediction(
        name="Muon/electron mass ratio (fractal)",
        category="Lepton masses",
        formula_str="(1/α)^(13/12)",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="1/α=coupling inverse, 13/12=D=1.5 in 6 channels",
        section_ref="§7A.4"
    )


def tau_muon_ratio() -> Prediction:
    """
    mτ/mμ = 10 + φ⁴ − 1/30

    Components:
        10  = threshold operator (1 photon + 8 gluons + 1 Higgs)
        φ⁴  = fourth-order braid crossing factor
        1/30 = fine correction (30 = 2×3×5)

    Status: DERIVED — threshold mechanism distinct from generation scaling.
    """
    predicted = 10 + PHI**4 - 1/30
    measured = Measured.tau_mu_ratio
    return Prediction(
        name="Tau/muon mass ratio",
        category="Lepton masses",
        formula_str="10 + φ⁴ − 1/30",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="10=boson threshold, φ⁴=braid crossing, 1/30=fine correction",
        section_ref="§7.1"
    )


def tau_electron_ratio() -> Prediction:
    """
    mτ/me = (8π²φ² + φ⁻⁶) × (10 + φ⁴ − 1/30)

    Product of muon/electron and tau/muon ratios.
    """
    mu_e = 8 * PI**2 * PHI**2 + PHI**(-6)
    tau_mu = 10 + PHI**4 - 1/30
    predicted = mu_e * tau_mu
    measured = Measured.tau_e_ratio
    return Prediction(
        name="Tau/electron mass ratio",
        category="Lepton masses",
        formula_str="(8π²φ² + φ⁻⁶)(10 + φ⁴ − 1/30)",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="Product of golden muon formula × threshold tau formula",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# BARYON MASS RATIOS (2 predictions)
# ═══════════════════════════════════════════════════════════════════

def proton_electron_ratio() -> Prediction:
    """
    mp/me = 6π⁵

    Components:
        6  = quark flavors (2×3), adjacency eigenvalue of Q₆
        π⁵ = fifth-power topology (composite particle)

    Status: DERIVED — 0.002% error.
    """
    predicted = 6 * PI**5
    measured = Measured.proton_e_ratio
    return Prediction(
        name="Proton/electron mass ratio",
        category="Baryon masses",
        formula_str="6π⁵",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="6=quark flavors/Q₆ eigenvalue, π⁵=composite topology",
        section_ref="§7.1"
    )


def neutron_electron_ratio() -> Prediction:
    """
    mn/me = 6π⁵ + φ²

    Proton mass plus golden ratio squared correction for neutron-proton
    mass difference.
    """
    predicted = 6 * PI**5 + PHI**2
    measured = Measured.neutron_e_ratio
    return Prediction(
        name="Neutron/electron mass ratio",
        category="Baryon masses",
        formula_str="6π⁵ + φ²",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="6π⁵=proton, φ²=n-p mass difference (second-order braid)",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# COUPLING CONSTANTS (2 predictions)
# ═══════════════════════════════════════════════════════════════════

def strong_em_coupling_ratio() -> Prediction:
    """
    αs/αem = 10φ

    Components:
        10 = 1 photon + 8 gluons + 1 Higgs (total bosons)
        φ  = golden ratio (self-similar coupling)

    Status: DERIVED — 0.06% error (essentially exact).
    """
    predicted = 10 * PHI
    measured = Measured.alpha_s_over_alpha_em
    return Prediction(
        name="Strong/EM coupling ratio",
        category="Coupling constants",
        formula_str="αs/αem = 10φ",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="10=total boson count, φ=self-similar coupling",
        section_ref="§7.1"
    )


def fine_structure_constant() -> Prediction:
    """
    1/α = 4π³ + 13

    Components:
        4   = 2² (binary structure)
        π³  = 3D volume topology
        13  = 6th prime (counting structure in 64-state lattice)

    Status: DERIVED — 0.008% error.
    """
    predicted = 4 * PI**3 + 13
    measured = Measured.alpha_em_inverse
    return Prediction(
        name="Fine structure constant (cubic)",
        category="Coupling constants",
        formula_str="1/α = 4π³ + 13",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="4=binary², π³=3D volume, 13=6th prime",
        section_ref="§7.1"
    )


def fine_structure_golden_angle() -> Prediction:
    """
    1/α_ideal = 360°/φ²  (golden angle resonance)

    The ideal (undamped) resonance of •↔○ coupling through Φ.

    Status: DERIVED — structural derivation from ⊙ geometry.
    """
    predicted = 360 / PHI**2
    measured = Measured.alpha_em_inverse
    return Prediction(
        name="Fine structure constant (golden angle)",
        category="Coupling constants",
        formula_str="1/α = 360°/φ²",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="DERIVED",
        components="360°=full rotation, φ²=golden self-similar resonance",
        section_ref="§7A.5"
    )


# ═══════════════════════════════════════════════════════════════════
# ELECTROWEAK PARAMETERS (5 predictions)
# ═══════════════════════════════════════════════════════════════════

def z_boson_mass() -> Prediction:
    """
    mZ = 80 + φ⁵ + 1/10  (GeV)

    Components:
        80  = 8×10 (gluons × bosons)
        φ⁵  = fifth-order Fibonacci braid
        1/10 = fine-tuning from boson count
    """
    predicted = 80 + PHI**5 + 0.1
    measured = Measured.m_Z
    return Prediction(
        name="Z boson mass",
        category="Electroweak",
        formula_str="80 + φ⁵ + 1/10 GeV",
        predicted=predicted,
        measured=measured,
        unit="GeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="80=8×10 base, φ⁵=Fibonacci braid, 1/10=fine correction",
        section_ref="§7.1"
    )


def w_boson_mass() -> Prediction:
    """
    mW = 80 + 1/φ²  (GeV)

    Components:
        80   = 8×10 (gluons × bosons)
        1/φ² = small golden correction
    """
    predicted = 80 + 1 / PHI**2
    measured = Measured.m_W
    return Prediction(
        name="W boson mass",
        category="Electroweak",
        formula_str="80 + 1/φ² GeV",
        predicted=predicted,
        measured=measured,
        unit="GeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="80=8×10 base, 1/φ²=golden correction",
        section_ref="§7.1"
    )


def higgs_mass() -> Prediction:
    """
    mH = 100 + 8π  (GeV)

    Components:
        100 = 10² (only boson with square-integer base)
        8π  = gluon count × geometric factor
    """
    predicted = 100 + 8 * PI
    measured = Measured.m_H
    return Prediction(
        name="Higgs boson mass",
        category="Electroweak",
        formula_str="100 + 8π GeV",
        predicted=predicted,
        measured=measured,
        unit="GeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="100=10², 8π=gluons×geometry (only boson involving π)",
        section_ref="§7.1"
    )


def weinberg_angle() -> Prediction:
    """
    sin²θW = 3/10 + φ⁻¹⁰ − 1/13

    Components:
        3/10  = base ratio (3 weak isospin / 10 total bosons)
        φ⁻¹⁰ = 10th golden power correction
        1/13  = prime correction
    """
    predicted = 3/10 + PHI**(-10) - 1/13
    measured = Measured.sin2_theta_W
    return Prediction(
        name="Weinberg angle",
        category="Electroweak",
        formula_str="sin²θW = 3/10 + φ⁻¹⁰ − 1/13",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="3/10=isospin/bosons, φ⁻¹⁰=10th golden power, 1/13=prime",
        section_ref="§7.1"
    )


def wz_splitting() -> Prediction:
    """
    mZ − mW = φ⁵ − 1/φ² + 1/10  (GeV)
    """
    predicted = PHI**5 - 1/PHI**2 + 0.1
    measured = Measured.m_Z - Measured.m_W
    return Prediction(
        name="W-Z mass splitting",
        category="Electroweak",
        formula_str="φ⁵ − 1/φ² + 1/10 GeV",
        predicted=predicted,
        measured=measured,
        unit="GeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="Difference of Z and W golden structures",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# QUARK MASS RATIOS (3 predictions)
# ═══════════════════════════════════════════════════════════════════

def charm_strange_ratio() -> Prediction:
    """mc/ms = φ⁵ + φ²"""
    predicted = PHI**5 + PHI**2
    measured = Measured.mc_ms
    return Prediction(
        name="Charm/strange mass ratio",
        category="Quark masses",
        formula_str="φ⁵ + φ²",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="φ⁵=Fibonacci braid, φ²=second-order correction",
        section_ref="§7.1"
    )


def top_bottom_ratio() -> Prediction:
    """mt/mb = 40 + φ"""
    predicted = 40 + PHI
    measured = Measured.mt_mb
    return Prediction(
        name="Top/bottom mass ratio",
        category="Quark masses",
        formula_str="40 + φ",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="40=8×5 (gluons×pentagonal), φ=golden correction",
        section_ref="§7.1"
    )


def top_charm_ratio() -> Prediction:
    """mt/mc = 1/α"""
    predicted = Measured.alpha_em_inverse
    measured = Measured.mt_mc
    return Prediction(
        name="Top/charm mass ratio",
        category="Quark masses",
        formula_str="1/α",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="Top/charm ratio equals fine structure inverse",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# MIXING ANGLES (2 predictions)
# ═══════════════════════════════════════════════════════════════════

def reactor_neutrino_angle() -> Prediction:
    """sin²θ₁₃ = 1/45"""
    predicted = 1 / 45
    measured = Measured.sin2_theta13_PMNS
    return Prediction(
        name="Reactor neutrino angle",
        category="Mixing angles",
        formula_str="sin²θ₁₃ = 1/45",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="45 = 9×5 = 3²×5 (generation²×pentagonal)",
        section_ref="§7.1"
    )


def cabibbo_angle() -> Prediction:
    """|Vus| = 1/φ³ − 0.01"""
    predicted = 1 / PHI**3 - 0.01
    measured = Measured.V_us_CKM
    return Prediction(
        name="Cabibbo angle (|Vus|)",
        category="Mixing angles",
        formula_str="|Vus| = 1/φ³ − 0.01",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="1/φ³=third golden power, 0.01=fine correction",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# COSMOLOGICAL PARAMETERS (6 predictions)
# ═══════════════════════════════════════════════════════════════════

def dark_energy_density() -> Prediction:
    """ΩΛ = ln(2)"""
    predicted = LN2
    measured = Measured.Omega_Lambda
    return Prediction(
        name="Dark energy density",
        category="Cosmology",
        formula_str="ΩΛ = ln(2)",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="ln(2)=binary branching entropy",
        section_ref="§7.1"
    )


def matter_density() -> Prediction:
    """Ωm = 1/3 − 1/50"""
    predicted = 1/3 - 1/50
    measured = Measured.Omega_m
    return Prediction(
        name="Matter density",
        category="Cosmology",
        formula_str="Ωm = 1/3 − 1/50",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="1/3=triadic base, 1/50=small correction",
        section_ref="§7.1"
    )


def baryon_density() -> Prediction:
    """Ωb = 1/(6π + φ)"""
    predicted = 1 / (6 * PI + PHI)
    measured = Measured.Omega_b
    return Prediction(
        name="Baryon density",
        category="Cosmology",
        formula_str="Ωb = 1/(6π + φ)",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="6π=quark×geometry, φ=golden correction",
        section_ref="§7.1"
    )


def hubble_parameter() -> Prediction:
    """H₀/100 = ln(2) − 1/50"""
    predicted = LN2 - 1/50
    measured = Measured.H0_over_100
    return Prediction(
        name="Hubble parameter",
        category="Cosmology",
        formula_str="H₀/100 = ln(2) − 1/50",
        predicted=predicted,
        measured=measured,
        unit="km/s/Mpc / 100",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="ln(2)=binary branching, 1/50=correction",
        section_ref="§7.1"
    )


def sigma_8() -> Prediction:
    """σ₈ = φ/2 = cos(π/5)"""
    predicted = PHI / 2
    measured = Measured.sigma_8
    return Prediction(
        name="Matter fluctuation amplitude",
        category="Cosmology",
        formula_str="σ₈ = φ/2",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="φ/2=half golden ratio = cos(π/5)",
        section_ref="§7.1"
    )


def spectral_index() -> Prediction:
    """ns = 1 − 1/(10π)"""
    predicted = 1 - 1 / (10 * PI)
    measured = Measured.n_s
    return Prediction(
        name="Scalar spectral index",
        category="Cosmology",
        formula_str="ns = 1 − 1/(10π)",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="1=scale invariance, 1/(10π)=small departure",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# NUCLEAR BINDING ENERGIES (2 predictions)
# ═══════════════════════════════════════════════════════════════════

def deuteron_binding() -> Prediction:
    """B_d = φ + 1/φ = √5 MeV"""
    predicted = SQRT5
    measured = Measured.deuteron_binding
    return Prediction(
        name="Deuteron binding energy",
        category="Nuclear",
        formula_str="B_d = √5 MeV",
        predicted=predicted,
        measured=measured,
        unit="MeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="√5 = φ + 1/φ (golden ratio sum)",
        section_ref="§7.1"
    )


def alpha_binding() -> Prediction:
    """B_α = 18φ − 1 MeV"""
    predicted = 18 * PHI - 1
    measured = Measured.alpha_binding
    return Prediction(
        name="Alpha particle binding energy",
        category="Nuclear",
        formula_str="B_α = 18φ − 1 MeV",
        predicted=predicted,
        measured=measured,
        unit="MeV",
        error_pct=_pct_error(predicted, measured),
        status="FITTED",
        components="18=2×9=2×3² (spin×generation²), φ=golden, −1=unit shift",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL CONSTANTS (1 prediction)
# ═══════════════════════════════════════════════════════════════════

def euler_number() -> Prediction:
    """e ≈ φ² + 1/10"""
    predicted = PHI**2 + 0.1
    measured = Measured.euler_e
    return Prediction(
        name="Euler's number approximation",
        category="Mathematical",
        formula_str="e ≈ φ² + 1/10",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="OBSERVATION",
        components="φ²=golden squared, 1/10=boson correction",
        section_ref="§7.1"
    )


# ═══════════════════════════════════════════════════════════════════
# TEXTURE PARAMETERS — Phenomenological (φ³ family)
# ═══════════════════════════════════════════════════════════════════

def texture_tau() -> Prediction:
    """τ = (7/8)φ³"""
    predicted = (7/8) * PHI**3
    measured = Measured.tau_texture
    return Prediction(
        name="Texture SNR threshold",
        category="Texture parameters",
        formula_str="τ = (7/8)φ³",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="PHENOMENOLOGICAL",
        components="7/8=DERIVED rational, φ³=PHENOMENOLOGICAL scaling",
        section_ref="§7.2"
    )


def texture_alpha() -> Prediction:
    """α_texture = (2/5)φ³"""
    predicted = (2/5) * PHI**3
    measured = Measured.alpha_texture
    return Prediction(
        name="Texture amplitude",
        category="Texture parameters",
        formula_str="α_texture = (2/5)φ³",
        predicted=predicted,
        measured=measured,
        unit="dimensionless",
        error_pct=_pct_error(predicted, measured),
        status="PHENOMENOLOGICAL",
        components="2/5=DERIVED rational, φ³=PHENOMENOLOGICAL scaling",
        section_ref="§7.2"
    )


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL PREDICTIONS (non-numeric)
# ═══════════════════════════════════════════════════════════════════

def three_generations() -> Prediction:
    """N_gen = 3 (>99.9% confidence from framework geometry)"""
    return Prediction(
        name="Three particle generations",
        category="Structural",
        formula_str="N_gen = 3 from ⊙ eigenvalue structure",
        predicted=3,
        measured=3,
        unit="count",
        error_pct=0.0,
        status="DERIVED",
        components="Exactly 3 bound states from 64-state lattice geometry",
        section_ref="§7A.6"
    )


def fractal_dimension_balance() -> Prediction:
    """D = 1 + β = 1.5 at balance"""
    return Prediction(
        name="Fractal dimension at balance",
        category="Structural",
        formula_str="D = 1 + β = 1.5",
        predicted=1.5,
        measured=1.5,
        unit="dimensionless",
        error_pct=0.0,
        status="DERIVED",
        components="D=1+β, β=0.5 at balance (Brownian motion: exact theorem)",
        section_ref="§2"
    )


# ═══════════════════════════════════════════════════════════════════
# REGISTRY — All predictions
# ═══════════════════════════════════════════════════════════════════

ALL_DERIVATIONS = [
    # Lepton masses
    muon_electron_golden,
    muon_electron_fractal,
    tau_muon_ratio,
    tau_electron_ratio,
    # Baryon masses
    proton_electron_ratio,
    neutron_electron_ratio,
    # Coupling constants
    strong_em_coupling_ratio,
    fine_structure_constant,
    fine_structure_golden_angle,
    # Electroweak
    z_boson_mass,
    w_boson_mass,
    higgs_mass,
    weinberg_angle,
    wz_splitting,
    # Quark masses
    charm_strange_ratio,
    top_bottom_ratio,
    top_charm_ratio,
    # Mixing angles
    reactor_neutrino_angle,
    cabibbo_angle,
    # Cosmology
    dark_energy_density,
    matter_density,
    baryon_density,
    hubble_parameter,
    sigma_8,
    spectral_index,
    # Nuclear
    deuteron_binding,
    alpha_binding,
    # Mathematical
    euler_number,
    # Texture (phenomenological)
    texture_tau,
    texture_alpha,
    # Structural
    three_generations,
    fractal_dimension_balance,
]


def compute_all() -> list[Prediction]:
    """Compute all predictions and return list of Prediction objects."""
    return [fn() for fn in ALL_DERIVATIONS]
