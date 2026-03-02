"""
Physical constants and measured values for the Circumpunct Framework.

Sources:
    - CODATA 2022 recommended values (fundamental constants)
    - Particle Data Group (PDG) 2024 Review of Particle Physics
    - Planck 2018 cosmological parameters (TT,TE,EE+lowE+lensing+BAO)

All values are central values. Uncertainties are documented but not
propagated here — the framework's predictions are tested at the percent
level, well above experimental uncertainty for most quantities.
"""

import math


# ═══════════════════════════════════════════════════════════════════
# FRAMEWORK BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════

PI = math.pi
PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.6180339887
LN2 = math.log(2)                      # ≈ 0.6931471806
SQRT5 = math.sqrt(5)                   # ≈ 2.2360679775

# Particle content integers
N_GLUONS = 8                            # SU(3) generators
N_BOSONS = 10                           # 1 photon + 8 gluons + 1 Higgs
N_GENERATIONS = 3                       # Fermion generations
N_COLORS = 3                            # QCD color charges


# ═══════════════════════════════════════════════════════════════════
# MEASURED VALUES (CODATA 2022 / PDG 2024 / Planck 2018)
# ═══════════════════════════════════════════════════════════════════

class Measured:
    """
    Central measured values for comparison with framework predictions.

    Each value includes a comment with source and uncertainty.
    The framework operates at ~0.01–2% accuracy, so experimental
    uncertainties (typically << 0.01%) are negligible for our purposes.
    """

    # --- Lepton mass ratios ---
    mu_e_ratio = 206.7682830           # mμ/me — CODATA 2022 (±0.0000046)
    tau_mu_ratio = 16.8170             # mτ/mμ — PDG 2024 (mτ=1776.86 MeV, mμ=105.6584 MeV)
    tau_e_ratio = 3477.23              # mτ/me — PDG 2024 (mτ=1776.86 MeV, me=0.51100 MeV)

    # --- Baryon mass ratios ---
    proton_e_ratio = 1836.15267363     # mp/me — CODATA 2022 (±0.00000017)
    neutron_e_ratio = 1838.68366173    # mn/me — CODATA 2022 (±0.00000089)

    # --- Coupling constants ---
    alpha_em_inverse = 137.035999177   # 1/α — CODATA 2022 (±0.000000021)
    alpha_s_at_mZ = 0.1180             # αs(mZ) — PDG 2024 (±0.0009)
    alpha_s_over_alpha_em = 0.1180 * 137.036  # αs/αem ≈ 16.170

    # --- Electroweak masses (GeV/c²) ---
    m_Z = 91.1876                      # Z boson mass — PDG 2024 (±0.0021)
    m_W = 80.3692                      # W boson mass — PDG 2024 world avg (±0.0133)
    m_H = 125.25                       # Higgs boson mass — PDG 2024 (±0.17)

    # --- Electroweak mixing ---
    sin2_theta_W = 0.23122             # sin²θW (MS-bar, at mZ) — PDG 2024 (±0.00004)

    # --- Quark mass ratios ---
    # MS-bar masses at μ = 2 GeV (light quarks) and self-consistent scale (heavy)
    mc_ms = 13.60                      # mc/ms — PDG 2024 (mc(mc)=1.270 GeV, ms(2 GeV)=93.4 MeV)
                                       # Ratio ≈ 13.6 at common scale; scheme-dependent
                                       # Some sources cite ~11.8 using different scale conventions
    mt_mb = 41.44                      # mt/mb — PDG 2024 (mt=172.57 GeV, mb=4.183 GeV)
    mt_mc = 135.88                     # mt/mc — PDG 2024 (mt=172.57 GeV, mc=1.270 GeV)

    # --- Mixing angles ---
    sin2_theta13_PMNS = 0.02220        # sin²θ₁₃ (PMNS) — PDG 2024 (±0.00068)
    V_us_CKM = 0.2243                  # |Vus| (CKM) — PDG 2024 (±0.0008)

    # --- Cosmological parameters (Planck 2018 + BAO) ---
    Omega_Lambda = 0.6889              # Dark energy density — Planck 2018 (±0.0056)
    Omega_m = 0.3111                   # Total matter density — Planck 2018 (±0.0056)
    Omega_b = 0.04897                  # Baryon density — Planck 2018 (±0.00030)
    H0_over_100 = 0.6766               # H₀ / 100 km/s/Mpc — Planck 2018 (±0.0042)
    sigma_8 = 0.8102                   # Matter fluctuation amplitude — Planck 2018 (±0.0060)
    n_s = 0.9665                       # Scalar spectral index — Planck 2018 (±0.0038)

    # --- Nuclear binding energies (MeV) ---
    deuteron_binding = 2.224566        # Deuteron binding energy — NUBASE 2020
    alpha_binding = 28.2957            # ⁴He binding energy — NUBASE 2020

    # --- Mathematical constants (for verification) ---
    euler_e = math.e                   # Euler's number e ≈ 2.71828...

    # --- Texture parameters (from Circumpunct framework measurements) ---
    # These are phenomenological values the framework aims to reproduce.
    # Currently placeholders — to be filled with empirical texture analysis.
    tau_texture = 3.694                # Texture SNR threshold (estimated)
    alpha_texture = 1.694              # Texture amplitude (estimated)
