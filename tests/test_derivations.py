"""Tests for individual derivation functions."""

import pytest
from circumpunct_ml.derivations import (
    muon_electron_golden,
    muon_electron_fractal,
    tau_muon_ratio,
    tau_electron_ratio,
    proton_electron_ratio,
    neutron_electron_ratio,
    strong_em_coupling_ratio,
    fine_structure_constant,
    fine_structure_golden_angle,
    three_generations,
    fractal_dimension_balance,
    compute_all,
)


class TestLeptonMasses:
    def test_muon_electron_golden(self):
        p = muon_electron_golden()
        assert p.error_pct < 0.01, f"Golden muon formula error: {p.error_pct:.4f}%"

    def test_muon_electron_fractal(self):
        p = muon_electron_fractal()
        assert p.error_pct < 0.5, f"Fractal muon formula error: {p.error_pct:.4f}%"

    def test_tau_muon(self):
        p = tau_muon_ratio()
        assert p.error_pct < 0.1, f"Tau/muon error: {p.error_pct:.4f}%"

    def test_tau_electron(self):
        p = tau_electron_ratio()
        assert p.error_pct < 0.1, f"Tau/electron error: {p.error_pct:.4f}%"


class TestBaryonMasses:
    def test_proton_electron(self):
        p = proton_electron_ratio()
        assert p.error_pct < 0.01, f"Proton/electron error: {p.error_pct:.4f}%"

    def test_neutron_electron(self):
        p = neutron_electron_ratio()
        assert p.error_pct < 0.1, f"Neutron/electron error: {p.error_pct:.4f}%"


class TestCouplings:
    def test_strong_em_ratio(self):
        p = strong_em_coupling_ratio()
        assert p.error_pct < 0.1, f"Strong/EM error: {p.error_pct:.4f}%"

    def test_fine_structure_cubic(self):
        p = fine_structure_constant()
        assert p.error_pct < 0.05, f"Fine structure (cubic) error: {p.error_pct:.4f}%"

    def test_fine_structure_golden(self):
        p = fine_structure_golden_angle()
        assert p.error_pct < 0.5, f"Fine structure (golden) error: {p.error_pct:.4f}%"


class TestStructural:
    def test_three_generations(self):
        p = three_generations()
        assert p.predicted == 3
        assert p.error_pct == 0.0

    def test_fractal_dimension(self):
        p = fractal_dimension_balance()
        assert p.predicted == 1.5
        assert p.error_pct == 0.0


class TestRegistry:
    def test_compute_all_returns_predictions(self):
        preds = compute_all()
        assert len(preds) >= 25
        for p in preds:
            assert hasattr(p, 'name')
            assert hasattr(p, 'predicted')
            assert hasattr(p, 'measured')
            assert hasattr(p, 'error_pct')
