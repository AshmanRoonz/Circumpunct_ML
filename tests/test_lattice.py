"""Tests for the 64-state lattice geometry."""

import pytest
import numpy as np
from circumpunct_ml import Lattice64


@pytest.fixture
def lattice():
    return Lattice64()


def test_state_count(lattice):
    """Q₆ should have exactly 64 states."""
    assert lattice.n_states == 64
    assert lattice.states.shape == (64, 6)


def test_adjacency_symmetric(lattice):
    """Adjacency matrix should be symmetric."""
    A = lattice.adjacency
    assert np.array_equal(A, A.T)


def test_adjacency_regular(lattice):
    """Each vertex should have exactly 6 neighbors (6-regular graph)."""
    A = lattice.adjacency
    degrees = A.sum(axis=1)
    assert np.all(degrees == 6)


def test_max_eigenvalue(lattice):
    """Maximum eigenvalue of Q₆ should be 6."""
    assert abs(lattice.max_eigenvalue - 6.0) < 1e-10


def test_eigenvalue_count(lattice):
    """Q₆ should have 7 distinct eigenvalues: -6, -4, -2, 0, 2, 4, 6."""
    evals = np.round(lattice.eigenvalues, 6)
    distinct = sorted(set(evals))
    expected = [-6, -4, -2, 0, 2, 4, 6]
    assert len(distinct) == 7
    for d, e in zip(distinct, expected):
        assert abs(d - e) < 1e-6, f"Expected {e}, got {d}"


def test_hamming_distance(lattice):
    """Hamming distance between all-0 and all-1 should be 6."""
    assert lattice.hamming_distance(0, 63) == 6


def test_sm_bijection(lattice):
    """SM bijection should map all 64 states."""
    particles = lattice.sm_bijection()
    assert len(particles) == 64


def test_selection_rule(lattice):
    """Selection rule should produce physical + virtual = 64."""
    result = lattice.selection_rule()
    assert result["physical"] + result["virtual"] == 64
    assert result["total"] == 64


def test_spectral_localization(lattice):
    """Spectral localization test should complete and return results."""
    result = lattice.spectral_localization(n_random=100)
    assert "z_score" in result
    assert "observed_smax" in result
    assert result["n_subcubes"] > 0
