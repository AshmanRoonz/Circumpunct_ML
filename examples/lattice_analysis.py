#!/usr/bin/env python3
"""Explore the 64-state lattice geometry."""

from circumpunct_ml import Lattice64

if __name__ == "__main__":
    lattice = Lattice64()

    print(lattice.info())

    print("\nSelection rule (22° threshold):")
    sel = lattice.selection_rule()
    for k, v in sel.items():
        print(f"  {k}: {v}")

    print("\nSpectral localization test:")
    spec = lattice.spectral_localization(n_random=1000)
    print(f"  {spec['interpretation']}")
    print(f"  Significant (z > 3σ): {spec['significant']}")
