#!/usr/bin/env python3
"""Verify D=1.5 for Brownian motion (the framework's anchor point)."""

from circumpunct_ml import verify_brownian_D15

if __name__ == "__main__":
    print("Verifying D = 1.5 for Brownian motion...\n")
    result = verify_brownian_D15(n_samples=20, n_steps=50000)

    print(f"  Mean D: {result['mean_D']:.4f}")
    print(f"  Std D:  {result['std_D']:.4f}")
    print(f"  Expected: {result['expected']}")
    print(f"  Error: {result['error_pct']:.2f}%")
    print(f"  Samples: {result['n_samples']}")
    print(f"\n  Individual estimates: {[f'{d:.3f}' for d in result['individual_D']]}")
