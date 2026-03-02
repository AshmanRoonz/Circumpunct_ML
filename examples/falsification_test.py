#!/usr/bin/env python3
"""Run falsification tests — both standard and strict bounds."""

from circumpunct_ml import FalsificationSuite

if __name__ == "__main__":
    suite = FalsificationSuite()

    print("=== Standard Bounds ===")
    report = suite.run()
    report.summary()

    print("\n=== Strict Bounds ===")
    strict_report = suite.run_strict()
    strict_report.summary()
