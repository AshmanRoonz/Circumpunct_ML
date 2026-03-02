#!/usr/bin/env python3
"""Run the full Circumpunct prediction suite and print results."""

from circumpunct_ml import run_all_predictions

if __name__ == "__main__":
    results = run_all_predictions()
    results.summary()
