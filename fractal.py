"""
Fractal dimension estimation tools.

The framework predicts D = 1 + β = 1.5 at balance (β = 0.5).
This module provides tools to estimate fractal dimension from data
using standard methods: box-counting, power spectrum, and Hurst exponent.

These tools enable empirical falsification: measure D for a system,
compare to the framework's prediction for that system's β value.
"""

import numpy as np
from typing import Optional


def box_counting_dimension(
    data: np.ndarray,
    min_box_size: int = 2,
    max_box_size: Optional[int] = None,
    n_sizes: int = 20,
) -> dict:
    """
    Estimate fractal dimension via box-counting method.

    Parameters
    ----------
    data : np.ndarray
        Binary 2D array (1 = occupied, 0 = empty) or
        1D time series (will be converted to binary trace).
    min_box_size : int
        Smallest box size in pixels.
    max_box_size : int or None
        Largest box size. Default: min(data.shape) // 2.
    n_sizes : int
        Number of box sizes to test.

    Returns
    -------
    dict with keys:
        dimension : float — estimated fractal dimension
        sizes : np.ndarray — box sizes used
        counts : np.ndarray — box counts at each size
        r_squared : float — quality of log-log fit
    """
    if data.ndim == 1:
        # Convert 1D time series to 2D binary trace
        data = _timeseries_to_binary_image(data)

    if max_box_size is None:
        max_box_size = min(data.shape) // 2

    sizes = np.unique(np.logspace(
        np.log10(min_box_size),
        np.log10(max_box_size),
        n_sizes,
    ).astype(int))

    counts = []
    for size in sizes:
        n = 0
        for i in range(0, data.shape[0], size):
            for j in range(0, data.shape[1], size):
                box = data[i:i+size, j:j+size]
                if np.any(box):
                    n += 1
        counts.append(n)

    counts = np.array(counts, dtype=float)
    sizes = sizes.astype(float)

    # Log-log linear fit: log(N) = -D * log(ε) + c
    valid = counts > 0
    log_sizes = np.log(sizes[valid])
    log_counts = np.log(counts[valid])

    coeffs = np.polyfit(log_sizes, log_counts, 1)
    dimension = -coeffs[0]

    # R² quality
    fit_values = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_counts - fit_values) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "dimension": dimension,
        "sizes": sizes,
        "counts": counts,
        "r_squared": r_squared,
    }


def power_spectrum_dimension(
    timeseries: np.ndarray,
    fs: float = 1.0,
) -> dict:
    """
    Estimate fractal dimension from power spectrum scaling.

    For a fractal signal: P(f) ∝ f^(-β_spectral)
    Fractal dimension D = (5 - β_spectral) / 2  (for 1D trace)

    Parameters
    ----------
    timeseries : np.ndarray
        1D time series.
    fs : float
        Sampling frequency.

    Returns
    -------
    dict with dimension, spectral_exponent, frequencies, power.
    """
    n = len(timeseries)
    fft = np.fft.rfft(timeseries - np.mean(timeseries))
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1/fs)

    # Fit in log-log space (exclude DC component)
    valid = freqs > 0
    log_f = np.log(freqs[valid])
    log_p = np.log(power[valid] + 1e-30)

    coeffs = np.polyfit(log_f, log_p, 1)
    beta_spectral = -coeffs[0]

    # D = (5 - β) / 2 for 1D → 2D embedding
    dimension = (5 - beta_spectral) / 2

    return {
        "dimension": dimension,
        "spectral_exponent": beta_spectral,
        "frequencies": freqs,
        "power": power,
    }


def hurst_exponent(
    timeseries: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
    n_windows: int = 20,
) -> dict:
    """
    Estimate Hurst exponent H via rescaled range (R/S) analysis.

    Fractal dimension D = 2 - H (for 1D trace).
    At balance: H = 0.5 (Brownian motion), D = 1.5.

    Parameters
    ----------
    timeseries : np.ndarray
        1D time series.

    Returns
    -------
    dict with hurst_H, dimension, and diagnostics.
    """
    n = len(timeseries)
    if max_window is None:
        max_window = n // 4

    windows = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        n_windows,
    ).astype(int))

    rs_values = []
    for w in windows:
        rs_list = []
        for start in range(0, n - w, w):
            segment = timeseries[start:start + w]
            mean = np.mean(segment)
            deviations = segment - mean
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)

    rs_values = np.array(rs_values)
    valid = ~np.isnan(rs_values) & (rs_values > 0)

    log_w = np.log(windows[valid].astype(float))
    log_rs = np.log(rs_values[valid])

    coeffs = np.polyfit(log_w, log_rs, 1)
    H = coeffs[0]
    D = 2 - H

    return {
        "hurst_H": H,
        "dimension": D,
        "windows": windows,
        "rs_values": rs_values,
        "interpretation": (
            f"H = {H:.3f} → D = {D:.3f} "
            f"({'balanced' if abs(D - 1.5) < 0.1 else 'imbalanced'})"
        ),
    }


def _timeseries_to_binary_image(
    ts: np.ndarray,
    resolution: int = 256,
) -> np.ndarray:
    """Convert 1D time series to binary image of its trace."""
    n = len(ts)
    ts_norm = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-30)

    img = np.zeros((resolution, resolution), dtype=np.int8)
    for i in range(n - 1):
        x0 = int(i / n * resolution)
        y0 = int(ts_norm[i] * (resolution - 1))
        x1 = int((i + 1) / n * resolution)
        y1 = int(ts_norm[i + 1] * (resolution - 1))
        # Simple line drawing
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            if 0 <= x < resolution and 0 <= y < resolution:
                img[y, x] = 1

    return img


def verify_brownian_D15(n_samples: int = 10, n_steps: int = 10000) -> dict:
    """
    Verify that Brownian motion has D = 1.5 (exact theorem).

    This is the framework's anchor point: at β = 0.5 (balanced random walk),
    the fractal dimension is exactly 1.5 by Mandelbrot's theorem.

    Generates multiple Brownian paths and estimates D for each.
    """
    rng = np.random.default_rng(42)
    dimensions = []

    for _ in range(n_samples):
        steps = rng.standard_normal(n_steps)
        path = np.cumsum(steps)
        result = hurst_exponent(path)
        dimensions.append(result["dimension"])

    dimensions = np.array(dimensions)
    return {
        "mean_D": float(np.mean(dimensions)),
        "std_D": float(np.std(dimensions)),
        "expected": 1.5,
        "error_pct": abs(np.mean(dimensions) - 1.5) / 1.5 * 100,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "individual_D": dimensions.tolist(),
    }
