"""Runtime atoms for quadrature expansion rules."""

from __future__ import annotations

import numpy as np


def analyze_integrand_smoothness(
    values: np.ndarray,
    points: np.ndarray,
) -> tuple[float, bool]:
    """Estimate the maximum absolute first derivative magnitude."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    pts = np.asarray(points, dtype=np.float64).ravel()
    if vals.size < 2 or pts.size < 2:
        return 0.0, True
    n = min(vals.size, pts.size)
    vals = vals[:n]
    pts = pts[:n]
    diffs = np.diff(pts)
    mask = diffs != 0.0
    if not np.any(mask):
        return 0.0, True
    deriv = np.diff(vals)[mask] / diffs[mask]
    max_derivative = float(np.max(np.abs(deriv))) if deriv.size else 0.0
    return max_derivative, max_derivative <= 1e6


def detect_singularity(
    values: np.ndarray,
) -> tuple[float, bool]:
    """Detect extreme magnitudes that suggest a singular integrand."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size == 0:
        return 0.0, True
    finite = np.abs(vals[np.isfinite(vals)])
    if finite.size == 0:
        return float("inf"), False
    max_value = float(np.max(finite))
    return max_value, max_value <= 1e10


def monitor_convergence_rate(
    estimates: np.ndarray,
) -> tuple[float, bool]:
    """Estimate the ratio of successive quadrature estimate differences."""
    est = np.asarray(estimates, dtype=np.float64).ravel()
    if est.size < 3:
        return 0.0, True
    diffs = np.abs(np.diff(est))
    prev = diffs[:-1]
    curr = diffs[1:]
    mask = prev > 0.0
    if not np.any(mask):
        return 0.0, True
    ratios = curr[mask] / prev[mask]
    rate = float(np.mean(ratios))
    return rate, rate < 0.5


def check_domain_coverage(
    points: np.ndarray,
    domain: np.ndarray,
) -> tuple[float, bool]:
    """Measure the largest sampling gap relative to domain width."""
    pts = np.sort(np.asarray(points, dtype=np.float64).ravel())
    bounds = np.asarray(domain, dtype=np.float64).ravel()
    if pts.size < 2 or bounds.size < 2:
        return 0.0, True
    lo = float(np.min(bounds))
    hi = float(np.max(bounds))
    width = hi - lo
    if width <= 0.0:
        return 0.0, True
    clipped = np.clip(pts, lo, hi)
    all_points = np.concatenate(([lo], clipped, [hi]))
    max_gap_ratio = float(np.max(np.diff(np.unique(all_points))) / width)
    return max_gap_ratio, max_gap_ratio <= (0.1 + 1e-12)
