"""Runtime atoms for ODE solver expansion rules."""

from __future__ import annotations

import numpy as np


def monitor_step_rejection_rate(
    accepted: np.ndarray,
) -> tuple[float, bool]:
    """Measure the fraction of rejected adaptive steps."""
    flags = np.asarray(accepted).ravel()
    if flags.size == 0:
        return 0.0, True
    accepted_mask = flags.astype(bool)
    rejection_rate = 1.0 - float(np.mean(accepted_mask))
    return rejection_rate, rejection_rate <= 0.5


def detect_stiffness(
    jacobian_eigenvalues: np.ndarray,
) -> tuple[float, bool]:
    """Estimate stiffness from the spread of Jacobian eigenvalue magnitudes."""
    eigs = np.asarray(jacobian_eigenvalues, dtype=np.float64).ravel()
    if eigs.size == 0:
        return 1.0, False
    mags = np.abs(eigs[np.isfinite(eigs)])
    if mags.size == 0:
        return float("inf"), True
    positive = mags[mags > 0.0]
    if positive.size == 0:
        return 1.0, False
    stiffness_ratio = float(np.max(positive) / np.min(positive))
    return stiffness_ratio, stiffness_ratio > 1e6


def check_energy_conservation(
    energy_values: np.ndarray,
) -> tuple[float, bool]:
    """Measure maximum absolute drift from the initial energy."""
    energies = np.asarray(energy_values, dtype=np.float64).ravel()
    if energies.size < 2:
        return 0.0, True
    baseline = float(energies[0])
    drift = np.abs(energies - baseline)
    max_drift = float(np.max(drift))
    return max_drift, max_drift <= 1e-6


def validate_order_of_accuracy(
    errors: np.ndarray,
    step_sizes: np.ndarray,
    expected_order: float,
) -> tuple[float, bool]:
    """Estimate empirical order from log(error) vs log(step size)."""
    err = np.asarray(errors, dtype=np.float64).ravel()
    h = np.asarray(step_sizes, dtype=np.float64).ravel()
    if err.size < 2 or h.size < 2:
        return 0.0, True
    n = min(err.size, h.size)
    err = err[:n]
    h = h[:n]
    mask = np.isfinite(err) & np.isfinite(h) & (err > 0.0) & (h > 0.0)
    if np.sum(mask) < 2:
        return 0.0, True
    log_h = np.log(h[mask])
    log_e = np.log(err[mask])
    denom = float(np.var(log_h))
    if denom == 0.0:
        empirical_order = 0.0
    else:
        empirical_order = float(np.polyfit(log_h, log_e, deg=1)[0])
    threshold = 0.8 * float(expected_order)
    return empirical_order, empirical_order >= threshold
