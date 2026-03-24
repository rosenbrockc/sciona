"""Runtime atoms for Continuous Optimization expansion rules.

Provides deterministic, pure functions for optimization pipeline
quality diagnostics:

  - Vanishing gradient detection (gradient norm collapse)
  - Loss landscape analysis (Hessian spectral condition)
  - Constraint violation checking (feasibility gap)
  - Convergence rate monitoring (empirical convergence order)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Vanishing gradient detection
# ---------------------------------------------------------------------------


def detect_vanishing_gradient(
    gradients: np.ndarray,
) -> tuple[float, bool]:
    """Detect vanishing gradients by checking minimum gradient norm.

    When gradient norms collapse to near-zero, optimization stalls.

    Args:
        gradients: 2D array (n_steps, n_params) of gradient vectors,
                   or 1D array for a single gradient.

    Returns:
        (min_norm, is_vanishing) where is_vanishing is True if
        min_norm < 1e-15.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if g.size == 0:
        return 0.0, False

    if g.ndim == 1:
        g = g.reshape(1, -1)

    norms = np.linalg.norm(g, axis=1)
    min_norm = float(np.min(norms))
    return min_norm, min_norm < 1e-15


# ---------------------------------------------------------------------------
# Loss landscape analysis
# ---------------------------------------------------------------------------


def analyze_loss_landscape(
    hessian_eigenvalues: np.ndarray,
) -> tuple[float, bool]:
    """Analyze local curvature via Hessian eigenvalue spectrum.

    A very large condition number (max/min eigenvalue ratio) indicates
    an ill-conditioned landscape that is hard to optimize.

    Args:
        hessian_eigenvalues: 1D array of Hessian eigenvalues.

    Returns:
        (condition_number, is_ill_conditioned) where is_ill_conditioned
        is True if condition_number > 1e10.
    """
    eigs = np.asarray(hessian_eigenvalues, dtype=np.float64).ravel()
    if eigs.size == 0:
        return 1.0, False

    abs_eigs = np.abs(eigs)
    max_eig = float(np.max(abs_eigs))
    min_eig = float(np.min(abs_eigs))

    if min_eig == 0.0:
        return float("inf"), True

    cond = max_eig / min_eig
    return cond, cond > 1e10


# ---------------------------------------------------------------------------
# Constraint violation checking
# ---------------------------------------------------------------------------


def check_constraint_violation(
    values: np.ndarray,
    bounds: np.ndarray,
) -> tuple[float, bool]:
    """Check maximum constraint violation for constrained optimization.

    Computes the maximum gap between parameter values and their bounds.
    Each row of bounds is [lower, upper].

    Args:
        values: 1D array of parameter values.
        bounds: (n, 2) array of [lower, upper] bounds per parameter.

    Returns:
        (max_violation, is_feasible) where is_feasible is True if
        max_violation <= 0.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    b = np.asarray(bounds, dtype=np.float64)

    if v.size == 0 or b.size == 0:
        return 0.0, True

    if b.ndim == 1:
        b = b.reshape(-1, 2)

    n = min(len(v), len(b))
    lower_violation = np.maximum(0, b[:n, 0] - v[:n])
    upper_violation = np.maximum(0, v[:n] - b[:n, 1])
    max_viol = float(np.max(np.concatenate([lower_violation, upper_violation])))
    return max_viol, max_viol <= 0.0


# ---------------------------------------------------------------------------
# Convergence rate monitoring
# ---------------------------------------------------------------------------


def monitor_convergence_rate(
    objective_history: np.ndarray,
) -> tuple[float, bool]:
    """Estimate empirical convergence order from objective history.

    Uses log-ratio of consecutive differences to estimate the
    convergence order. Orders below 0.5 indicate very slow convergence.

    Args:
        objective_history: 1D array of objective values over iterations.

    Returns:
        (convergence_order, is_converging) where is_converging is True
        if convergence_order >= 0.5.
    """
    h = np.asarray(objective_history, dtype=np.float64).ravel()
    if h.size < 3:
        return 1.0, True

    diffs = np.abs(np.diff(h))
    # Need at least 2 consecutive nonzero diffs
    mask = (diffs[:-1] > 0) & (diffs[1:] > 0)
    if not np.any(mask):
        return 1.0, True

    ratios = diffs[1:][mask] / diffs[:-1][mask]
    ratios = np.clip(ratios, 1e-30, 1e30)
    # Convergence order is estimated from the log-ratio
    log_ratios = np.log(ratios)
    order = float(np.mean(log_ratios))
    # Negative log-ratio means convergence; transform to order measure
    # For geometric convergence, ratio < 1 → log < 0 → order > 0
    convergence_order = float(-np.mean(log_ratios)) if np.mean(log_ratios) < 0 else 0.0
    return convergence_order, convergence_order >= 0.5
