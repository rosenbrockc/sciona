"""Runtime atoms for Linear Algebra expansion rules.

Provides deterministic, pure functions for matrix decomposition
quality diagnostics:

  - Matrix conditioning analysis (ill-conditioning detection)
  - Decomposition accuracy validation (residual check)
  - Rank deficiency detection (numerical rank estimation)
  - Iterative convergence monitoring (residual norm decay)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Matrix conditioning
# ---------------------------------------------------------------------------


def check_matrix_conditioning(
    A: np.ndarray,
) -> tuple[float, bool]:
    """Analyze the condition number of a matrix.

    A large condition number indicates the matrix is ill-conditioned
    and numerical results may be unreliable.

    Args:
        A: 2D array representing the matrix.

    Returns:
        (condition_number, is_well_conditioned) where
        is_well_conditioned is True if condition_number <= 1e12.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        return 1.0, True

    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf"), False

    if not np.isfinite(cond):
        return float("inf"), False

    return cond, cond <= 1e12


# ---------------------------------------------------------------------------
# Decomposition accuracy
# ---------------------------------------------------------------------------


def validate_decomposition_accuracy(
    A: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[float, bool]:
    """Validate decomposition by checking reconstruction residual.

    Computes ||A - reconstructed|| / ||A|| and checks it is below
    a tolerance threshold.

    Args:
        A: Original matrix.
        reconstructed: Matrix reconstructed from the decomposition factors.

    Returns:
        (relative_error, is_accurate) where is_accurate is True if
        relative_error <= 1e-8.
    """
    A = np.asarray(A, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if A.size == 0 or reconstructed.size == 0:
        return 0.0, True

    if A.shape != reconstructed.shape:
        return float("inf"), False

    norm_A = float(np.linalg.norm(A))
    if norm_A == 0.0:
        norm_diff = float(np.linalg.norm(reconstructed))
        return norm_diff, norm_diff <= 1e-8

    relative_error = float(np.linalg.norm(A - reconstructed)) / norm_A
    return relative_error, relative_error <= 1e-8


# ---------------------------------------------------------------------------
# Rank deficiency detection
# ---------------------------------------------------------------------------


def detect_rank_deficiency(
    singular_values: np.ndarray,
    expected_rank: int,
) -> tuple[int, bool]:
    """Estimate effective rank and compare to expected.

    Counts singular values above 1e-10 * max(sv) as contributing
    to the rank.

    Args:
        singular_values: 1D array of singular values (descending order).
        expected_rank: The expected rank of the matrix.

    Returns:
        (effective_rank, is_full_rank) where is_full_rank is True
        if effective_rank >= expected_rank.
    """
    sv = np.asarray(singular_values, dtype=np.float64).ravel()
    if sv.size == 0:
        return 0, expected_rank <= 0

    max_sv = float(np.max(np.abs(sv)))
    if max_sv == 0.0:
        return 0, expected_rank <= 0

    threshold = 1e-10 * max_sv
    effective_rank = int(np.sum(np.abs(sv) > threshold))
    return effective_rank, effective_rank >= expected_rank


# ---------------------------------------------------------------------------
# Iterative convergence monitoring
# ---------------------------------------------------------------------------


def monitor_iterative_convergence(
    residual_norms: np.ndarray,
) -> tuple[float, bool]:
    """Monitor convergence of an iterative solver via residual norms.

    Computes the geometric mean of consecutive residual ratios.
    A ratio close to 1 means slow convergence; below 0.99 is healthy.

    Args:
        residual_norms: 1D array of residual norms over iterations.

    Returns:
        (convergence_rate, is_converging) where convergence_rate is
        the geometric mean of consecutive ratios and is_converging
        is True if the rate < 0.99.
    """
    norms = np.asarray(residual_norms, dtype=np.float64).ravel()
    if norms.size < 2:
        return 0.0, True

    # Compute consecutive ratios, avoiding division by zero
    prev = norms[:-1]
    curr = norms[1:]
    mask = prev > 0
    if not np.any(mask):
        return 0.0, True

    ratios = curr[mask] / prev[mask]
    # Clip to avoid log issues
    ratios = np.clip(ratios, 1e-30, 1e30)

    geo_mean = float(np.exp(np.mean(np.log(ratios))))
    return geo_mean, geo_mean < 0.99
