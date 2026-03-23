"""Runtime atoms for Kalman Filter expansion rules.

Provides deterministic, pure functions for Kalman filter
quality diagnostics:

  - Innovation consistency check (normalized innovation squared)
  - Covariance positive-definiteness validation
  - Kalman gain magnitude analysis (filter divergence detection)
  - State estimate smoothness check (outlier detection)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Innovation consistency check
# ---------------------------------------------------------------------------


def check_innovation_consistency(
    innovations: np.ndarray,
    innovation_covariance: np.ndarray,
) -> tuple[float, bool]:
    """Check whether innovations are consistent with their predicted covariance.

    The normalized innovation squared (NIS) should follow a chi-squared
    distribution.  Values consistently outside expected bounds indicate
    model mismatch.

    Args:
        innovations: (n_steps, d) array of innovation vectors.
        innovation_covariance: (d, d) predicted innovation covariance.

    Returns:
        (mean_nis, is_consistent) where mean_nis is the average NIS
        and is_consistent is True if mean_nis is within [0.5*d, 2*d].
    """
    y = np.asarray(innovations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    S = np.asarray(innovation_covariance, dtype=np.float64)
    n_steps, d = y.shape

    if n_steps == 0 or d == 0:
        return 0.0, True

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("inf"), False

    nis_values = np.array([float(y[i] @ S_inv @ y[i]) for i in range(n_steps)])
    mean_nis = float(np.mean(nis_values))
    return mean_nis, 0.5 * d <= mean_nis <= 2.0 * d


# ---------------------------------------------------------------------------
# Covariance positive-definiteness
# ---------------------------------------------------------------------------


def validate_covariance_pd(
    covariance: np.ndarray,
) -> tuple[float, bool]:
    """Validate that a covariance matrix is positive definite.

    Numerical errors can cause the Kalman filter covariance to lose
    positive-definiteness, leading to divergence.

    Args:
        covariance: (d, d) covariance matrix.

    Returns:
        (min_eigenvalue, is_pd) where is_pd is True if all eigenvalues
        are positive.
    """
    P = np.asarray(covariance, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return -float("inf"), False

    if P.shape[0] == 0:
        return 0.0, True

    eigs = np.linalg.eigvalsh(P)
    min_eig = float(np.min(eigs))
    return min_eig, min_eig > 0


# ---------------------------------------------------------------------------
# Kalman gain magnitude analysis
# ---------------------------------------------------------------------------


def analyze_kalman_gain_magnitude(
    kalman_gains: np.ndarray,
) -> tuple[float, bool]:
    """Analyze the magnitude of Kalman gains over time.

    Excessively large gains indicate the filter trusts measurements
    far more than the model, which can cause instability.

    Args:
        kalman_gains: (n_steps, d_state, d_obs) or flattened array of
            Kalman gain matrices over time.

    Returns:
        (max_gain_norm, is_bounded) where max_gain_norm is the
        maximum Frobenius norm and is_bounded is True if < 100.
    """
    K = np.asarray(kalman_gains, dtype=np.float64)
    if K.size == 0:
        return 0.0, True

    if K.ndim == 3:
        norms = np.array([float(np.linalg.norm(K[i], "fro")) for i in range(K.shape[0])])
    elif K.ndim == 2:
        norms = np.array([float(np.linalg.norm(K[i])) for i in range(K.shape[0])])
    else:
        norms = np.array([float(np.abs(K).max())])

    max_norm = float(np.max(norms))
    return max_norm, max_norm < 100.0


# ---------------------------------------------------------------------------
# State estimate smoothness
# ---------------------------------------------------------------------------


def check_state_smoothness(
    state_estimates: np.ndarray,
    max_jump_ratio: float = 5.0,
) -> tuple[int, float]:
    """Check for sudden jumps in state estimates.

    Large jumps indicate outlier measurements or model mismatch
    that cause the filter to over-correct.

    Args:
        state_estimates: (n_steps, d) array of state estimates.
        max_jump_ratio: threshold for jump / median_step_size.

    Returns:
        (n_jumps, jump_fraction) where n_jumps is the number of
        time steps with anomalous jumps.
    """
    x = np.asarray(state_estimates, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n = x.shape[0]
    if n < 3:
        return 0, 0.0

    diffs = np.linalg.norm(np.diff(x, axis=0), axis=1)
    median_diff = float(np.median(diffs))
    if median_diff == 0:
        return 0, 0.0

    n_jumps = int(np.sum(diffs > max_jump_ratio * median_diff))
    return n_jumps, n_jumps / (n - 1)
