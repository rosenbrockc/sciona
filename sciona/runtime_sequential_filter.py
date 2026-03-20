"""Runtime atoms for Sequential Filter expansion rules.

Provides deterministic, pure functions for Kalman/particle filter
quality diagnostics and adaptive corrections:

  - Observability analysis (rank test on observability matrix)
  - Innovation whiteness validation (autocorrelation test)
  - Normalized Innovation Squared (NIS) divergence detection
  - Adaptive process-noise estimation (Robbins-Monro)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def check_observability(
    F: np.ndarray,
    H: np.ndarray,
    n_states: int,
) -> tuple[bool, np.ndarray]:
    """Check observability of a linear system (F, H).

    Constructs the observability matrix  O = [H; H*F; H*F^2; ...; H*F^{n-1}]
    and tests whether it has full column rank.

    Returns:
        (is_observable, observability_matrix)
    """
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    H = np.atleast_2d(np.asarray(H, dtype=np.float64))
    n = int(n_states)

    rows = [H]
    HFk = H.copy()
    for _ in range(n - 1):
        HFk = HFk @ F
        rows.append(HFk)

    O = np.vstack(rows)
    rank = int(np.linalg.matrix_rank(O))
    return rank >= n, O


# ---------------------------------------------------------------------------
# Innovation whiteness
# ---------------------------------------------------------------------------


def validate_innovation_whiteness(
    innovations: np.ndarray,
    max_lag: int = 10,
) -> tuple[np.ndarray, bool]:
    """Test whether the innovation sequence is white (uncorrelated).

    Computes normalized autocorrelation at lags 1..max_lag.
    The sequence is considered white if all autocorrelation values
    fall within the approximate 95% confidence bound  ±1.96/√N.

    Returns:
        (autocorrelation_array, is_white)
    """
    innovations = np.asarray(innovations, dtype=np.float64)
    if innovations.ndim > 1:
        # Flatten multi-dimensional innovations (use norm per step)
        innovations = np.array(
            [np.linalg.norm(innovations[i]) for i in range(len(innovations))]
        )

    n = len(innovations)
    if n < max_lag + 2:
        return np.zeros(max_lag), True

    centered = innovations - np.mean(innovations)
    var = float(np.var(centered))
    if var < 1e-15:
        return np.zeros(max_lag), True

    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = float(np.mean(centered[:-lag] * centered[lag:])) / var

    bound = 1.96 / np.sqrt(n)
    is_white = bool(np.all(np.abs(acf) < bound))
    return acf, is_white


# ---------------------------------------------------------------------------
# NIS divergence detection
# ---------------------------------------------------------------------------


def detect_filter_divergence(
    innovations: np.ndarray,
    S_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect Kalman filter divergence via the NIS (Normalized Innovation Squared) test.

    For each step k, computes  NIS_k = y_k^T S_k^{-1} y_k  where y_k is the
    innovation and S_k is the innovation covariance.  Under correct tuning,
    NIS_k ~ chi-squared(m) where m is the measurement dimension.

    Returns:
        (nis_values, divergence_mask)  where divergence_mask[k] is True
        if NIS_k exceeds the 95th percentile of chi-squared(m).
    """
    innovations = np.asarray(innovations, dtype=np.float64)
    S_matrices = np.asarray(S_matrices, dtype=np.float64)

    n = len(innovations)
    if n == 0:
        return np.empty(0), np.empty(0, dtype=bool)

    # Handle scalar innovations
    if innovations.ndim == 1:
        innovations = innovations.reshape(-1, 1)

    m = innovations.shape[1]  # measurement dimension

    # Chi-squared 95th percentile approximation (Wilson-Hilferty)
    # For m degrees of freedom: chi2_95 ≈ m * (1 - 2/(9m) + 1.645*sqrt(2/(9m)))^3
    if m == 1:
        chi2_95 = 3.841
    elif m == 2:
        chi2_95 = 5.991
    elif m == 3:
        chi2_95 = 7.815
    else:
        p = 2.0 / (9.0 * m)
        chi2_95 = m * (1.0 - p + 1.645 * np.sqrt(p)) ** 3

    nis = np.zeros(n)
    for k in range(n):
        y = innovations[k]
        if S_matrices.ndim == 3:
            S = S_matrices[k]
        elif S_matrices.ndim == 2:
            S = S_matrices
        else:
            S = S_matrices.reshape(1, 1)

        try:
            S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
            nis[k] = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            nis[k] = float("inf")

    divergence_mask = nis > chi2_95
    return nis, divergence_mask


# ---------------------------------------------------------------------------
# Adaptive process noise
# ---------------------------------------------------------------------------


def adapt_process_noise(
    innovations: np.ndarray,
    K_matrices: np.ndarray,
    Q_prior: np.ndarray,
    *,
    alpha: float = 0.1,
) -> np.ndarray:
    """Adaptively estimate process noise Q from innovation sequence.

    Uses exponential smoothing (Robbins-Monro style):
        Q_{k+1} = (1 - alpha) * Q_k + alpha * (K_k @ y_k @ y_k^T @ K_k^T)

    This converges to the innovation-based Q estimate under stationarity.

    Returns:
        Q_adapted — the adapted process noise covariance.
    """
    innovations = np.asarray(innovations, dtype=np.float64)
    K_matrices = np.asarray(K_matrices, dtype=np.float64)
    Q = np.asarray(Q_prior, dtype=np.float64).copy()

    if innovations.ndim == 1:
        innovations = innovations.reshape(-1, 1)

    n = len(innovations)
    if n == 0:
        return Q

    for k in range(n):
        y = innovations[k].reshape(-1, 1)
        if K_matrices.ndim == 3:
            K = K_matrices[k]
        else:
            K = K_matrices

        correction = K @ y @ y.T @ K.T
        # Ensure correction has same shape as Q
        q_size = Q.shape[0]
        correction = correction[:q_size, :q_size]
        Q = (1.0 - alpha) * Q + alpha * correction

    # Ensure symmetry
    Q = 0.5 * (Q + Q.T)
    return Q
