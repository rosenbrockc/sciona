"""Runtime atoms for VI/ADVI expansion rules.

Provides deterministic, pure functions for variational inference
quality diagnostics:

  - ELBO convergence monitoring (optimization progress)
  - Gradient variance analysis (noisy gradient detection)
  - Posterior collapse detection (KL vanishing)
  - Step size stability check (optimizer health)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# ELBO convergence monitoring
# ---------------------------------------------------------------------------


def monitor_elbo_convergence(
    elbo_history: np.ndarray,
    window: int = 10,
) -> tuple[float, bool]:
    """Monitor ELBO convergence from optimization history.

    Checks whether the ELBO has plateaued by comparing recent
    improvement to earlier improvement rates.

    Args:
        elbo_history: 1-D array of ELBO values per iteration.
        window: number of recent iterations to average.

    Returns:
        (relative_improvement, has_converged) where
        relative_improvement is the fractional change over the last
        window and has_converged is True if < 1e-4.
    """
    h = np.asarray(elbo_history, dtype=np.float64).ravel()
    if len(h) < 2:
        return 0.0, False

    w = min(window, len(h) // 2)
    if w < 1:
        w = 1

    recent = float(np.mean(h[-w:]))
    earlier = float(np.mean(h[-2 * w:-w])) if len(h) >= 2 * w else float(h[0])

    denom = max(abs(earlier), 1e-30)
    rel_improvement = abs(recent - earlier) / denom
    return rel_improvement, rel_improvement < 1e-4


# ---------------------------------------------------------------------------
# Gradient variance analysis
# ---------------------------------------------------------------------------


def analyze_gradient_variance(
    gradient_samples: np.ndarray,
) -> tuple[float, bool]:
    """Analyze variance of stochastic gradient estimates.

    High gradient variance indicates the Monte Carlo estimate of
    the ELBO gradient is noisy, slowing convergence.

    Args:
        gradient_samples: (n_samples, d) array of gradient samples.

    Returns:
        (mean_cv, is_low_variance) where mean_cv is the average
        coefficient of variation across dimensions and is_low_variance
        is True if mean_cv < 1.0.
    """
    g = np.asarray(gradient_samples, dtype=np.float64)
    if g.ndim == 1:
        g = g.reshape(-1, 1)

    if g.shape[0] < 2:
        return 0.0, True

    means = np.mean(g, axis=0)
    stds = np.std(g, axis=0)

    # CV per dimension, avoiding division by zero
    safe_means = np.where(np.abs(means) > 1e-30, means, 1.0)
    cvs = stds / np.abs(safe_means)
    mean_cv = float(np.mean(cvs))
    return mean_cv, mean_cv < 1.0


# ---------------------------------------------------------------------------
# Posterior collapse detection
# ---------------------------------------------------------------------------


def detect_posterior_collapse(
    kl_per_dimension: np.ndarray,
    threshold: float = 0.01,
) -> tuple[int, float]:
    """Detect posterior collapse (KL vanishing) per latent dimension.

    When KL(q||p) for a dimension is near zero, the posterior has
    collapsed to the prior and that dimension carries no information.

    Args:
        kl_per_dimension: 1-D array of per-dimension KL divergences.
        threshold: KL below this is considered collapsed.

    Returns:
        (n_collapsed, collapse_fraction) where n_collapsed is the
        number of dimensions with KL < threshold.
    """
    kl = np.asarray(kl_per_dimension, dtype=np.float64).ravel()
    if len(kl) == 0:
        return 0, 0.0

    n_collapsed = int(np.sum(kl < threshold))
    fraction = n_collapsed / len(kl)
    return n_collapsed, fraction


# ---------------------------------------------------------------------------
# Step size stability check
# ---------------------------------------------------------------------------


def check_step_size_stability(
    step_sizes: np.ndarray,
) -> tuple[float, bool]:
    """Check whether optimizer step sizes are stable.

    Wildly varying step sizes indicate optimizer instability,
    often caused by ill-conditioned ELBO landscape.

    Args:
        step_sizes: 1-D array of per-iteration step sizes.

    Returns:
        (coefficient_of_variation, is_stable) where CV is
        std / mean and is_stable is True if CV < 0.5.
    """
    s = np.asarray(step_sizes, dtype=np.float64).ravel()
    if len(s) < 2:
        return 0.0, True

    mean_s = float(np.mean(s))
    if mean_s == 0:
        return 0.0, True

    cv = float(np.std(s)) / mean_s
    return cv, cv < 0.5
