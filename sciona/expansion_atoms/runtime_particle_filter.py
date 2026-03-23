"""Runtime atoms for Particle Filter expansion rules.

Provides deterministic, pure functions for particle filter
quality diagnostics:

  - Effective sample size monitoring (weight degeneracy detection)
  - Particle diversity analysis (mode collapse detection)
  - Weight variance tracking (resampling necessity)
  - Resampling quality check (particle duplication analysis)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Effective sample size monitoring
# ---------------------------------------------------------------------------


def monitor_effective_sample_size(
    log_weights: np.ndarray,
) -> tuple[float, bool]:
    """Monitor the effective sample size (ESS) of particle weights.

    Low ESS indicates weight degeneracy where few particles
    dominate the distribution.

    Args:
        log_weights: 1-D array of log-weights (unnormalized).

    Returns:
        (ess_fraction, is_healthy) where ess_fraction is ESS / N
        and is_healthy is True if ess_fraction > 0.5.
    """
    lw = np.asarray(log_weights, dtype=np.float64).ravel()
    n = len(lw)
    if n == 0:
        return 0.0, False

    # Log-sum-exp for numerical stability
    max_lw = float(np.max(lw))
    w = np.exp(lw - max_lw)
    w_sum = float(np.sum(w))
    if w_sum == 0:
        return 0.0, False

    normalized = w / w_sum
    ess = 1.0 / float(np.sum(normalized ** 2))
    ess_frac = ess / n
    return ess_frac, ess_frac > 0.5


# ---------------------------------------------------------------------------
# Particle diversity analysis
# ---------------------------------------------------------------------------


def analyze_particle_diversity(
    particles: np.ndarray,
) -> tuple[float, bool]:
    """Analyze diversity of particle positions.

    Low diversity indicates mode collapse where particles cluster
    too tightly around a single mode.

    Args:
        particles: (n_particles, d) array of particle states.

    Returns:
        (mean_pairwise_distance, is_diverse) where is_diverse is True
        if mean distance > 0.01 * max_range.
    """
    p = np.asarray(particles, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(-1, 1)

    n = p.shape[0]
    if n < 2:
        return 0.0, False

    # Sample pairwise distances for efficiency
    rng = np.random.RandomState(42)
    n_samples = min(500, n * (n - 1) // 2)

    total_dist = 0.0
    for _ in range(n_samples):
        i, j = rng.choice(n, 2, replace=False)
        total_dist += float(np.linalg.norm(p[i] - p[j]))

    mean_dist = total_dist / max(n_samples, 1)
    max_range = float(np.max(np.ptp(p, axis=0)))
    threshold = max_range * 0.01 if max_range > 0 else 1e-10
    return mean_dist, mean_dist > threshold


# ---------------------------------------------------------------------------
# Weight variance tracking
# ---------------------------------------------------------------------------


def track_weight_variance(
    log_weights_history: np.ndarray,
) -> tuple[float, bool]:
    """Track variance of particle weights over time.

    Increasing weight variance indicates progressive degeneracy
    that resampling is not addressing.

    Args:
        log_weights_history: (n_steps, n_particles) array of
            log-weights per step.

    Returns:
        (variance_trend, is_stable) where variance_trend is the
        slope of weight variance over time and is_stable is True
        if the trend is non-positive.
    """
    h = np.asarray(log_weights_history, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(1, -1)

    n_steps = h.shape[0]
    if n_steps < 2:
        return 0.0, True

    variances = np.array([float(np.var(h[i])) for i in range(n_steps)])
    # Linear regression slope
    x = np.arange(n_steps, dtype=np.float64)
    x_mean = np.mean(x)
    var_mean = np.mean(variances)
    slope = float(np.sum((x - x_mean) * (variances - var_mean)) / max(np.sum((x - x_mean) ** 2), 1e-30))
    return slope, slope <= 0


# ---------------------------------------------------------------------------
# Resampling quality check
# ---------------------------------------------------------------------------


def check_resampling_quality(
    parent_indices: np.ndarray,
    n_particles: int,
) -> tuple[float, bool]:
    """Check the quality of resampling by analyzing duplication.

    If a few parent particles are duplicated many times, the
    resampling is too aggressive and diversity is lost.

    Args:
        parent_indices: 1-D array of parent particle indices after resampling.
        n_particles: total number of particles.

    Returns:
        (max_duplication_fraction, is_acceptable) where
        max_duplication_fraction is (max_count / n_particles) and
        is_acceptable is True if < 0.1.
    """
    idx = np.asarray(parent_indices, dtype=np.int64).ravel()
    if len(idx) == 0 or n_particles == 0:
        return 0.0, True

    _, counts = np.unique(idx, return_counts=True)
    max_frac = float(np.max(counts)) / n_particles
    return max_frac, max_frac < 0.1
