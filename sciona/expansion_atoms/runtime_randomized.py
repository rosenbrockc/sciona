"""Runtime atoms for randomized algorithm expansion rules."""

from __future__ import annotations

import numpy as np


def validate_hash_independence(
    observed_collisions: float,
    expected_collisions: float,
) -> tuple[float, bool]:
    """Compare observed collisions to the independence baseline."""
    observed = float(observed_collisions)
    expected = max(float(expected_collisions), 1e-12)
    collision_ratio = observed / expected
    return collision_ratio, collision_ratio <= 2.0


def analyze_sketch_accuracy(
    true_values: np.ndarray,
    estimated_values: np.ndarray,
) -> tuple[float, bool]:
    """Measure mean relative sketch error."""
    truth = np.asarray(true_values, dtype=np.float64).ravel()
    est = np.asarray(estimated_values, dtype=np.float64).ravel()
    if truth.size == 0 or est.size == 0:
        return 0.0, True
    n = min(truth.size, est.size)
    truth = truth[:n]
    est = est[:n]
    denom = np.maximum(np.abs(truth), 1e-12)
    relative_error = float(np.mean(np.abs(est - truth) / denom))
    return relative_error, relative_error <= 0.1


def monitor_sample_coverage(
    samples: np.ndarray,
    population_size: int,
) -> tuple[float, bool]:
    """Estimate unique sample coverage of the population."""
    arr = np.asarray(samples).ravel()
    pop = max(int(population_size), 1)
    if arr.size == 0:
        return 0.0, False
    coverage = float(np.unique(arr).size) / float(pop)
    return coverage, coverage >= 0.1


def check_concentration_bound(
    empirical_errors: np.ndarray,
    theoretical_bound: float,
) -> tuple[float, bool]:
    """Measure the fraction of samples violating a concentration bound."""
    errs = np.asarray(empirical_errors, dtype=np.float64).ravel()
    bound = max(float(theoretical_bound), 0.0)
    if errs.size == 0:
        return 0.0, True
    violation_rate = float(np.mean(np.abs(errs) > bound))
    return violation_rate, violation_rate <= 0.05
