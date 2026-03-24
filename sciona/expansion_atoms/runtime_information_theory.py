"""Runtime atoms for Information Theory expansion rules."""

from __future__ import annotations

import numpy as np


def check_distribution_support(
    probabilities: np.ndarray,
) -> tuple[float, bool]:
    """Measure the fraction of bins with zero or negative support."""
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0, True
    zero_mass_fraction = float(np.mean(p <= 0.0))
    return zero_mass_fraction, zero_mass_fraction == 0.0


def analyze_sample_sufficiency(
    sample_count: int,
    support_size: int,
) -> tuple[float, bool]:
    """Estimate average samples per support element."""
    samples = max(int(sample_count), 0)
    support = max(int(support_size), 1)
    samples_per_symbol = samples / support
    return samples_per_symbol, samples_per_symbol >= 5.0


def detect_numerical_underflow(
    log_probabilities: np.ndarray,
) -> tuple[float, bool]:
    """Estimate the fraction of log-probability entries that are numerically unstable."""
    lp = np.asarray(log_probabilities, dtype=np.float64).ravel()
    if lp.size == 0:
        return 0.0, True
    underflow_mask = ~np.isfinite(lp) | (lp < -700.0)
    underflow_fraction = float(np.mean(underflow_mask))
    return underflow_fraction, underflow_fraction <= 0.05


def validate_information_inequality(
    lhs_values: np.ndarray,
    rhs_values: np.ndarray,
) -> tuple[float, bool]:
    """Measure the maximum violation of an inequality expected to satisfy lhs <= rhs."""
    lhs = np.asarray(lhs_values, dtype=np.float64).ravel()
    rhs = np.asarray(rhs_values, dtype=np.float64).ravel()
    if lhs.size == 0 or rhs.size == 0:
        return 0.0, True
    n = min(lhs.size, rhs.size)
    violations = lhs[:n] - rhs[:n]
    max_violation = float(max(np.max(violations), 0.0))
    return max_violation, max_violation <= 1e-9
