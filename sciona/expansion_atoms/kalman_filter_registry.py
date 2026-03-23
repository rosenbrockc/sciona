"""Registry for Kalman filter primitives and expansion atoms."""

from __future__ import annotations

KALMAN_FILTER_DECLARATIONS = {
    "check_innovation_consistency": (
        "sciona.expansion_atoms.runtime_kalman_filter.check_innovation_consistency",
        "ndarray, ndarray -> tuple[float, bool]",
        "Check whether innovations are consistent with their predicted covariance.",
    ),
    "validate_covariance_pd": (
        "sciona.expansion_atoms.runtime_kalman_filter.validate_covariance_pd",
        "ndarray -> tuple[float, bool]",
        "Validate that a covariance matrix is positive definite.",
    ),
    "analyze_kalman_gain_magnitude": (
        "sciona.expansion_atoms.runtime_kalman_filter.analyze_kalman_gain_magnitude",
        "ndarray -> tuple[float, bool]",
        "Analyze the magnitude of Kalman gains over time.",
    ),
    "check_state_smoothness": (
        "sciona.expansion_atoms.runtime_kalman_filter.check_state_smoothness",
        "ndarray, float -> tuple[int, float]",
        "Check for sudden jumps in state estimates.",
    ),
}
