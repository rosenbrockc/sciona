"""Registry for information theory primitives and expansion atoms."""

from __future__ import annotations

INFORMATION_THEORY_DECLARATIONS = {
    "check_distribution_support": (
        "sciona.expansion_atoms.runtime_information_theory.check_distribution_support",
        "ndarray -> tuple[float, bool]",
        "Measure the fraction of bins with zero or negative support.",
    ),
    "analyze_sample_sufficiency": (
        "sciona.expansion_atoms.runtime_information_theory.analyze_sample_sufficiency",
        "int, int -> tuple[float, bool]",
        "Estimate the average number of samples per support element.",
    ),
    "detect_numerical_underflow": (
        "sciona.expansion_atoms.runtime_information_theory.detect_numerical_underflow",
        "ndarray -> tuple[float, bool]",
        "Estimate the fraction of log-probability entries that are numerically unstable.",
    ),
    "validate_information_inequality": (
        "sciona.expansion_atoms.runtime_information_theory.validate_information_inequality",
        "ndarray, ndarray -> tuple[float, bool]",
        "Measure the maximum violation of an information inequality.",
    ),
}
