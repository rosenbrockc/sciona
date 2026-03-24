"""Registry for randomized algorithm primitives and expansion atoms."""

from __future__ import annotations

RANDOMIZED_DECLARATIONS = {
    "validate_hash_independence": (
        "sciona.expansion_atoms.runtime_randomized.validate_hash_independence",
        "float, float -> tuple[float, bool]",
        "Compare observed collisions to independent-hash expectations.",
    ),
    "analyze_sketch_accuracy": (
        "sciona.expansion_atoms.runtime_randomized.analyze_sketch_accuracy",
        "ndarray, ndarray -> tuple[float, bool]",
        "Measure mean relative error between sketch estimates and truth.",
    ),
    "monitor_sample_coverage": (
        "sciona.expansion_atoms.runtime_randomized.monitor_sample_coverage",
        "ndarray, int -> tuple[float, bool]",
        "Measure unique sample coverage of a finite population.",
    ),
    "check_concentration_bound": (
        "sciona.expansion_atoms.runtime_randomized.check_concentration_bound",
        "ndarray, float -> tuple[float, bool]",
        "Measure empirical violations of a theoretical concentration bound.",
    ),
}
