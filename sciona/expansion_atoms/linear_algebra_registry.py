"""Registry for linear algebra primitives and expansion atoms."""

from __future__ import annotations

LINEAR_ALGEBRA_DECLARATIONS = {
    "check_matrix_conditioning": (
        "sciona.expansion_atoms.runtime_linear_algebra.check_matrix_conditioning",
        "ndarray -> tuple[float, bool]",
        "Analyze the condition number of a matrix.",
    ),
    "validate_decomposition_accuracy": (
        "sciona.expansion_atoms.runtime_linear_algebra.validate_decomposition_accuracy",
        "ndarray, ndarray -> tuple[float, bool]",
        "Validate decomposition by checking reconstruction residual.",
    ),
    "detect_rank_deficiency": (
        "sciona.expansion_atoms.runtime_linear_algebra.detect_rank_deficiency",
        "ndarray, int -> tuple[int, bool]",
        "Estimate effective rank and compare to expected.",
    ),
    "monitor_iterative_convergence": (
        "sciona.expansion_atoms.runtime_linear_algebra.monitor_iterative_convergence",
        "ndarray -> tuple[float, bool]",
        "Monitor convergence of an iterative solver via residual norms.",
    ),
}
