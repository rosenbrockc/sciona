"""Registry for continuous optimization primitives and expansion atoms."""

from __future__ import annotations

OPTIMIZATION_DECLARATIONS = {
    "detect_vanishing_gradient": (
        "sciona.expansion_atoms.runtime_optimization.detect_vanishing_gradient",
        "ndarray -> tuple[float, bool]",
        "Detect vanishing gradients by checking minimum gradient norm.",
    ),
    "analyze_loss_landscape": (
        "sciona.expansion_atoms.runtime_optimization.analyze_loss_landscape",
        "ndarray -> tuple[float, bool]",
        "Analyze local curvature via Hessian eigenvalue spectrum.",
    ),
    "check_constraint_violation": (
        "sciona.expansion_atoms.runtime_optimization.check_constraint_violation",
        "ndarray, ndarray -> tuple[float, bool]",
        "Check maximum constraint violation for constrained optimization.",
    ),
    "monitor_convergence_rate": (
        "sciona.expansion_atoms.runtime_optimization.monitor_convergence_rate",
        "ndarray -> tuple[float, bool]",
        "Estimate empirical convergence order from objective history.",
    ),
}
