"""Registry for ODE solver primitives and expansion atoms."""

from __future__ import annotations

ODE_SOLVER_DECLARATIONS = {
    "monitor_step_rejection_rate": (
        "sciona.expansion_atoms.runtime_ode_solver.monitor_step_rejection_rate",
        "ndarray -> tuple[float, bool]",
        "Measure the rejection rate in adaptive step control.",
    ),
    "detect_stiffness": (
        "sciona.expansion_atoms.runtime_ode_solver.detect_stiffness",
        "ndarray -> tuple[float, bool]",
        "Estimate stiffness from Jacobian eigenvalue scales.",
    ),
    "check_energy_conservation": (
        "sciona.expansion_atoms.runtime_ode_solver.check_energy_conservation",
        "ndarray -> tuple[float, bool]",
        "Check drift in a conserved energy quantity.",
    ),
    "validate_order_of_accuracy": (
        "sciona.expansion_atoms.runtime_ode_solver.validate_order_of_accuracy",
        "ndarray, ndarray, float -> tuple[float, bool]",
        "Estimate empirical convergence order from error-step data.",
    ),
}
