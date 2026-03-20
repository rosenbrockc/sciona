"""Registry for MCMC/HMC primitives and expansion atoms."""

from __future__ import annotations

MCMC_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "detect_divergent_transitions": (
        "sciona.runtime_mcmc.detect_divergent_transitions",
        "np.ndarray, np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Detect divergent transitions via energy conservation violation.",
    ),
    "compute_dual_averaging_step_size": (
        "sciona.runtime_mcmc.compute_dual_averaging_step_size",
        "np.ndarray, float, float, float, float, float -> float",
        "Adapt HMC step size via Nesterov dual averaging.",
    ),
    "estimate_mass_matrix": (
        "sciona.runtime_mcmc.estimate_mass_matrix",
        "np.ndarray, bool -> np.ndarray",
        "Estimate mass matrix M from warmup samples.",
    ),
    "compute_convergence_diagnostics": (
        "sciona.runtime_mcmc.compute_convergence_diagnostics",
        "np.ndarray -> tuple[np.ndarray, np.ndarray]",
        "Compute split R-hat and bulk ESS across chains.",
    ),
}
