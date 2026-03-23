"""Registry for VI/ADVI primitives and expansion atoms."""

from __future__ import annotations

VI_ADVI_DECLARATIONS = {
    "monitor_elbo_convergence": (
        "sciona.expansion_atoms.runtime_vi_advi.monitor_elbo_convergence",
        "ndarray, int -> tuple[float, bool]",
        "Monitor ELBO convergence from optimization history.",
    ),
    "analyze_gradient_variance": (
        "sciona.expansion_atoms.runtime_vi_advi.analyze_gradient_variance",
        "ndarray -> tuple[float, bool]",
        "Analyze variance of stochastic gradient estimates.",
    ),
    "detect_posterior_collapse": (
        "sciona.expansion_atoms.runtime_vi_advi.detect_posterior_collapse",
        "ndarray, float -> tuple[int, float]",
        "Detect posterior collapse (KL vanishing) per latent dimension.",
    ),
    "check_step_size_stability": (
        "sciona.expansion_atoms.runtime_vi_advi.check_step_size_stability",
        "ndarray -> tuple[float, bool]",
        "Check whether optimizer step sizes are stable.",
    ),
}
