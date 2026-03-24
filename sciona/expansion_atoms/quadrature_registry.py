"""Registry for quadrature primitives and expansion atoms."""

from __future__ import annotations

QUADRATURE_DECLARATIONS = {
    "analyze_integrand_smoothness": (
        "sciona.expansion_atoms.runtime_quadrature.analyze_integrand_smoothness",
        "ndarray, ndarray -> tuple[float, bool]",
        "Estimate first-derivative scale of the sampled integrand.",
    ),
    "detect_singularity": (
        "sciona.expansion_atoms.runtime_quadrature.detect_singularity",
        "ndarray -> tuple[float, bool]",
        "Detect extreme integrand magnitudes suggestive of singularities.",
    ),
    "monitor_convergence_rate": (
        "sciona.expansion_atoms.runtime_quadrature.monitor_convergence_rate",
        "ndarray -> tuple[float, bool]",
        "Track refinement convergence of quadrature estimates.",
    ),
    "check_domain_coverage": (
        "sciona.expansion_atoms.runtime_quadrature.check_domain_coverage",
        "ndarray, ndarray -> tuple[float, bool]",
        "Measure largest uncovered gap across the integration domain.",
    ),
}
