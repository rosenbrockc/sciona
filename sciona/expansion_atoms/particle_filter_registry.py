"""Registry for particle filter primitives and expansion atoms."""

from __future__ import annotations

PARTICLE_FILTER_DECLARATIONS = {
    "monitor_effective_sample_size": (
        "sciona.expansion_atoms.runtime_particle_filter.monitor_effective_sample_size",
        "ndarray -> tuple[float, bool]",
        "Monitor the effective sample size (ESS) of particle weights.",
    ),
    "analyze_particle_diversity": (
        "sciona.expansion_atoms.runtime_particle_filter.analyze_particle_diversity",
        "ndarray -> tuple[float, bool]",
        "Analyze diversity of particle positions.",
    ),
    "track_weight_variance": (
        "sciona.expansion_atoms.runtime_particle_filter.track_weight_variance",
        "ndarray -> tuple[float, bool]",
        "Track variance of particle weights over time.",
    ),
    "check_resampling_quality": (
        "sciona.expansion_atoms.runtime_particle_filter.check_resampling_quality",
        "ndarray, int -> tuple[float, bool]",
        "Check the quality of resampling by analyzing duplication.",
    ),
}
