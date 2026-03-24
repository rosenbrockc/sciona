"""Registry for clustering primitives and expansion atoms."""

from __future__ import annotations

CLUSTERING_DECLARATIONS = {
    "analyze_cluster_balance": (
        "sciona.expansion_atoms.runtime_clustering.analyze_cluster_balance",
        "ndarray -> tuple[float, bool]",
        "Analyze cluster size balance via max/min ratio.",
    ),
    "monitor_assignment_stability": (
        "sciona.expansion_atoms.runtime_clustering.monitor_assignment_stability",
        "ndarray, ndarray -> tuple[float, bool]",
        "Monitor clustering assignment stability between iterations.",
    ),
    "detect_empty_clusters": (
        "sciona.expansion_atoms.runtime_clustering.detect_empty_clusters",
        "ndarray -> tuple[int, bool]",
        "Detect clusters with zero members.",
    ),
    "validate_separation": (
        "sciona.expansion_atoms.runtime_clustering.validate_separation",
        "ndarray, ndarray -> tuple[float, bool]",
        "Validate cluster separation quality via inter/intra distance ratio.",
    ),
}
