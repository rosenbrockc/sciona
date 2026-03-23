"""Registry for graph optimization primitives and expansion atoms."""

from __future__ import annotations

GRAPH_OPTIMIZATION_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "detect_negative_weights": (
        "sciona.expansion_atoms.runtime_graph_optimization.detect_negative_weights",
        "ndarray -> tuple[int, float]",
        "Detect negative edge weights that invalidate Dijkstra's algorithm.",
    ),
    "monitor_relaxation_convergence": (
        "sciona.expansion_atoms.runtime_graph_optimization.monitor_relaxation_convergence",
        "ndarray -> tuple[int, bool]",
        "Monitor whether edge relaxation has converged.",
    ),
    "detect_distance_overflow": (
        "sciona.expansion_atoms.runtime_graph_optimization.detect_distance_overflow",
        "ndarray, float -> tuple[int, float]",
        "Detect numeric overflow in distance computations.",
    ),
    "analyze_graph_density": (
        "sciona.expansion_atoms.runtime_graph_optimization.analyze_graph_density",
        "int, int -> tuple[float, str]",
        "Analyze graph density for algorithm selection guidance.",
    ),
}
