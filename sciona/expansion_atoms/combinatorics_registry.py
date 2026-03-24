"""Registry for combinatorial optimization primitives and expansion atoms."""

from __future__ import annotations

COMBINATORICS_DECLARATIONS = {
    "analyze_branching_factor": (
        "sciona.expansion_atoms.runtime_combinatorics.analyze_branching_factor",
        "ndarray -> tuple[float, bool]",
        "Analyze the effective branching factor of a search tree.",
    ),
    "monitor_bound_tightness": (
        "sciona.expansion_atoms.runtime_combinatorics.monitor_bound_tightness",
        "ndarray, ndarray -> tuple[float, bool]",
        "Monitor the gap between upper and lower bounds over time.",
    ),
    "detect_symmetry": (
        "sciona.expansion_atoms.runtime_combinatorics.detect_symmetry",
        "ndarray, int -> tuple[float, bool]",
        "Detect symmetry in the search space.",
    ),
    "check_pruning_effectiveness": (
        "sciona.expansion_atoms.runtime_combinatorics.check_pruning_effectiveness",
        "int, int -> tuple[float, bool]",
        "Check the effectiveness of pruning in branch-and-bound.",
    ),
}
