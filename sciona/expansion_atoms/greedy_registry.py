"""Registry for greedy primitives and expansion atoms."""

from __future__ import annotations

GREEDY_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "validate_matroid_exchange": (
        "sciona.expansion_atoms.runtime_greedy.validate_matroid_exchange",
        "list[ndarray], int -> tuple[float, bool]",
        "Check the exchange property on observed selection sets.",
    ),
    "detect_criterion_ties": (
        "sciona.expansion_atoms.runtime_greedy.detect_criterion_ties",
        "ndarray, float -> tuple[int, ndarray]",
        "Detect near-ties in greedy criterion ordering.",
    ),
    "estimate_solution_quality": (
        "sciona.expansion_atoms.runtime_greedy.estimate_solution_quality",
        "float, float -> tuple[float, bool]",
        "Compute approximation ratio of greedy solution against a known bound.",
    ),
    "detect_redundant_feasibility": (
        "sciona.expansion_atoms.runtime_greedy.detect_redundant_feasibility",
        "ndarray -> tuple[float, bool]",
        "Detect when feasibility checks are always passing.",
    ),
}
