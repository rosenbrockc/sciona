"""Registry for divide-and-conquer primitives and expansion atoms."""

from __future__ import annotations

DIVIDE_AND_CONQUER_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "measure_split_balance": (
        "sciona.expansion_atoms.runtime_divide_and_conquer.measure_split_balance",
        "ndarray, ndarray -> tuple[float, ndarray]",
        "Measure the balance of divide-and-conquer splits.",
    ),
    "check_recursion_depth": (
        "sciona.expansion_atoms.runtime_divide_and_conquer.check_recursion_depth",
        "int, int -> tuple[float, bool]",
        "Check whether recursion depth is excessive relative to input size.",
    ),
    "profile_merge_cost": (
        "sciona.expansion_atoms.runtime_divide_and_conquer.profile_merge_cost",
        "ndarray, ndarray -> tuple[float, ndarray]",
        "Profile the fraction of total time spent in merge operations.",
    ),
    "detect_subproblem_overlap": (
        "sciona.expansion_atoms.runtime_divide_and_conquer.detect_subproblem_overlap",
        "ndarray -> tuple[float, int]",
        "Detect repeated subproblems suggesting DP would be more efficient.",
    ),
}
