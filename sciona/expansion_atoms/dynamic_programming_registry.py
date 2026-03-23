"""Registry for dynamic programming primitives and expansion atoms."""

from __future__ import annotations

DYNAMIC_PROGRAMMING_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "detect_table_sparsity": (
        "sciona.expansion_atoms.runtime_dynamic_programming.detect_table_sparsity",
        "np.ndarray, Optional[np.ndarray] -> tuple[float, np.ndarray]",
        "Compute fraction of DP table cells that are actually filled/used.",
    ),
    "prune_infeasible_states": (
        "sciona.expansion_atoms.runtime_dynamic_programming.prune_infeasible_states",
        "tuple[int, ...], np.ndarray, np.ndarray -> tuple[np.ndarray, int]",
        "Build a feasibility mask over the DP state space given bound constraints.",
    ),
    "compress_dp_table": (
        "sciona.expansion_atoms.runtime_dynamic_programming.compress_dp_table",
        "np.ndarray, int -> tuple[np.ndarray, float]",
        "Retain only the most recent reuse_distance rows of a DP table.",
    ),
    "validate_subproblem_overlap": (
        "sciona.expansion_atoms.runtime_dynamic_programming.validate_subproblem_overlap",
        "np.ndarray -> tuple[float, bool]",
        "Check whether subproblems are reused enough to justify memoization.",
    ),
}
