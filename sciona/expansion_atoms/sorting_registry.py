"""Registry for sorting primitives and expansion atoms."""

from __future__ import annotations

SORTING_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "measure_presortedness": (
        "sciona.expansion_atoms.runtime_sorting.measure_presortedness",
        "ndarray -> tuple[float, int]",
        "Measure how sorted the input already is.",
    ),
    "analyze_comparison_count": (
        "sciona.expansion_atoms.runtime_sorting.analyze_comparison_count",
        "int, int -> tuple[float, bool]",
        "Check whether comparison count is excessive.",
    ),
    "analyze_swap_count": (
        "sciona.expansion_atoms.runtime_sorting.analyze_swap_count",
        "int, int -> tuple[float, bool]",
        "Check whether swap/move count is excessive.",
    ),
    "validate_stability": (
        "sciona.expansion_atoms.runtime_sorting.validate_stability",
        "ndarray, ndarray, ndarray -> tuple[int, bool]",
        "Check whether a sort preserves the relative order of equal keys.",
    ),
}
