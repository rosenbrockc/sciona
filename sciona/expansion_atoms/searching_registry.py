"""Registry for searching primitives and expansion atoms."""

from __future__ import annotations

SEARCHING_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "validate_sorted_order": (
        "sciona.expansion_atoms.runtime_searching.validate_sorted_order",
        "ndarray -> tuple[int, bool]",
        "Validate that input data is sorted in non-decreasing order.",
    ),
    "analyze_distribution_uniformity": (
        "sciona.expansion_atoms.runtime_searching.analyze_distribution_uniformity",
        "ndarray -> tuple[float, str]",
        "Analyze how uniformly distributed the search data is.",
    ),
    "detect_midpoint_overflow": (
        "sciona.expansion_atoms.runtime_searching.detect_midpoint_overflow",
        "int, int -> tuple[bool, int]",
        "Detect potential integer overflow in midpoint calculation.",
    ),
    "analyze_iteration_count": (
        "sciona.expansion_atoms.runtime_searching.analyze_iteration_count",
        "int, int -> tuple[float, bool]",
        "Check whether search iteration count is excessive.",
    ),
}
