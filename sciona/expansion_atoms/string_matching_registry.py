"""Registry for string matching primitives and expansion atoms."""

from __future__ import annotations

STRING_MATCHING_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "analyze_alphabet_size": (
        "sciona.expansion_atoms.runtime_string_matching.analyze_alphabet_size",
        "ndarray, ndarray -> tuple[int, int, float]",
        "Analyze alphabet sizes for algorithm selection guidance.",
    ),
    "check_pattern_text_ratio": (
        "sciona.expansion_atoms.runtime_string_matching.check_pattern_text_ratio",
        "int, int -> tuple[float, str]",
        "Check the ratio of pattern length to text length.",
    ),
    "measure_hash_collision_rate": (
        "sciona.expansion_atoms.runtime_string_matching.measure_hash_collision_rate",
        "int, int -> tuple[float, bool]",
        "Measure the spurious match rate for Rabin-Karp style algorithms.",
    ),
    "validate_failure_function": (
        "sciona.expansion_atoms.runtime_string_matching.validate_failure_function",
        "ndarray, int -> tuple[int, bool]",
        "Validate basic properties of a KMP failure function table.",
    ),
}
