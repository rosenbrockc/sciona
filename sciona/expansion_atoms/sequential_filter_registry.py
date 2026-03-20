"""Registry for sequential filter (Kalman/particle) primitives and expansion atoms."""

from __future__ import annotations

SEQUENTIAL_FILTER_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "check_observability": (
        "sciona.expansion_atoms.runtime_sequential_filter.check_observability",
        "np.ndarray, np.ndarray, int -> tuple[bool, np.ndarray]",
        "Check observability of a linear system (F, H) via rank test.",
    ),
    "validate_innovation_whiteness": (
        "sciona.expansion_atoms.runtime_sequential_filter.validate_innovation_whiteness",
        "np.ndarray, int -> tuple[np.ndarray, bool]",
        "Test whether innovation sequence is white (uncorrelated).",
    ),
    "detect_filter_divergence": (
        "sciona.expansion_atoms.runtime_sequential_filter.detect_filter_divergence",
        "np.ndarray, np.ndarray -> tuple[np.ndarray, np.ndarray]",
        "Detect Kalman filter divergence via NIS chi-squared test.",
    ),
    "adapt_process_noise": (
        "sciona.expansion_atoms.runtime_sequential_filter.adapt_process_noise",
        "np.ndarray, np.ndarray, np.ndarray -> np.ndarray",
        "Adaptively estimate process noise Q from innovations via Robbins-Monro.",
    ),
}
