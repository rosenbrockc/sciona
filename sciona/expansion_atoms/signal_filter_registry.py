"""Registry for signal filter primitives and expansion atoms."""

from __future__ import annotations

SIGNAL_FILTER_DECLARATIONS = {
    "analyze_pole_stability": (
        "sciona.expansion_atoms.runtime_signal_filter.analyze_pole_stability",
        "ndarray, float -> tuple[float, bool]",
        "Analyze filter stability from pole locations.",
    ),
    "measure_passband_ripple": (
        "sciona.expansion_atoms.runtime_signal_filter.measure_passband_ripple",
        "ndarray, ndarray -> tuple[float, bool]",
        "Measure peak-to-peak ripple in the filter passband.",
    ),
    "analyze_group_delay_variation": (
        "sciona.expansion_atoms.runtime_signal_filter.analyze_group_delay_variation",
        "ndarray -> tuple[float, bool]",
        "Analyze group delay variation across frequency.",
    ),
    "detect_transient_response": (
        "sciona.expansion_atoms.runtime_signal_filter.detect_transient_response",
        "ndarray, int -> tuple[int, float]",
        "Detect startup transient in filter output.",
    ),
}
