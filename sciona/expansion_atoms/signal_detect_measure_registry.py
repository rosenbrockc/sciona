"""Registry for signal detect measure primitives and expansion atoms."""

from __future__ import annotations

SIGNAL_DETECT_MEASURE_DECLARATIONS = {
    "estimate_snr": (
        "sciona.expansion_atoms.runtime_signal_detect_measure.estimate_snr",
        "ndarray, float -> tuple[float, bool]",
        "Estimate signal-to-noise ratio.",
    ),
    "analyze_peak_threshold_sensitivity": (
        "sciona.expansion_atoms.runtime_signal_detect_measure.analyze_peak_threshold_sensitivity",
        "ndarray, float -> tuple[float, bool]",
        "Analyze how sensitive detection count is to threshold changes.",
    ),
    "check_event_rate_stationarity": (
        "sciona.expansion_atoms.runtime_signal_detect_measure.check_event_rate_stationarity",
        "ndarray, int -> tuple[float, bool]",
        "Check whether the event rate is stationary over time.",
    ),
    "estimate_false_positive_rate": (
        "sciona.expansion_atoms.runtime_signal_detect_measure.estimate_false_positive_rate",
        "ndarray, float, float -> tuple[float, bool]",
        "Estimate the false positive detection rate.",
    ),
}
