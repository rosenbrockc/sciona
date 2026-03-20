"""Shared registry for curated signal event-rate primitives and variants."""

from __future__ import annotations


SIGNAL_EVENT_RATE_DECLARATIONS = {
    "filter_signal_for_detection": (
        "sciona.expansion_atoms.runtime_signal_event_rate.filter_signal_for_detection",
        "np.ndarray, float -> np.ndarray",
        "Condition a sampled waveform for downstream peak/event detection.",
    ),
    "detect_peaks_in_signal": (
        "sciona.expansion_atoms.runtime_signal_event_rate.detect_peaks_in_signal",
        "np.ndarray, float -> np.ndarray",
        "Detect salient events in a conditioned waveform using robust thresholds.",
    ),
    "compute_event_rate": (
        "sciona.expansion_atoms.runtime_signal_event_rate.compute_event_rate",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into midpoint indices and per-minute rate.",
    ),
    "compute_event_rate_smoothed": (
        "sciona.expansion_atoms.runtime_signal_event_rate.compute_event_rate_smoothed",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into a smoothed per-minute rate estimate.",
    ),
    # Expansion atoms — inserted by the DPO expansion engine
    "assess_signal_quality": (
        "sciona.expansion_atoms.runtime_signal_event_rate.assess_signal_quality",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Compute per-window signal quality mask using kurtosis.",
    ),
    "remove_signal_jumps": (
        "sciona.expansion_atoms.runtime_signal_event_rate.remove_signal_jumps",
        "np.ndarray, float -> np.ndarray",
        "Remove step discontinuities from raw signal.",
    ),
    "reject_outlier_intervals": (
        "sciona.expansion_atoms.runtime_signal_event_rate.reject_outlier_intervals",
        "np.ndarray, float -> np.ndarray",
        "Remove events creating physiologically implausible intervals.",
    ),
}


SIGNAL_EVENT_RATE_ALTERNATIVES = {
    "compute_event_rate": ("compute_event_rate_smoothed",),
}


def next_signal_event_rate_variant(primitive_name: str) -> str | None:
    """Return the next curated variant for a primitive, when one exists."""
    variants = SIGNAL_EVENT_RATE_ALTERNATIVES.get(primitive_name)
    if not variants:
        return None
    return variants[0]
