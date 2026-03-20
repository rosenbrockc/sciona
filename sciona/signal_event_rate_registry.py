"""Shared registry for curated signal event-rate primitives and variants."""

from __future__ import annotations


SIGNAL_EVENT_RATE_DECLARATIONS = {
    "filter_signal_for_detection": (
        "sciona.runtime_signal_event_rate.filter_signal_for_detection",
        "np.ndarray, float -> np.ndarray",
        "Condition a sampled waveform for downstream peak/event detection.",
    ),
    "detect_peaks_in_signal": (
        "sciona.runtime_signal_event_rate.detect_peaks_in_signal",
        "np.ndarray, float -> np.ndarray",
        "Detect salient events in a conditioned waveform using robust thresholds.",
    ),
    "compute_event_rate": (
        "sciona.runtime_signal_event_rate.compute_event_rate",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into midpoint indices and per-minute rate.",
    ),
    "compute_event_rate_smoothed": (
        "sciona.runtime_signal_event_rate.compute_event_rate_smoothed",
        "np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
        "Convert ordered event indices into a smoothed per-minute rate estimate.",
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
