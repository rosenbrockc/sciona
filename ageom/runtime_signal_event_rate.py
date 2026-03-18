from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def _coerce_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not finite_mask.all():
        if not finite_mask.any():
            return np.zeros(0, dtype=np.float64)
        fill = float(np.median(values[finite_mask]))
        values = values.copy()
        values[~finite_mask] = fill
    return values


def _coerce_sampling_rate(sampling_rate: float | int) -> float:
    rate = float(sampling_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate!r}")
    return rate


def _robust_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return mad
    std = float(np.std(values))
    return std if std > 0 else 1.0


def filter_signal_for_detection(
    signal: np.ndarray,
    sampling_rate: float | int,
) -> np.ndarray:
    """Condition a sampled waveform for downstream peak/event detection."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size == 0:
        return values

    centered = values - float(np.median(values))
    scale = _robust_scale(centered)
    clipped = np.clip(centered, -8.0 * scale, 8.0 * scale)

    nyquist = rate / 2.0
    high = min(25.0, 0.45 * rate)
    low = min(3.0, high / 3.0)
    if low <= 0 or high <= low or high >= nyquist:
        return clipped

    sos = butter(
        4,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, clipped)


def _pick_peak_orientation(
    values: np.ndarray,
    *,
    distance: int,
    prominence: float,
) -> np.ndarray:
    pos_peaks, pos_props = find_peaks(values, distance=distance, prominence=prominence)
    neg_peaks, neg_props = find_peaks(-values, distance=distance, prominence=prominence)
    pos_score = float(np.median(pos_props["prominences"])) if len(pos_peaks) else 0.0
    neg_score = float(np.median(neg_props["prominences"])) if len(neg_peaks) else 0.0
    return neg_peaks if neg_score > pos_score else pos_peaks


def detect_peaks_in_signal(
    conditioned_signal: np.ndarray,
    sampling_rate: float | int,
) -> np.ndarray:
    """Detect salient peaks in a conditioned waveform using robust thresholds."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(conditioned_signal)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)

    scale = _robust_scale(values)
    prominence = max(1.5 * scale, 1e-6)
    distance = max(1, int(round(0.45 * rate)))
    peaks = _pick_peak_orientation(
        values,
        distance=distance,
        prominence=prominence,
    )
    return np.asarray(peaks, dtype=np.int64)


def compute_event_rate(
    events: np.ndarray,
    sampling_rate: float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered event indices into midpoint indices and per-minute rate."""
    rate = _coerce_sampling_rate(sampling_rate)
    event_idx = np.asarray(events, dtype=np.int64).reshape(-1)
    if event_idx.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    event_idx = np.unique(event_idx[event_idx >= 0])
    if event_idx.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    intervals = np.diff(event_idx).astype(np.float64)
    valid = intervals > 0
    if not np.any(valid):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    intervals = intervals[valid]
    left = event_idx[:-1][valid]
    midpoints = left + (intervals // 2).astype(np.int64)
    event_rate = 60.0 * rate / intervals
    return midpoints.astype(np.int64), event_rate.astype(np.float64)


def compute_event_rate_smoothed(
    events: np.ndarray,
    sampling_rate: float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event indices into a smoothed per-minute rate estimate."""
    midpoints, event_rate = compute_event_rate(events, sampling_rate)
    if event_rate.size == 0:
        return midpoints, event_rate

    if event_rate.size < 5:
        window = max(1, event_rate.size)
    else:
        window = 5
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(event_rate, kernel, mode="same")
    return midpoints, smoothed.astype(np.float64)
