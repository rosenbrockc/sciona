from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter
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
    *,
    filter_order: int = 4,
    clipping_scale: float = 8.0,
    low_cutoff_hz: float = 3.0,
    high_cutoff_hz: float = 25.0,
) -> np.ndarray:
    """Condition a sampled waveform for downstream peak/event detection."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size == 0:
        return values

    centered = values - float(np.median(values))
    scale = _robust_scale(centered)
    clipped = np.clip(centered, -clipping_scale * scale, clipping_scale * scale)

    nyquist = rate / 2.0
    high = min(high_cutoff_hz, 0.45 * rate)
    low = min(low_cutoff_hz, high / 3.0)
    if low <= 0 or high <= low or high >= nyquist:
        return clipped

    sos = butter(
        filter_order,
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
    *,
    prominence_scale: float = 1.5,
    refractory_scale: float = 0.45,
) -> np.ndarray:
    """Detect salient peaks in a conditioned waveform using robust thresholds."""
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(conditioned_signal)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)

    scale = _robust_scale(values)
    prominence = max(prominence_scale * scale, 1e-6)
    distance = max(1, int(round(refractory_scale * rate)))
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
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event indices into a smoothed per-minute rate estimate."""
    midpoints, event_rate = compute_event_rate(events, sampling_rate)
    if event_rate.size == 0:
        return midpoints, event_rate

    if event_rate.size < smoothing_window:
        window = max(1, event_rate.size)
    else:
        window = smoothing_window
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(event_rate, kernel, mode="same")
    return midpoints, smoothed.astype(np.float64)


def compute_event_rate_median_smoothed(
    events: np.ndarray,
    sampling_rate: float | int,
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert event indices into a robust median-smoothed rate estimate."""
    midpoints, event_rate = compute_event_rate(events, sampling_rate)
    if event_rate.size == 0:
        return midpoints, event_rate

    window = max(1, int(smoothing_window))
    if window > event_rate.size:
        window = int(event_rate.size)
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return midpoints, event_rate

    smoothed = median_filter(event_rate.astype(np.float64), size=window, mode="nearest")
    return midpoints, np.asarray(smoothed, dtype=np.float64)


# ---------------------------------------------------------------------------
# Expansion atoms — inserted by the DPO expansion engine
# ---------------------------------------------------------------------------


def assess_signal_quality(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    window_seconds: float = 10.0,
    min_kurtosis: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window signal quality mask using excess kurtosis.

    Windows with kurtosis below *min_kurtosis* are flagged as low-quality
    (ECG with sharp R-peaks has high kurtosis; noise does not).

    Returns ``(signal_passthrough, quality_mask)`` where ``quality_mask``
    is a boolean array (True = adequate quality).
    """
    rate = _coerce_sampling_rate(sampling_rate)
    values = _coerce_signal(signal)
    if values.size == 0:
        return values, np.ones(0, dtype=bool)

    window = max(1, int(round(window_seconds * rate)))
    mask = np.ones(values.size, dtype=bool)

    for start in range(0, values.size, window):
        end = min(start + window, values.size)
        seg = values[start:end]
        if seg.size < 4:
            continue
        centered = seg - np.mean(seg)
        std = float(np.std(centered))
        if std < 1e-10:
            mask[start:end] = False
            continue
        kurt = float(np.mean((centered / std) ** 4)) - 3.0
        if kurt < min_kurtosis:
            mask[start:end] = False

    return values, mask


def remove_signal_jumps(
    signal: np.ndarray,
    sampling_rate: float | int,
    *,
    jump_threshold_scale: float = 5.0,
) -> np.ndarray:
    """Remove step discontinuities from a raw signal.

    Detects sample-to-sample jumps exceeding ``jump_threshold_scale * MAD``
    of the first-differenced signal, then subtracts the cumulative shift
    so downstream filters see a continuous baseline.
    """
    _coerce_sampling_rate(sampling_rate)  # validate
    values = _coerce_signal(signal)
    if values.size < 2:
        return values

    diff = np.diff(values)
    median_diff = float(np.median(diff))
    mad_diff = float(np.median(np.abs(diff - median_diff)))
    if mad_diff < 1e-10:
        return values

    threshold = jump_threshold_scale * mad_diff
    jumps = np.where(np.abs(diff - median_diff) > threshold)[0]
    if jumps.size == 0:
        return values

    result = values.copy()
    for idx in jumps:
        shift = result[idx + 1] - result[idx]
        result[idx + 1 :] -= shift

    return result


def reject_outlier_intervals(
    events: np.ndarray,
    sampling_rate: float | int,
    *,
    mad_scale: float = 3.0,
) -> np.ndarray:
    """Remove detected events that create physiologically implausible intervals.

    Keeps an event only if at least one of its adjacent inter-event intervals
    falls within ``median ± mad_scale * MAD`` of all intervals.
    """
    _coerce_sampling_rate(sampling_rate)  # validate
    idx = np.asarray(events, dtype=np.int64).reshape(-1)
    if idx.size < 3:
        return idx

    idx = np.unique(idx)
    intervals = np.diff(idx).astype(np.float64)
    median_ivl = float(np.median(intervals))
    mad_ivl = float(np.median(np.abs(intervals - median_ivl)))
    if mad_ivl < 1e-10:
        return idx

    lo = max(1.0, median_ivl - mad_scale * mad_ivl)
    hi = median_ivl + mad_scale * mad_ivl
    good = (intervals >= lo) & (intervals <= hi)

    # Keep a peak if at least one adjacent interval is good.
    keep = np.ones(idx.size, dtype=bool)
    for i in range(1, idx.size - 1):
        left_good = bool(good[i - 1])
        right_good = bool(good[i])
        if not left_good and not right_good:
            keep[i] = False

    return idx[keep]
