"""Runtime atoms for baseline-path scoring in AHI-style analyzers.

These atoms intentionally mirror the baseline-only scoring layer used by the
canonical HappyML AHI configuration while staying generic enough to compose
into a CDG:

  - accumulate analyzed sleep time from an anchor-aligned sleep mask
  - accumulate padded prediction-window coverage from component probabilities
  - compute hourly event rates from labeled or interval-style events
  - apply the mild-path BMI correction
  - score the baseline SQI path (`sAHI`)
  - score the BMI-corrected baseline path (`bAHI`)
  - score the PAT baseline path (`pAHI`)
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_BMI = 22.0


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_bool_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=bool).reshape(-1)


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask_arr = _as_bool_array(mask)
    if mask_arr.size == 0:
        return []
    padded = np.concatenate(([False], mask_arr, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _merge_intervals(intervals: list[tuple[float, float]]) -> np.ndarray:
    if not intervals:
        return np.empty((0, 2), dtype=np.float64)

    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = [[float(ordered[0][0]), float(ordered[0][1])]]
    for start, stop in ordered[1:]:
        current = merged[-1]
        if float(start) <= current[1]:
            current[1] = max(current[1], float(stop))
        else:
            merged.append([float(start), float(stop)])
    return np.asarray(merged, dtype=np.float64)


def _count_mean_crossings(values: np.ndarray) -> int:
    arr = _as_float_array(values)
    if arr.size < 2:
        return 0
    centered = arr - float(np.mean(arr))
    signs = centered >= 0.0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _mode_value(values: np.ndarray) -> float:
    arr = _as_float_array(values)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    uniq, counts = np.unique(arr, return_counts=True)
    return float(uniq[int(np.argmax(counts))])


def _moving_average(values: np.ndarray, window: int = 11) -> np.ndarray:
    arr = _as_float_array(values)
    if arr.size == 0:
        return arr
    width = max(1, min(int(window), arr.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(arr, kernel, mode="same")


def _count_events(events: np.ndarray | int | float) -> int:
    if np.isscalar(events):
        value = float(events)
        if not math.isfinite(value) or value <= 0.0:
            return 0
        return int(round(value))

    arr = np.asarray(events)
    if arr.size == 0:
        return 0
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return int(arr.shape[0])

    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0

    rounded = np.round(flat)
    if np.allclose(flat, rounded):
        ints = rounded.astype(np.int64, copy=False)
        unique = np.unique(ints)
        if np.all(np.isin(unique, [0, 1])):
            return len(_contiguous_regions(ints > 0))
        return int(np.unique(ints[ints > 0]).size)

    return int(np.count_nonzero(flat > 0.0))


def _short_night_adjustment(ahi: float, analyzed_time_hours: float) -> float:
    if analyzed_time_hours < 5.0 and ahi < 25.0:
        return float(2.0 * (ahi + 2.5))
    return float(ahi)


def _filter_baseline_score(ahi: float, density_hours: float) -> float:
    if not math.isfinite(ahi) or not math.isfinite(density_hours):
        return float("nan")
    if density_hours < 3.0 and ahi < 6.0:
        return float("nan")
    return float(ahi)


def _infer_spo2_moderate_or_severe(
    spo2_probabilities: np.ndarray | None,
    analyzed_time_hours: float,
) -> bool:
    if spo2_probabilities is None or analyzed_time_hours <= 0.0:
        return False

    probabilities = _as_float_array(spo2_probabilities)
    if probabilities.size == 0:
        return False

    residual = np.abs(probabilities - _mode_value(probabilities))
    flattened = _moving_average(residual, window=11)
    crossings = _count_mean_crossings(residual)
    residual_crossings = _count_mean_crossings(np.abs(residual - flattened))
    hahi = crossings / analyzed_time_hours
    jahi = residual_crossings / analyzed_time_hours
    denom = math.log1p(hahi)
    if denom <= 0.0:
        return False
    x = math.log1p(hahi)
    y = math.log1p(math.log1p(jahi) / denom)
    return bool(x < 4.0 and 0.76 < y < 0.89)


def _score_combined_path(
    combined_events: np.ndarray | int | float,
    analyzed_time_hours: float,
    density_hours: float,
    *,
    moderate_or_severe: bool,
) -> float:
    ahi, ok = compute_event_rate_per_hour(combined_events, analyzed_time_hours)
    if not ok:
        return float("nan")
    ahi = _filter_baseline_score(ahi, density_hours)
    if np.isnan(ahi):
        return ahi
    if moderate_or_severe and ahi < 20.0:
        ahi *= 1.5
    return _short_night_adjustment(ahi, analyzed_time_hours)


def _score_combined_bmi_path(
    combined_events: np.ndarray | int | float,
    analyzed_time_hours: float,
    density_hours: float,
    bmi: float | None,
    *,
    moderate_or_severe: bool,
) -> float:
    ahi = _score_combined_path(
        combined_events,
        analyzed_time_hours,
        density_hours,
        moderate_or_severe=moderate_or_severe,
    )
    if np.isnan(ahi):
        return ahi

    bmi_value = DEFAULT_BMI if bmi is None or not math.isfinite(float(bmi)) else float(bmi)
    dbmi = (bmi_value - 18.0) / 65.0
    return float(ahi * (1.0 + (dbmi * 1.5) ** 2))


def accumulate_analyzed_time(
    anchor: np.ndarray,
    sleep_mask: np.ndarray,
    *,
    seconds_per_hour: float = 3600.0,
) -> tuple[float, bool]:
    """Accumulate analyzed time from sleep-mask regions aligned to an anchor."""
    anchor_arr = _as_float_array(anchor)
    mask_arr = _as_bool_array(sleep_mask)
    if anchor_arr.size == 0 or mask_arr.size == 0:
        return 0.0, False
    if anchor_arr.size != mask_arr.size:
        raise ValueError("anchor and sleep_mask must have the same length")
    if seconds_per_hour <= 0.0:
        raise ValueError("seconds_per_hour must be positive")

    seconds = 0.0
    for start, stop in _contiguous_regions(mask_arr):
        seconds += max(0.0, float(anchor_arr[stop - 1] - anchor_arr[start]))
    hours = seconds / float(seconds_per_hour)
    return hours, hours > 0.0


def accumulate_prediction_window_time(
    probabilities: np.ndarray,
    anchor: np.ndarray,
    *,
    threshold: float = 0.0,
    pad: float = 300.0,
    seconds_per_hour: float = 3600.0,
) -> tuple[float, bool]:
    """Accumulate padded prediction-window coverage from component probabilities."""
    probability_arr = _as_float_array(probabilities)
    anchor_arr = _as_float_array(anchor)
    if probability_arr.size == 0 or anchor_arr.size == 0:
        return 0.0, False
    if probability_arr.size != anchor_arr.size:
        raise ValueError("probabilities and anchor must have the same length")
    if seconds_per_hour <= 0.0:
        raise ValueError("seconds_per_hour must be positive")

    intervals: list[tuple[float, float]] = []
    for start, stop in _contiguous_regions(probability_arr > threshold):
        intervals.append((float(anchor_arr[start] - pad), float(anchor_arr[stop - 1] + pad)))

    if not intervals:
        return 0.0, False

    merged = _merge_intervals(intervals)
    seconds = float(np.sum(merged[:, 1] - merged[:, 0])) if merged.size > 0 else 0.0
    hours = seconds / float(seconds_per_hour)
    return hours, hours > 0.0


def compute_event_rate_per_hour(
    events: np.ndarray | int | float,
    analyzed_time_hours: float,
    *,
    divisor: float = 1.0,
) -> tuple[float, bool]:
    """Compute an hourly event rate from event labels, intervals, or counts."""
    hours = float(analyzed_time_hours)
    if not math.isfinite(hours) or hours <= 0.0:
        return float("nan"), False
    if divisor <= 0.0:
        raise ValueError("divisor must be positive")

    count = _count_events(events)
    return float(count / hours / float(divisor)), True


def apply_bmi_correction(
    ahi: float,
    bmi: float | None,
    *,
    default_bmi: float = DEFAULT_BMI,
    subtract_bias: float = 4.0,
    clip_min: float = 0.0,
    clip_max: float = 100.0,
) -> tuple[float, bool]:
    """Apply the baseline-path BMI correction used by the mild baseline branch."""
    ahi_value = float(ahi)
    if not math.isfinite(ahi_value):
        return float("nan"), False

    bmi_value = default_bmi if bmi is None or not math.isfinite(float(bmi)) else float(bmi)
    dbmi = (bmi_value - 18.0) / 65.0
    corrected = ahi_value * (1.0 + (dbmi * 1.5) ** 2) - float(subtract_bias)
    corrected = float(np.clip(corrected, clip_min, clip_max))
    return corrected, True


def score_baseline_path(
    predictor_events: np.ndarray | int | float,
    combined_events: np.ndarray | int | float,
    analyzed_time_hours: float,
    density_hours: float,
    *,
    spo2_probabilities: np.ndarray | None = None,
    moderate_or_severe: bool | None = None,
) -> tuple[float, bool]:
    """Score the SQI baseline path and return an `sAHI`-style value."""
    ahi, ok = compute_event_rate_per_hour(
        predictor_events,
        analyzed_time_hours,
        divisor=2.0,
    )
    if not ok:
        return float("nan"), False

    ahi = _filter_baseline_score(ahi, float(density_hours))
    if np.isnan(ahi):
        return ahi, False

    ahi = _short_night_adjustment(ahi, float(analyzed_time_hours))
    if ahi > 5.0:
        modsev = (
            _infer_spo2_moderate_or_severe(spo2_probabilities, float(analyzed_time_hours))
            if moderate_or_severe is None
            else bool(moderate_or_severe)
        )
        ahi = _score_combined_path(
            combined_events,
            float(analyzed_time_hours),
            float(density_hours),
            moderate_or_severe=modsev,
        )

    return float(ahi), bool(np.isfinite(ahi))


def score_bmi_baseline_path(
    predictor_events: np.ndarray | int | float,
    combined_events: np.ndarray | int | float,
    analyzed_time_hours: float,
    density_hours: float,
    bmi: float | None,
    *,
    spo2_probabilities: np.ndarray | None = None,
    moderate_or_severe: bool | None = None,
) -> tuple[float, bool]:
    """Score the BMI-corrected SQI baseline path and return a `bAHI`-style value."""
    ahi, ok = compute_event_rate_per_hour(
        predictor_events,
        analyzed_time_hours,
        divisor=2.0,
    )
    if not ok:
        return float("nan"), False

    ahi = _filter_baseline_score(ahi, float(density_hours))
    if np.isnan(ahi):
        return ahi, False

    ahi = _short_night_adjustment(ahi, float(analyzed_time_hours))
    ahi, _ = apply_bmi_correction(ahi, bmi)
    if ahi > 5.0:
        modsev = (
            _infer_spo2_moderate_or_severe(spo2_probabilities, float(analyzed_time_hours))
            if moderate_or_severe is None
            else bool(moderate_or_severe)
        )
        ahi = _score_combined_bmi_path(
            combined_events,
            float(analyzed_time_hours),
            float(density_hours),
            bmi,
            moderate_or_severe=modsev,
        )

    return float(ahi), bool(np.isfinite(ahi))


def score_pat_baseline_path(
    pat_events: np.ndarray | int | float,
    analyzed_time_hours: float,
    density_hours: float,
) -> tuple[float, bool]:
    """Score the PAT baseline branch and return a `pAHI`-style value."""
    ahi, ok = compute_event_rate_per_hour(pat_events, analyzed_time_hours, divisor=2.0)
    if not ok:
        return float("nan"), False

    ahi = _filter_baseline_score(ahi, float(density_hours))
    if np.isnan(ahi):
        return ahi, False

    ahi = _short_night_adjustment(ahi, float(analyzed_time_hours))
    return float(ahi), True
