"""Runtime atoms for Signal Detect Measure expansion rules.

Provides deterministic, pure functions for signal detection and
measurement diagnostics:

  - SNR estimation (signal-to-noise quality before detection)
  - Peak threshold sensitivity (detection stability analysis)
  - Event rate stationarity check (rate consistency over time)
  - False positive rate estimation (detection reliability)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


def estimate_snr(
    signal: np.ndarray,
    noise_floor: float = 0.0,
) -> tuple[float, bool]:
    """Estimate signal-to-noise ratio.

    Low SNR degrades peak detection reliability.

    Args:
        signal: 1-D signal array.
        noise_floor: estimated noise power (0 = auto-estimate from median).

    Returns:
        (snr_db, is_sufficient) where snr_db is the SNR in dB and
        is_sufficient is True if snr_db > 10.
    """
    s = np.asarray(signal, dtype=np.float64).ravel()
    if len(s) < 2:
        return 0.0, False

    signal_power = float(np.mean(s ** 2))

    if noise_floor <= 0:
        # Estimate noise from median absolute deviation
        mad = float(np.median(np.abs(s - np.median(s))))
        noise_power = (mad * 1.4826) ** 2  # MAD to std approximation
    else:
        noise_power = noise_floor

    if noise_power == 0:
        return float("inf"), True

    snr = signal_power / noise_power
    snr_db = 10.0 * np.log10(max(snr, 1e-30))
    return float(snr_db), snr_db > 10.0


# ---------------------------------------------------------------------------
# Peak threshold sensitivity
# ---------------------------------------------------------------------------


def analyze_peak_threshold_sensitivity(
    peaks: np.ndarray,
    threshold: float,
) -> tuple[float, bool]:
    """Analyze how sensitive detection count is to threshold changes.

    If many peaks cluster near the threshold, small changes in the
    threshold cause large changes in detection count.

    Args:
        peaks: 1-D array of peak amplitudes.
        threshold: current detection threshold.

    Returns:
        (sensitivity, is_stable) where sensitivity is the fraction of
        peaks within 10% of the threshold and is_stable is True if
        sensitivity < 0.2.
    """
    p = np.asarray(peaks, dtype=np.float64).ravel()
    if len(p) == 0 or threshold == 0:
        return 0.0, True

    margin = abs(threshold) * 0.1
    near_threshold = np.sum(np.abs(p - threshold) < margin)
    sensitivity = float(near_threshold) / len(p)
    return sensitivity, sensitivity < 0.2


# ---------------------------------------------------------------------------
# Event rate stationarity
# ---------------------------------------------------------------------------


def check_event_rate_stationarity(
    event_times: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, bool]:
    """Check whether the event rate is stationary over time.

    Non-stationary rates suggest the signal statistics are changing
    during the measurement window.

    Args:
        event_times: 1-D array of event timestamps.
        n_bins: number of time bins to divide the observation into.

    Returns:
        (coefficient_of_variation, is_stationary) where
        coefficient_of_variation is std(counts) / mean(counts) and
        is_stationary is True if CV < 0.5.
    """
    t = np.asarray(event_times, dtype=np.float64).ravel()
    if len(t) < 2:
        return 0.0, True

    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_max == t_min:
        return 0.0, True

    bins = np.linspace(t_min, t_max, n_bins + 1)
    counts, _ = np.histogram(t, bins=bins)
    counts = counts.astype(float)

    mean_count = float(np.mean(counts))
    if mean_count == 0:
        return 0.0, True

    cv = float(np.std(counts)) / mean_count
    return cv, cv < 0.5


# ---------------------------------------------------------------------------
# False positive rate estimation
# ---------------------------------------------------------------------------


def estimate_false_positive_rate(
    detected_amplitudes: np.ndarray,
    noise_std: float,
    threshold: float,
) -> tuple[float, bool]:
    """Estimate the false positive detection rate.

    Uses the noise distribution to estimate the probability that
    pure noise exceeds the detection threshold.

    Args:
        detected_amplitudes: 1-D array of detected peak amplitudes.
        noise_std: estimated standard deviation of the noise.
        threshold: detection threshold.

    Returns:
        (estimated_fpr, is_reliable) where estimated_fpr is the
        fraction of detections likely due to noise and is_reliable
        is True if fpr < 0.05.
    """
    d = np.asarray(detected_amplitudes, dtype=np.float64).ravel()
    if len(d) == 0 or noise_std <= 0:
        return 0.0, True

    # Detections within 2*noise_std of threshold are suspect
    suspect = np.sum(d < threshold + 2 * noise_std)
    fpr = float(suspect) / len(d)
    return fpr, fpr < 0.05
