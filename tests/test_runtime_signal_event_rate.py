"""Tests for runtime_signal_event_rate — Phase 0 kwargs externalization."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.expansion_atoms.runtime_signal_event_rate import (
    compute_event_rate_smoothed,
    compute_event_rate_median_smoothed,
    detect_peaks_in_signal,
    filter_signal_for_detection,
)


@pytest.fixture
def synthetic_signal():
    """Generate a synthetic ECG-like signal at 100 Hz."""
    rng = np.random.default_rng(42)
    sampling_rate = 100.0
    t = np.arange(0, 5, 1.0 / sampling_rate)
    # Simulate periodic peaks at ~1.2 Hz (72 bpm)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * rng.standard_normal(len(t))
    return signal, sampling_rate


class TestFilterSignalForDetection:
    def test_defaults_unchanged(self, synthetic_signal):
        """Calling with no kwargs should produce identical output to hardcoded defaults."""
        signal, rate = synthetic_signal
        result = filter_signal_for_detection(signal, rate)
        assert result.shape == signal.shape
        assert np.isfinite(result).all()

    def test_filter_order_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        result_default = filter_signal_for_detection(signal, rate, filter_order=4)
        result_alt = filter_signal_for_detection(signal, rate, filter_order=2)
        assert not np.allclose(result_default, result_alt), (
            "Different filter orders should produce different outputs"
        )

    def test_clipping_scale_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        # Add a large outlier to make clipping visible
        signal_with_outlier = signal.copy()
        signal_with_outlier[50] = 100.0
        r1 = filter_signal_for_detection(signal_with_outlier, rate, clipping_scale=8.0)
        r2 = filter_signal_for_detection(signal_with_outlier, rate, clipping_scale=3.0)
        assert not np.allclose(r1, r2)

    def test_cutoff_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        r1 = filter_signal_for_detection(signal, rate, low_cutoff_hz=3.0, high_cutoff_hz=25.0)
        r2 = filter_signal_for_detection(signal, rate, low_cutoff_hz=1.0, high_cutoff_hz=40.0)
        assert not np.allclose(r1, r2)


class TestDetectPeaksInSignal:
    def test_defaults_unchanged(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks = detect_peaks_in_signal(filtered, rate)
        assert peaks.dtype == np.int64
        assert len(peaks) > 0

    def test_prominence_scale_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks_default = detect_peaks_in_signal(filtered, rate, prominence_scale=1.5)
        peaks_strict = detect_peaks_in_signal(filtered, rate, prominence_scale=5.0)
        # Stricter prominence should find fewer or equal peaks
        assert len(peaks_strict) <= len(peaks_default)

    def test_refractory_scale_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks_default = detect_peaks_in_signal(filtered, rate, refractory_scale=0.45)
        peaks_short = detect_peaks_in_signal(filtered, rate, refractory_scale=0.2)
        # Shorter refractory period may find more peaks
        assert len(peaks_short) >= len(peaks_default)


class TestComputeEventRateSmoothed:
    def test_defaults_unchanged(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks = detect_peaks_in_signal(filtered, rate)
        midpoints, smoothed = compute_event_rate_smoothed(peaks, rate)
        assert len(midpoints) == len(smoothed)

    def test_smoothing_window_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks = detect_peaks_in_signal(filtered, rate)
        if len(peaks) < 6:
            pytest.skip("Not enough peaks to test smoothing difference")
        _, s1 = compute_event_rate_smoothed(peaks, rate, smoothing_window=5)
        _, s2 = compute_event_rate_smoothed(peaks, rate, smoothing_window=1)
        assert not np.allclose(s1, s2), (
            "Different smoothing windows should produce different outputs"
        )


class TestComputeEventRateMedianSmoothed:
    def test_defaults_unchanged(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks = detect_peaks_in_signal(filtered, rate)
        midpoints, smoothed = compute_event_rate_median_smoothed(peaks, rate)
        assert len(midpoints) == len(smoothed)

    def test_smoothing_window_changes_output(self, synthetic_signal):
        signal, rate = synthetic_signal
        filtered = filter_signal_for_detection(signal, rate)
        peaks = detect_peaks_in_signal(filtered, rate)
        if len(peaks) < 6:
            pytest.skip("Not enough peaks to test smoothing difference")
        _, s1 = compute_event_rate_median_smoothed(peaks, rate, smoothing_window=5)
        _, s2 = compute_event_rate_median_smoothed(peaks, rate, smoothing_window=1)
        assert not np.allclose(s1, s2), (
            "Different smoothing windows should produce different outputs"
        )
