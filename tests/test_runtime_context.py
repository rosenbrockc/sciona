"""Focused tests for canonical runtime context resolution and summaries."""

from __future__ import annotations

import numpy as np

from sciona.principal.runtime_context import (
    resolve_canonical_runtime_context,
    summarize_events,
    summarize_runtime_context,
    summarize_waveform,
)


def test_resolve_canonical_runtime_context_prefers_ecg_stream() -> None:
    signal_data = {
        "capnostream_value": np.linspace(0.0, 1.0, 30),
        "capnostream_sampling_rate": 21.0,
        "h10_ecg_value": np.sin(np.linspace(0.0, 10.0, 300)),
        "ecg_sampling_rate": 129.96,
        "h10_ecg_t": np.linspace(0.0, 10.0, 300),
    }

    context = resolve_canonical_runtime_context(signal_data)

    assert context.canonical_inputs["signal"].raw_key == "h10_ecg_value"
    assert context.canonical_inputs["sampling_rate"].raw_key == "ecg_sampling_rate"
    assert context.canonical_inputs["sampling_rate"].stream_id == "ecg"
    assert context.alias_resolution["signal"] == "h10_ecg_value"


def test_runtime_context_tracks_stream_aliases_and_sampling_rates() -> None:
    context = resolve_canonical_runtime_context(
        {
            "h10_ecg_value": np.array([0.0, 1.0, 0.0]),
            "ecg_sampling_rate": 128.0,
            "ppg_value": np.array([1.0, 2.0, 1.0]),
            "ppg_sampling_rate": 64.0,
        }
    )

    by_stream = {stream.stream_id: stream for stream in context.streams}

    assert sorted(by_stream) == ["ecg", "ppg"]
    assert "h10_ecg_value" in by_stream["ecg"].aliases
    assert by_stream["ecg"].sampling_rate == 128.0
    assert by_stream["ppg"].sampling_rate == 64.0


def test_summarize_waveform_emits_compact_artifact_metrics() -> None:
    signal = np.zeros(200, dtype=float)
    for idx in range(20, signal.size, 20):
        signal[idx:] += 10.0

    summary = summarize_waveform(signal)

    assert summary["count"] == 200.0
    assert summary["max_abs"] >= 10.0
    assert summary["discontinuity_count"] >= 1.0


def test_summarize_events_emits_density_and_interval_stats() -> None:
    events = np.array([0, 128, 256, 384, 900, 1028], dtype=float)

    summary = summarize_events(events, sampling_rate=128.0)

    assert summary["count"] == 6.0
    assert summary["density_per_minute"] > 0.0
    assert summary["interval_median_samples"] == 128.0
    assert summary["outlier_fraction"] > 0.0


def test_summarize_runtime_context_has_stable_schema() -> None:
    context = resolve_canonical_runtime_context(
        {
            "h10_ecg_value": np.array([0.0, 1.0, 0.0]),
            "ecg_sampling_rate": 128.0,
        }
    )

    summary = summarize_runtime_context(context)

    assert summary["stream_count"] == 1
    assert summary["canonical_inputs"]["signal"] == "h10_ecg_value"
    assert summary["sampling_rates"] == {"ecg": 128.0}
