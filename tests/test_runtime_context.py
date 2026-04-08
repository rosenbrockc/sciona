"""Focused tests for canonical runtime context resolution and summaries."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.principal.runtime_context import (
    canonicalize_intermediates,
    canonicalize_runtime_inputs,
    canonicalize_signal_data,
    resolve_canonical_runtime_context,
    serialize_runtime_context,
    summarize_events,
    summarize_runtime_evidence,
    summarize_runtime_context,
    summarize_series,
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
    by_stream = {stream.stream_id: stream for stream in context.streams}

    assert context.canonical_inputs["signal"].raw_key == "h10_ecg_value"
    assert context.canonical_inputs["sampling_rate"].raw_key == "ecg_sampling_rate"
    assert context.canonical_inputs["sampling_rate"].stream_id == "ecg"
    assert context.canonical_inputs["time"].raw_key == "h10_ecg_t"
    assert context.alias_resolution["signal"] == "h10_ecg_value"
    assert context.canonical_inputs["signal"].provenance == "preferred_waveform_stream"
    assert context.canonical_inputs["sampling_rate"].provenance == "stream_sampling_rate"
    assert by_stream["ecg"].provenance == "primary_stream"
    assert by_stream["capnostream"].provenance == "auxiliary_stream"


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


def test_canonicalize_signal_data_overrides_wrong_generic_aliases() -> None:
    ecg = np.sin(np.linspace(0.0, 10.0, 300))
    signal_data = {
        "signal": np.linspace(0.0, 1.0, 30),
        "sampling_rate": 21.0,
        "time": np.linspace(0.0, 10.0, 30),
        "capnostream_value": np.linspace(0.0, 1.0, 30),
        "capnostream_sampling_rate": 21.0,
        "h10_ecg_value": ecg,
        "ecg_sampling_rate": 129.96,
        "h10_ecg_t": np.linspace(0.0, 10.0, 300),
    }

    normalized, context = canonicalize_signal_data(signal_data)

    assert context.canonical_inputs["signal"].raw_key == "h10_ecg_value"
    assert normalized["signal"] is ecg
    assert normalized["sampling_rate"] == pytest.approx(129.96)
    assert np.array_equal(normalized["time"], signal_data["h10_ecg_t"])


def test_canonicalize_runtime_inputs_overrides_wrong_generic_aliases() -> None:
    ecg = np.sin(np.linspace(0.0, 10.0, 300))
    runtime_inputs = {
        "signal": np.linspace(0.0, 1.0, 30),
        "sampling_rate": 21.0,
        "time": np.linspace(0.0, 10.0, 30),
        "capnostream_value": np.linspace(0.0, 1.0, 30),
        "capnostream_sampling_rate": 21.0,
        "h10_ecg_value": ecg,
        "ecg_sampling_rate": 129.96,
        "h10_ecg_t": np.linspace(0.0, 10.0, 300),
    }

    normalized, context = canonicalize_runtime_inputs(runtime_inputs)

    assert context.canonical_inputs["signal"].raw_key == "h10_ecg_value"
    assert normalized["signal"] is ecg
    assert normalized["sampling_rate"] == pytest.approx(129.96)
    assert np.array_equal(normalized["time"], runtime_inputs["h10_ecg_t"])


def test_canonicalize_intermediates_adds_events_and_rate_aliases() -> None:
    intermediates, aliases = canonicalize_intermediates(
        {
            "rpeaks": np.array([10.0, 20.0, 30.0]),
            "heart_rate": np.array([70.0, 71.0, 69.0]),
        }
    )

    assert aliases == {"events": "rpeaks", "rate": "heart_rate"}
    assert np.array_equal(intermediates["events"], intermediates["rpeaks"])
    assert np.array_equal(intermediates["rate"], intermediates["heart_rate"])


def test_resolve_canonical_runtime_context_binds_generic_signal_to_single_stream() -> None:
    context = resolve_canonical_runtime_context(
        {
            "signal": np.array([0.0, 1.0, 0.0]),
            "time": np.array([0.0, 0.01, 0.02]),
            "ecg_sampling_rate": 100.0,
        }
    )

    assert context.canonical_inputs["signal"].raw_key == "signal"
    assert context.canonical_inputs["signal"].stream_id == "ecg"
    assert context.canonical_inputs["sampling_rate"].raw_key == "ecg_sampling_rate"
    assert context.alias_resolution["sampling_rate"] == "ecg_sampling_rate"
    assert context.canonical_inputs["time"].raw_key == "time"


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
    assert summary["duration_seconds"] == 8.03125
    assert summary["density_per_minute"] > 0.0
    assert summary["interval_median_samples"] == 128.0
    assert summary["interval_median_seconds"] == 1.0
    assert summary["outlier_fraction"] > 0.0


def test_summarize_series_emits_compact_distribution_metrics() -> None:
    summary = summarize_series(np.array([10.0, 12.0, 14.0, 16.0]))

    assert summary["count"] == 4.0
    assert summary["mean"] == 13.0
    assert summary["max"] == 16.0


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


def test_serialize_runtime_context_avoids_raw_arrays() -> None:
    context = resolve_canonical_runtime_context(
        {
            "h10_ecg_value": np.array([0.0, 1.0, 0.0]),
            "h10_ecg_t": np.array([0.0, 0.01, 0.02]),
            "ecg_sampling_rate": 128.0,
        }
    )

    payload = serialize_runtime_context(context)

    assert payload["streams"][0]["signal_key"] == "h10_ecg_value"
    assert "signal_values" not in payload["streams"][0]
    assert payload["canonical_inputs"]["signal"]["raw_key"] == "h10_ecg_value"


def test_summarize_runtime_evidence_emits_canonical_contract() -> None:
    runtime_inputs = {
        "capnostream_value": np.linspace(0.0, 1.0, 30),
        "capnostream_sampling_rate": 21.0,
        "h10_ecg_value": np.sin(np.linspace(0.0, 20.0, 2000)),
        "ecg_sampling_rate": 100.0,
    }
    evidence = summarize_runtime_evidence(
        runtime_inputs,
        intermediates={"events": np.array([100.0, 350.0, 600.0])},
        outputs={"rate": np.array([70.0, 71.0, 69.5])},
    )

    assert evidence["runtime_context"]["canonical_inputs"]["signal"] == "h10_ecg_value"
    assert (
        evidence["canonical_runtime_context"]["canonical_inputs"]["sampling_rate"]["stream_id"]
        == "ecg"
    )
    assert evidence["runtime_inputs"]["signal"].shape == runtime_inputs["h10_ecg_value"].shape
    assert evidence["signal_data"]["signal"].shape == runtime_inputs["h10_ecg_value"].shape
    assert evidence["runtime_inputs"]["sampling_rate"] == 100.0
    assert evidence["signal_data"]["sampling_rate"] == 100.0
    assert evidence["telemetry_summary"]["streams"]["ecg"]["signal"]["sampling_rate"] == 100.0
    assert evidence["telemetry_summary"]["events"]["count"] == 3.0
    assert evidence["telemetry_summary"]["outputs"]["rate"]["mean"] > 0.0


def test_summarize_runtime_evidence_preserves_generic_runtime_inputs() -> None:
    runtime_inputs = {
        "sensor_value": np.array([0.5, 0.75, 0.9]),
        "sensor_sampling_rate": 50.0,
    }
    evidence = summarize_runtime_evidence(
        runtime_inputs=runtime_inputs,
        intermediates={"score": np.array([1.0, 2.0, 3.0])},
        outputs={"quality": np.array([0.2, 0.4, 0.6])},
    )

    assert evidence["runtime_inputs"]["sensor_value"].shape == (3,)
    assert evidence["signal_data"]["sensor_sampling_rate"] == 50.0
    assert evidence["runtime_context"]["stream_count"] >= 1
    assert evidence["telemetry_summary"]["outputs"]["quality"]["mean"] == pytest.approx(0.4)


def test_summarize_runtime_evidence_derives_canonical_output_aliases() -> None:
    evidence = summarize_runtime_evidence(
        {
            "h10_ecg_value": np.sin(np.linspace(0.0, 20.0, 2000)),
            "ecg_sampling_rate": 100.0,
        },
        intermediates={"rpeaks": np.array([100.0, 350.0, 600.0])},
        outputs={"heart_rate": np.array([70.0, 71.0, 69.5])},
    )

    assert evidence["runtime_context"]["canonical_intermediates"] == {"events": "rpeaks"}
    assert evidence["telemetry_summary"]["events"]["source_key"] == "rpeaks"
    assert evidence["telemetry_summary"]["outputs"]["heart_rate"]["mean"] > 0.0
