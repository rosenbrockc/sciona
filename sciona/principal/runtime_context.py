"""Canonical runtime context resolution and compact telemetry summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class CanonicalInputRef(BaseModel):
    """Reference from a canonical input name to one resolved raw key/stream."""

    canonical_name: str
    raw_key: str
    stream_id: str
    data_kind: str
    provenance: str = ""


class RuntimeStream(BaseModel):
    """One resolved runtime stream with stable identity and aliases."""

    stream_id: str
    data_kind: str
    signal_key: str = ""
    signal_values: Any = None
    time_key: str = ""
    time_values: Any = None
    sampling_rate_key: str = ""
    sampling_rate: float | None = None
    aliases: list[str] = Field(default_factory=list)
    provenance: str = ""


class CanonicalRuntimeContext(BaseModel):
    """Canonical multi-stream runtime context for refinement/admissibility."""

    streams: list[RuntimeStream] = Field(default_factory=list)
    canonical_inputs: dict[str, CanonicalInputRef] = Field(default_factory=dict)
    alias_resolution: dict[str, str] = Field(default_factory=dict)


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _infer_data_kind(key: str) -> str:
    lowered = key.lower()
    if any(token in lowered for token in ("event", "peak", "beat", "r_peak")):
        return "event_sequence"
    if any(token in lowered for token in ("rate", "hr", "bpm", "frequency")):
        return "rate_series"
    if any(token in lowered for token in ("signal", "wave", "ecg", "ppg", "eeg", "emg")):
        return "waveform"
    return "generic"


def _infer_stream_id(key: str) -> str:
    lowered = key.lower()
    for token in ("ecg", "ppg", "eeg", "emg", "capnostream"):
        if token in lowered:
            return token
    if lowered.endswith("_sampling_rate"):
        return lowered[: -len("_sampling_rate")]
    if lowered.endswith("_value"):
        return lowered[: -len("_value")]
    return lowered


def _signal_priority(stream: RuntimeStream) -> tuple[int, str]:
    preferred = {
        "ecg": 0,
        "ppg": 1,
        "eeg": 2,
        "emg": 3,
    }
    return (preferred.get(stream.stream_id, 99), stream.stream_id)


def resolve_canonical_runtime_context(signal_data: dict[str, Any]) -> CanonicalRuntimeContext:
    """Resolve dataset-specific aliases into a canonical multi-stream context."""
    if not isinstance(signal_data, dict):
        return CanonicalRuntimeContext()

    streams: dict[str, RuntimeStream] = {}

    def ensure_stream(stream_id: str, *, data_kind: str) -> RuntimeStream:
        stream = streams.get(stream_id)
        if stream is None:
            stream = RuntimeStream(stream_id=stream_id, data_kind=data_kind)
            streams[stream_id] = stream
        elif stream.data_kind == "generic" and data_kind != "generic":
            stream.data_kind = data_kind
        return stream

    for raw_key, raw_value in signal_data.items():
        if raw_key in {"signal", "sampling_rate", "time"}:
            continue
        key = str(raw_key)
        lowered = key.lower()
        data_kind = _infer_data_kind(lowered)
        stream_id = _infer_stream_id(lowered)
        stream = ensure_stream(stream_id, data_kind=data_kind)
        if key not in stream.aliases:
            stream.aliases.append(key)

        if lowered.endswith("_sampling_rate"):
            try:
                stream.sampling_rate = float(raw_value)
                stream.sampling_rate_key = key
            except Exception:
                pass
            continue
        if lowered.endswith("_t") or lowered == "t" or lowered.endswith("_time"):
            stream.time_key = key
            stream.time_values = raw_value
            continue
        if stream.signal_key:
            continue
        stream.signal_key = key
        stream.signal_values = raw_value

    if not streams:
        generic = RuntimeStream(
            stream_id="generic",
            data_kind=_infer_data_kind("signal"),
            signal_key=str("signal") if "signal" in signal_data else "",
            signal_values=signal_data.get("signal"),
            time_key=str("time") if "time" in signal_data else "",
            time_values=signal_data.get("time"),
            sampling_rate_key=str("sampling_rate") if "sampling_rate" in signal_data else "",
            sampling_rate=(
                float(signal_data["sampling_rate"])
                if "sampling_rate" in signal_data
                else None
            ),
            aliases=[key for key in ("signal", "time", "sampling_rate") if key in signal_data],
            provenance="raw_signal_data",
        )
        streams[generic.stream_id] = generic

    stream_list = sorted(streams.values(), key=_signal_priority)

    primary = next(
        (stream for stream in stream_list if stream.signal_key and stream.data_kind == "waveform"),
        next((stream for stream in stream_list if stream.signal_key), None),
    )

    canonical_inputs: dict[str, CanonicalInputRef] = {}
    alias_resolution: dict[str, str] = {}
    if primary is not None and primary.signal_key:
        canonical_inputs["signal"] = CanonicalInputRef(
            canonical_name="signal",
            raw_key=primary.signal_key,
            stream_id=primary.stream_id,
            data_kind=primary.data_kind,
            provenance="preferred_waveform_stream",
        )
        alias_resolution["signal"] = primary.signal_key
    if primary is not None and primary.sampling_rate_key and primary.sampling_rate is not None:
        canonical_inputs["sampling_rate"] = CanonicalInputRef(
            canonical_name="sampling_rate",
            raw_key=primary.sampling_rate_key,
            stream_id=primary.stream_id,
            data_kind="sampling_context",
            provenance="stream_sampling_rate",
        )
        alias_resolution["sampling_rate"] = primary.sampling_rate_key
    if primary is not None and primary.time_key:
        canonical_inputs["time"] = CanonicalInputRef(
            canonical_name="time",
            raw_key=primary.time_key,
            stream_id=primary.stream_id,
            data_kind="time_axis",
            provenance="stream_time_axis",
        )
        alias_resolution["time"] = primary.time_key

    for stream in stream_list:
        if stream.provenance:
            continue
        if stream.stream_id == (primary.stream_id if primary is not None else ""):
            stream.provenance = "primary_stream"
        else:
            stream.provenance = "auxiliary_stream"

    return CanonicalRuntimeContext(
        streams=stream_list,
        canonical_inputs=canonical_inputs,
        alias_resolution=alias_resolution,
    )


def summarize_waveform(values: Any) -> dict[str, float]:
    """Emit compact waveform-oriented summary metrics."""
    arr = _as_array(values)
    if arr.size == 0:
        return {"count": 0.0}
    diff = np.diff(arr) if arr.size > 1 else np.array([], dtype=np.float64)
    if diff.size:
        median_diff = float(np.median(diff))
        mad_diff = float(np.median(np.abs(diff - median_diff)))
        discontinuity_count = float(
            np.sum(np.abs(diff - median_diff) > 5.0 * max(mad_diff, 1e-9))
        )
    else:
        discontinuity_count = 0.0
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "max_abs": float(np.max(np.abs(arr))),
        "discontinuity_count": discontinuity_count,
    }


def summarize_events(
    events: Any,
    *,
    sampling_rate: float | None = None,
    duration_seconds: float | None = None,
) -> dict[str, float]:
    """Emit compact event-stream summary metrics."""
    arr = np.sort(_as_array(events))
    if arr.size == 0:
        return {"count": 0.0}

    intervals = np.diff(arr)
    intervals = intervals[intervals > 0]
    duration = duration_seconds
    if duration is None and sampling_rate is not None and arr.size > 1:
        duration = float(arr[-1] - arr[0]) / float(sampling_rate)
    density = (
        float(arr.size) / max(duration / 60.0, 1e-9)
        if duration is not None and duration > 0
        else 0.0
    )
    if intervals.size:
        interval_median = float(np.median(intervals))
        interval_mad = float(np.median(np.abs(intervals - interval_median)))
        lo = interval_median - 3.0 * interval_mad
        hi = interval_median + 3.0 * interval_mad
        outlier_fraction = float(np.mean((intervals < lo) | (intervals > hi)))
    else:
        interval_median = 0.0
        interval_mad = 0.0
        outlier_fraction = 0.0
    summary = {
        "count": float(arr.size),
        "duration_seconds": float(duration) if duration is not None else 0.0,
        "density_per_minute": density,
        "interval_median_samples": interval_median,
        "interval_mad_samples": interval_mad,
        "outlier_fraction": outlier_fraction,
    }
    if sampling_rate is not None and sampling_rate > 0 and interval_median > 0:
        summary["interval_median_seconds"] = interval_median / float(sampling_rate)
    return summary


def summarize_runtime_context(context: CanonicalRuntimeContext) -> dict[str, Any]:
    """Emit a compact stable summary for the resolved runtime context."""
    return {
        "stream_count": len(context.streams),
        "canonical_inputs": {
            name: ref.raw_key for name, ref in sorted(context.canonical_inputs.items())
        },
        "stream_ids": [stream.stream_id for stream in context.streams],
        "sampling_rates": {
            stream.stream_id: stream.sampling_rate
            for stream in context.streams
            if stream.sampling_rate is not None
        },
    }
