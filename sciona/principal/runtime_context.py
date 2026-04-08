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


def _data_kind_priority(kind: str) -> int:
    priorities = {
        "generic": 0,
        "rate_series": 1,
        "event_sequence": 2,
        "waveform": 3,
    }
    return priorities.get(kind, 0)


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


def _choose_primary_stream(streams: list[RuntimeStream]) -> RuntimeStream | None:
    canonical = [
        stream
        for stream in streams
        if stream.signal_key == "signal" or "signal" in stream.aliases
    ]
    if canonical:
        return sorted(canonical, key=_signal_priority)[0]
    waveform = [stream for stream in streams if stream.signal_key and stream.data_kind == "waveform"]
    if waveform:
        return sorted(waveform, key=_signal_priority)[0]
    signaled = [stream for stream in streams if stream.signal_key]
    if signaled:
        return sorted(signaled, key=_signal_priority)[0]
    return None


def _bind_canonical_aliases(
    runtime_inputs: dict[str, Any],
    streams: dict[str, RuntimeStream],
) -> None:
    """Attach generic canonical aliases to a stable stream when possible."""
    stream_list = sorted(streams.values(), key=_signal_priority)
    if not stream_list and any(key in runtime_inputs for key in ("signal", "time", "sampling_rate")):
        streams["primary"] = RuntimeStream(stream_id="primary", data_kind=_infer_data_kind("signal"))
        stream_list = [streams["primary"]]
    if not stream_list:
        return

    waveform_bindable = [stream for stream in stream_list if stream.data_kind == "waveform"]
    if waveform_bindable:
        target = sorted(waveform_bindable, key=_signal_priority)[0]
    elif len(stream_list) == 1 and stream_list[0].signal_key in {"", "signal"}:
        target = stream_list[0]
    elif any(key in runtime_inputs for key in ("signal", "time", "sampling_rate")):
        target = streams.get("primary")
        if target is None:
            target = RuntimeStream(stream_id="primary", data_kind=_infer_data_kind("signal"))
            streams[target.stream_id] = target
    else:
        target = None

    if target is None:
        return
    if "signal" in runtime_inputs and not target.signal_key:
        target.signal_key = "signal"
        target.signal_values = runtime_inputs.get("signal")
        if "signal" not in target.aliases:
            target.aliases.append("signal")
    if "time" in runtime_inputs and not target.time_key:
        target.time_key = "time"
        target.time_values = runtime_inputs.get("time")
        if "time" not in target.aliases:
            target.aliases.append("time")
    if "sampling_rate" in runtime_inputs and target.sampling_rate is None:
        try:
            target.sampling_rate = float(runtime_inputs["sampling_rate"])
            target.sampling_rate_key = "sampling_rate"
            if "sampling_rate" not in target.aliases:
                target.aliases.append("sampling_rate")
        except Exception:
            pass


def _coerce_runtime_inputs(
    runtime_inputs: dict[str, Any] | None = None,
    *,
    signal_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the family-neutral runtime input contract with legacy aliases."""
    merged: dict[str, Any] = {}
    if isinstance(signal_data, dict):
        merged.update(signal_data)
    if isinstance(runtime_inputs, dict):
        merged.update(runtime_inputs)
    return merged


def resolve_canonical_runtime_context(
    runtime_inputs: dict[str, Any] | None = None,
    *,
    signal_data: dict[str, Any] | None = None,
) -> CanonicalRuntimeContext:
    """Resolve runtime aliases into a canonical multi-stream context."""
    runtime_inputs = _coerce_runtime_inputs(runtime_inputs, signal_data=signal_data)
    if not isinstance(runtime_inputs, dict):
        return CanonicalRuntimeContext()

    streams: dict[str, RuntimeStream] = {}

    def ensure_stream(stream_id: str, *, data_kind: str) -> RuntimeStream:
        stream = streams.get(stream_id)
        if stream is None:
            stream = RuntimeStream(stream_id=stream_id, data_kind=data_kind)
            streams[stream_id] = stream
        elif _data_kind_priority(data_kind) > _data_kind_priority(stream.data_kind):
            stream.data_kind = data_kind
        return stream

    for raw_key, raw_value in runtime_inputs.items():
        if raw_key in {"signal", "sampling_rate", "time"}:
            continue
        if hasattr(raw_value, "columns"):
            continue
        key = str(raw_key)
        lowered = key.lower()
        stream_id = _infer_stream_id(lowered)

        if lowered.endswith("_sampling_rate"):
            stream = ensure_stream(stream_id, data_kind="generic")
            if key not in stream.aliases:
                stream.aliases.append(key)
            try:
                stream.sampling_rate = float(raw_value)
                stream.sampling_rate_key = key
            except Exception:
                pass
            continue
        if lowered.endswith("_t") or lowered == "t" or lowered.endswith("_time"):
            stream = ensure_stream(stream_id, data_kind="generic")
            if key not in stream.aliases:
                stream.aliases.append(key)
            stream.time_key = key
            stream.time_values = raw_value
            continue
        data_kind = _infer_data_kind(lowered)
        stream = ensure_stream(stream_id, data_kind=data_kind)
        if key not in stream.aliases:
            stream.aliases.append(key)
        if stream.signal_key:
            continue
        stream.signal_key = key
        stream.signal_values = raw_value

    _bind_canonical_aliases(runtime_inputs, streams)

    if not streams:
        generic = RuntimeStream(
            stream_id="generic",
            data_kind=_infer_data_kind("signal"),
            signal_key=str("signal") if "signal" in runtime_inputs else "",
            signal_values=runtime_inputs.get("signal"),
            time_key=str("time") if "time" in runtime_inputs else "",
            time_values=runtime_inputs.get("time"),
            sampling_rate_key=str("sampling_rate")
            if "sampling_rate" in runtime_inputs
            else "",
            sampling_rate=(
                float(runtime_inputs["sampling_rate"])
                if "sampling_rate" in runtime_inputs
                else None
            ),
            aliases=[
                key for key in ("signal", "time", "sampling_rate") if key in runtime_inputs
            ],
            provenance="raw_runtime_inputs",
        )
        streams[generic.stream_id] = generic

    stream_list = sorted(streams.values(), key=_signal_priority)

    primary = _choose_primary_stream(stream_list)

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


def summarize_time_axis(values: Any) -> dict[str, float]:
    """Emit compact time-axis metrics for canonical runtime evidence."""
    arr = _as_array(values)
    if arr.size == 0:
        return {"count": 0.0}
    start = float(arr[0])
    end = float(arr[-1])
    duration = max(end - start, 0.0) if arr.size > 1 else 0.0
    return {
        "count": float(arr.size),
        "start": start,
        "end": end,
        "duration_seconds": duration,
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


def summarize_series(values: Any) -> dict[str, float]:
    """Emit compact summary metrics for generic numeric series."""
    arr = _as_array(values)
    if arr.size == 0:
        return {"count": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(np.quantile(arr, 0.01)),
        "p50": float(np.quantile(arr, 0.50)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_named_value(
    name: str,
    values: Any,
    *,
    sampling_rate: float | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Summarize one named runtime value using cross-family data-kind inference."""
    kind = _infer_data_kind(name)
    if kind == "event_sequence":
        return summarize_events(
            values,
            sampling_rate=sampling_rate,
            duration_seconds=duration_seconds,
        )
    if kind == "waveform":
        return summarize_waveform(values)
    return summarize_series(values)


def summarize_runtime_context(context: CanonicalRuntimeContext) -> dict[str, Any]:
    """Emit a compact stable summary for the resolved runtime context."""
    primary_signal = context.canonical_inputs.get("signal")
    return {
        "stream_count": len(context.streams),
        "primary_stream_id": primary_signal.stream_id if primary_signal is not None else "",
        "canonical_inputs": {
            name: ref.raw_key for name, ref in sorted(context.canonical_inputs.items())
        },
        "canonical_streams": {
            name: ref.stream_id for name, ref in sorted(context.canonical_inputs.items())
        },
        "alias_resolution": dict(context.alias_resolution),
        "stream_ids": [stream.stream_id for stream in context.streams],
        "sampling_rates": {
            stream.stream_id: stream.sampling_rate
            for stream in context.streams
            if stream.sampling_rate is not None
        },
        "streams": [
            {
                "stream_id": stream.stream_id,
                "data_kind": stream.data_kind,
                "signal_key": stream.signal_key,
                "time_key": stream.time_key,
                "sampling_rate_key": stream.sampling_rate_key,
                "sampling_rate": stream.sampling_rate,
                "aliases": list(stream.aliases),
                "provenance": stream.provenance,
            }
            for stream in context.streams
        ],
    }


def canonicalize_runtime_inputs(
    runtime_inputs: dict[str, Any] | None = None,
    *,
    signal_data: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], CanonicalRuntimeContext]:
    """Resolve deterministic canonical aliases onto runtime inputs."""
    normalized = _coerce_runtime_inputs(runtime_inputs, signal_data=signal_data)
    context = resolve_canonical_runtime_context(normalized)
    for canonical_name, ref in context.canonical_inputs.items():
        if ref.raw_key in normalized:
            normalized[canonical_name] = normalized[ref.raw_key]
    return normalized, context


def canonicalize_signal_data(
    signal_data: dict[str, Any] | None,
) -> tuple[dict[str, Any], CanonicalRuntimeContext]:
    """Backward-compatible alias for :func:`canonicalize_runtime_inputs`."""
    return canonicalize_runtime_inputs(signal_data=signal_data)


def _intermediate_role_priority(key: str) -> tuple[str, int] | None:
    lowered = key.lower()
    if lowered == "events":
        return ("events", 0)
    if any(token in lowered for token in ("rpeak", "r_peak", "peaks", "peak", "beats", "beat", "events", "event")):
        return ("events", 1)
    if lowered == "rate":
        return ("rate", 0)
    if (
        "heart_rate" in lowered
        or lowered.endswith("_rate")
        or lowered.endswith("_bpm")
        or lowered == "bpm"
        or lowered == "hr"
        or lowered.startswith("hr_")
        or lowered.endswith("_hr")
    ):
        return ("rate", 1)
    return None


def canonicalize_intermediates(
    intermediates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve deterministic canonical aliases onto runtime intermediates."""
    normalized = dict(intermediates) if isinstance(intermediates, dict) else {}
    winners: dict[str, tuple[int, str]] = {}
    for raw_key in normalized:
        candidate = _intermediate_role_priority(str(raw_key))
        if candidate is None:
            continue
        canonical_name, priority = candidate
        current = winners.get(canonical_name)
        if current is None or priority < current[0]:
            winners[canonical_name] = (priority, str(raw_key))
    alias_resolution = {
        canonical_name: raw_key for canonical_name, (_, raw_key) in winners.items()
    }
    for canonical_name, raw_key in alias_resolution.items():
        normalized[canonical_name] = normalized[raw_key]
    return normalized, alias_resolution


def _duration_seconds_for_signal(
    *,
    signal_values: Any,
    time_values: Any,
    sampling_rate: float | None,
) -> float | None:
    if time_values is not None:
        summary = summarize_time_axis(time_values)
        duration = float(summary.get("duration_seconds", 0.0))
        if duration > 0:
            return duration
    if sampling_rate is not None and sampling_rate > 0 and signal_values is not None:
        try:
            return float(len(signal_values)) / float(sampling_rate)
        except Exception:
            return None
    return None


def build_canonical_runtime_evidence(
    runtime_inputs: dict[str, Any] | None = None,
    intermediates: dict[str, Any] | None = None,
    *,
    signal_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical runtime context and compact telemetry summaries."""
    canonical_runtime_inputs, context = canonicalize_runtime_inputs(
        runtime_inputs,
        signal_data=signal_data,
    )
    canonical_intermediates, intermediate_aliases = canonicalize_intermediates(
        intermediates or {}
    )
    runtime_context = summarize_runtime_context(context)
    if intermediate_aliases:
        runtime_context["canonical_intermediates"] = dict(intermediate_aliases)

    telemetry: dict[str, Any] = {}
    signal_ref = context.canonical_inputs.get("signal")
    time_ref = context.canonical_inputs.get("time")
    sampling_ref = context.canonical_inputs.get("sampling_rate")

    sampling_rate: float | None = None
    if sampling_ref is not None and sampling_ref.raw_key in canonical_runtime_inputs:
        try:
            sampling_rate = float(canonical_runtime_inputs[sampling_ref.raw_key])
        except Exception:
            sampling_rate = None

    signal_values = (
        canonical_runtime_inputs.get(signal_ref.raw_key)
        if signal_ref is not None
        else None
    )
    time_values = (
        canonical_runtime_inputs.get(time_ref.raw_key)
        if time_ref is not None
        else None
    )
    duration_seconds = _duration_seconds_for_signal(
        signal_values=signal_values,
        time_values=time_values,
        sampling_rate=sampling_rate,
    )

    if signal_values is not None:
        telemetry["signal"] = summarize_waveform(signal_values)
        telemetry["signal"]["source_key"] = signal_ref.raw_key if signal_ref is not None else "signal"
        telemetry["signal"]["stream_id"] = signal_ref.stream_id if signal_ref is not None else ""
        if sampling_rate is not None:
            telemetry["signal"]["sampling_rate"] = sampling_rate
        if duration_seconds is not None:
            telemetry["signal"]["duration_seconds"] = duration_seconds

    if time_values is not None:
        telemetry["time"] = summarize_time_axis(time_values)
        telemetry["time"]["source_key"] = time_ref.raw_key if time_ref is not None else "time"
        telemetry["time"]["stream_id"] = time_ref.stream_id if time_ref is not None else ""

    events_key = intermediate_aliases.get("events")
    if events_key:
        telemetry["events"] = summarize_events(
            canonical_intermediates[events_key],
            sampling_rate=sampling_rate,
            duration_seconds=duration_seconds,
        )
        telemetry["events"]["source_key"] = events_key
        if signal_ref is not None:
            telemetry["events"]["stream_id"] = signal_ref.stream_id

    rate_key = intermediate_aliases.get("rate")
    if rate_key:
        telemetry["rate"] = summarize_series(canonical_intermediates[rate_key])
        telemetry["rate"]["source_key"] = rate_key

    return {
        "runtime_context": runtime_context,
        "telemetry": telemetry,
        "runtime_inputs": canonical_runtime_inputs,
        "signal_data": canonical_runtime_inputs,
        "intermediates": canonical_intermediates,
    }


def serialize_runtime_context(context: CanonicalRuntimeContext) -> dict[str, Any]:
    """Serialize the canonical runtime context without embedding raw arrays."""
    return {
        "primary_stream_id": (
            context.canonical_inputs["signal"].stream_id
            if "signal" in context.canonical_inputs
            else ""
        ),
        "streams": [
            {
                "stream_id": stream.stream_id,
                "data_kind": stream.data_kind,
                "signal_key": stream.signal_key,
                "time_key": stream.time_key,
                "sampling_rate_key": stream.sampling_rate_key,
                "sampling_rate": stream.sampling_rate,
                "aliases": list(stream.aliases),
                "provenance": stream.provenance,
            }
            for stream in context.streams
        ],
        "canonical_inputs": {
            name: {
                "raw_key": ref.raw_key,
                "stream_id": ref.stream_id,
                "data_kind": ref.data_kind,
                "provenance": ref.provenance,
            }
            for name, ref in sorted(context.canonical_inputs.items())
        },
        "alias_resolution": dict(sorted(context.alias_resolution.items())),
    }


def summarize_runtime_evidence(
    runtime_inputs: dict[str, Any] | None = None,
    *,
    signal_data: dict[str, Any] | None = None,
    intermediates: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical evidence contract used by refinement and profiling."""
    base = build_canonical_runtime_evidence(
        runtime_inputs,
        intermediates,
        signal_data=signal_data,
    )
    canonical_runtime_inputs = dict(base["runtime_inputs"])
    canonical_intermediates = dict(base["intermediates"])
    canonical_outputs, _ = canonicalize_intermediates(outputs or {})

    canonical = resolve_canonical_runtime_context(canonical_runtime_inputs)
    runtime_context = dict(base["runtime_context"])
    telemetry = dict(base["telemetry"])
    sampling_rate = telemetry.get("signal", {}).get("sampling_rate")
    signal_duration = telemetry.get("signal", {}).get("duration_seconds")
    streams: dict[str, Any] = {}
    for stream in canonical.streams:
        stream_summary: dict[str, Any] = {
            "data_kind": stream.data_kind,
            "provenance": stream.provenance,
            "aliases": list(stream.aliases),
        }
        if stream.sampling_rate is not None:
            stream_summary["sampling_rate"] = float(stream.sampling_rate)
        if stream.signal_key:
            values = canonical_runtime_inputs.get(stream.signal_key)
            if values is not None:
                signal_summary = summarize_waveform(values)
                if stream.sampling_rate is not None and stream.sampling_rate > 0:
                    signal_summary["sampling_rate"] = float(stream.sampling_rate)
                    signal_summary["duration_seconds"] = float(len(values)) / float(stream.sampling_rate)
                stream_summary["signal"] = signal_summary
        if stream.time_key:
            time_values = canonical_runtime_inputs.get(stream.time_key)
            if time_values is not None:
                stream_summary["time"] = summarize_time_axis(time_values)
        streams[stream.stream_id] = stream_summary

    def _summarize_named_values(values_by_name: dict[str, Any]) -> dict[str, Any]:
        summarized: dict[str, Any] = {}
        for name, values in sorted(values_by_name.items()):
            summarized[name] = summarize_named_value(
                name,
                values,
                sampling_rate=sampling_rate,
                duration_seconds=signal_duration,
            )
        return summarized

    return {
        "runtime_context": runtime_context,
        "canonical_runtime_context": serialize_runtime_context(canonical),
        "telemetry_summary": {
            "canonical": runtime_context,
            "streams": streams,
            "signal": telemetry.get("signal", {}),
            "time": telemetry.get("time", {}),
            "events": telemetry.get("events", {}),
            "rate": telemetry.get("rate", {}),
            "intermediates": _summarize_named_values(canonical_intermediates),
            "outputs": _summarize_named_values(canonical_outputs),
        },
        "runtime_inputs": canonical_runtime_inputs,
        "signal_data": canonical_runtime_inputs,
        "intermediates": canonical_intermediates,
    }
