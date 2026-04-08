"""Helpers for applying relative dataset slices to adapter-backed collections."""

from __future__ import annotations

from typing import Any


def _coerce_numeric(value: Any) -> float | None:
    """Best-effort conversion of a collection time boundary to float seconds."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced >= 0.0 else None


def _ensure_collection_loaded(collection: Any) -> Any | None:
    """Load collection data once so relative slices can anchor to its start time."""
    data = getattr(collection, "data", None)
    if data is not None:
        return data
    autoload = getattr(collection, "autoload", None)
    if callable(autoload):
        data = autoload()
        if data is not None:
            return data
    load_all = getattr(collection, "load_all", None)
    if callable(load_all):
        load_all()
        return getattr(collection, "data", None)
    return getattr(collection, "data", None)


def apply_relative_dataset_slice(
    collection: Any,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> None:
    """Apply a relative `(start, stop)` window anchored to collection start time."""
    if start_s is None and stop_s is None:
        return
    slicer = getattr(collection, "slice", None)
    if not callable(slicer):
        return

    data = _ensure_collection_loaded(collection)
    anchor = _coerce_numeric(getattr(data, "min", None))
    if anchor is None:
        slicer(start_s, stop_s)
        return

    absolute_start = (
        anchor + float(start_s)
        if start_s is not None
        else anchor
    )
    absolute_stop = anchor + float(stop_s) if stop_s is not None else None
    slicer(absolute_start, absolute_stop)
