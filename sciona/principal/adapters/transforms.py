"""Repo-local transforms used by principal adapter datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sciona.principal.datasets.core import msg

log = logging.getLogger(__name__)


def shift_time_explicit(
    t: np.ndarray,
    fixed: float,
    filename: str | None = None,
    before: bool = True,
    scaler: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Shift a time vector by an explicit timestamp offset in seconds."""
    del filename, kwargs
    if t is None or len(t) == 0:
        msg.warn(
            "Time vector in 'sciona.principal.adapters.transforms.shift_time_explicit' was empty."
        )
        return t

    if before:
        return t * scaler + fixed
    return (t - t[-1]) * scaler + fixed


def shift_time_meta_attr(
    t: np.ndarray,
    attr: str,
    meta: object | dict[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Shift a time vector using a metadata attribute expressed in seconds."""
    del kwargs
    if meta is None:
        return t

    if isinstance(meta, dict):
        shift = meta[attr]
    else:
        shift = getattr(meta, attr)

    if isinstance(shift, str):
        try:
            shift = float(shift)
        except ValueError:
            msg.err(f"Could not convert {shift!r} to float in shift_time_meta_attr")
            return t

    if isinstance(shift, (int, float)):
        return shift_time_explicit(t, float(shift))

    log.warning("Unsupported shift type for %s: %r", attr, type(shift))
    return t
