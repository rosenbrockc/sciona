"""Synthetic grouped helpers modeled on the tempo_jl/offsets lesson."""

from __future__ import annotations

import math


def offset_tt2tdb(seconds: float) -> float:
    phase = 6.239996 + 1.99096871e-7 * float(seconds)
    return 1.657e-3 * math.sin(phase)


def offset_tt2tdbh(seconds: float) -> float:
    centuries = float(seconds) / (86400.0 * 36525.0)
    return offset_tt2tdb(seconds) + 2.2e-5 * math.sin(575.3385 * centuries + 4.2970)


def tt2tdb_offset(seconds):
    if isinstance(seconds, (list, tuple)):
        return [offset_tt2tdb(value) for value in seconds]
    return offset_tt2tdb(seconds)
