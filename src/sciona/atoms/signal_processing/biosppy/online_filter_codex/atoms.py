"""Faithful wrappers around :class:`biosppy.signals.tools.OnlineFilter`."""

from __future__ import annotations

import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from biosppy.signals.tools import OnlineFilter

from .state_models import FilterState
from .witnesses import witness_filterstep, witness_filterstateinit


def _is_vector(array: np.ndarray) -> bool:
    return isinstance(array, np.ndarray) and array.ndim == 1 and array.size >= 1


def _as_numeric_vector(array: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(array, dtype=float)
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError(f"{name} must be a non-empty 1D numeric array")
    return vector


@register_atom(witness_filterstateinit)
@icontract.require(lambda b: b is None or _is_vector(b), "b must be None or a non-empty 1D numpy array")
@icontract.require(lambda a: a is None or _is_vector(a), "a must be None or a non-empty 1D numpy array")
@icontract.require(lambda a: a is None or float(a[0]) != 0.0, "a[0] must be non-zero")
@icontract.ensure(lambda result: result[0][2] is None, "initial zi must be None")
def filterstateinit(
    b: np.ndarray | None = None,
    a: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray | None], FilterState]:
    if b is None:
        raise TypeError("Please specify the numerator coefficients.")
    if a is None:
        raise TypeError("Please specify the denominator coefficients.")
    coeff_b = _as_numeric_vector(b, "b")
    coeff_a = _as_numeric_vector(a, "a")
    obj = OnlineFilter(b=coeff_b, a=coeff_a)
    state = FilterState(b=coeff_b, a=coeff_a, zi=obj.zi)
    return (state.b, state.a, state.zi), state


@register_atom(witness_filterstep)
@icontract.require(lambda signal: signal is None or _is_vector(signal), "signal must be None or a non-empty 1D numpy array")
@icontract.require(lambda state: state is not None, "state cannot be None")
@icontract.require(lambda state: state.b is not None and state.a is not None, "state must carry filter coefficients")
@icontract.ensure(lambda result, signal: result[0][0].shape == signal.shape, "filtered output shape must match input")
def filterstep(
    signal: np.ndarray | None = None,
    state: FilterState | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], FilterState]:
    if signal is None:
        raise TypeError("Please specify the input signal.")
    chunk = _as_numeric_vector(signal, "signal")
    obj = OnlineFilter(b=np.asarray(state.b, dtype=float), a=np.asarray(state.a, dtype=float))
    if state.zi is not None:
        obj.zi = np.asarray(state.zi, dtype=float).copy()
    result = obj.filter(signal=chunk)
    filtered = np.asarray(result["filtered"], dtype=float)
    next_state = FilterState(
        b=np.asarray(obj.b, dtype=float),
        a=np.asarray(obj.a, dtype=float),
        zi=np.asarray(obj.zi, dtype=float),
    )
    return (filtered, next_state.zi), next_state
