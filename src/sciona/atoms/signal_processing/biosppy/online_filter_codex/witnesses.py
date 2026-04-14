"""Ghost witnesses for the BioSPPy online filter wrappers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractSignal


def witness_filterstateinit(
    b: AbstractArray,
    a: AbstractArray,
) -> tuple[tuple[AbstractArray, AbstractArray, AbstractArray], AbstractArray]:
    order = max(max(b.shape[0], a.shape[0]) - 1, 0)
    zi = AbstractArray(shape=(order,), dtype="float64")
    return (b, a, zi), zi


def witness_filterstep(
    signal: AbstractSignal,
    state: AbstractArray,
) -> tuple[tuple[AbstractSignal, AbstractArray], AbstractArray]:
    next_state = AbstractArray(shape=state.shape, dtype="float64")
    filtered = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
        units=signal.units,
    )
    return (filtered, next_state), next_state
