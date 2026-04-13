"""Ghost witnesses for the BioSPPy online filter v2 wrappers."""

from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractSignal


def witness_filterstateinit(
    b: AbstractArray,
    a: AbstractArray,
) -> tuple[tuple[AbstractArray, AbstractArray, AbstractArray], AbstractArray]:
    """Initialization preserves coefficient shapes and allocates filter state."""
    order = max(max(b.shape[0], a.shape[0]) - 1, 0)
    zi = AbstractArray(shape=(order,), dtype="float64")
    return (b, a, zi), zi


def witness_filterstep(
    signal: AbstractSignal,
    state: AbstractArray,
) -> tuple[tuple[AbstractSignal, AbstractArray], AbstractArray]:
    """Filtering preserves chunk shape and threads a same-rank state vector."""
    next_state = AbstractArray(shape=state.shape, dtype="float64")
    filtered = AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=signal.sampling_rate,
        domain=signal.domain,
        units=signal.units,
    )
    return (filtered, next_state), next_state
