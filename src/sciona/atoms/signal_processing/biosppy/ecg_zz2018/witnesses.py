from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_calculatecompositesqi_zz2018(
    signal: AbstractSignal,
    detector_1: AbstractSignal,
    detector_2: AbstractSignal,
    fs: AbstractScalar,
    search_window: AbstractScalar,
    nseg: AbstractScalar,
    mode: AbstractScalar,
) -> AbstractArray:
    return AbstractArray(shape=signal.shape, dtype="float64")


def witness_calculatebeatagreementsqi(
    detector_1: AbstractSignal,
    detector_2: AbstractSignal,
    fs: AbstractScalar,
    mode: AbstractScalar,
    search_window: AbstractScalar,
) -> AbstractSignal:
    return AbstractSignal(
        shape=detector_1.shape,
        dtype="float64",
        sampling_rate=getattr(detector_1, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_calculatefrequencypowersqi(
    ecg_signal: AbstractSignal,
    fs: AbstractScalar,
    nseg: AbstractScalar,
    num_spectrum: AbstractSignal,
    dem_spectrum: AbstractSignal,
    mode: AbstractScalar,
) -> AbstractSignal:
    return AbstractSignal(
        shape=ecg_signal.shape,
        dtype="float64",
        sampling_rate=getattr(ecg_signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_calculatekurtosissqi(signal: AbstractSignal, fisher: AbstractScalar) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )
