from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_thresholdbasedsignalsegmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    Pth: AbstractScalar,
) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_asi_signal_segmenter(
    signal: AbstractSignal,
    sampling_rate: AbstractSignal,
    Pth: AbstractSignal,
) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_christovqrsdetect(signal: AbstractSignal, sampling_rate: AbstractScalar) -> AbstractArray:
    return AbstractArray(shape=(signal.shape[0],), dtype="int64", min_val=0, max_val=signal.shape[0] - 1)


def witness_christov_qrs_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_engzee_signal_segmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    threshold: AbstractScalar,
) -> AbstractArray:
    return AbstractArray(shape=(signal.shape[0],), dtype="int64", min_val=0, max_val=signal.shape[0] - 1)


def witness_engzee_qrs_segmentation(signal: AbstractSignal, sampling_rate: AbstractSignal, threshold: AbstractSignal) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_gamboa_segmentation(
    signal: AbstractSignal,
    sampling_rate: AbstractScalar,
    tol: AbstractScalar,
) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_gamboa_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal, tol: AbstractSignal) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_hamilton_segmentation(signal: AbstractSignal, sampling_rate: AbstractScalar) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )


def witness_hamilton_segmenter(signal: AbstractSignal, sampling_rate: AbstractSignal) -> AbstractSignal:
    return AbstractSignal(
        shape=signal.shape,
        dtype="float64",
        sampling_rate=getattr(signal, "sampling_rate_prime", 44100.0),
        domain="time",
    )
