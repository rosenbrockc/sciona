from __future__ import annotations

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from biosppy.signals.ecg import ASI_segmenter
from biosppy.signals.ecg import christov_segmenter as _christov_segmenter
from biosppy.signals.ecg import engzee_segmenter as _engzee_segmenter
from biosppy.signals.ecg import gamboa_segmenter as _gamboa_segmenter
from biosppy.signals.ecg import hamilton_segmenter as _hamilton_segmenter

from .witnesses import (
    witness_asi_signal_segmenter,
    witness_christov_qrs_segmenter,
    witness_christovqrsdetect,
    witness_engzee_qrs_segmentation,
    witness_engzee_signal_segmentation,
    witness_gamboa_segmenter,
    witness_gamboa_segmentation,
    witness_hamilton_segmenter,
    witness_hamilton_segmentation,
    witness_thresholdbasedsignalsegmentation,
)


@register_atom(witness_thresholdbasedsignalsegmentation)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda Pth: isinstance(Pth, (float, int, np.number)), "Pth must be numeric")
@icontract.ensure(lambda result: result is not None, "ThresholdBasedSignalSegmentation output must not be None")
def thresholdbasedsignalsegmentation(
    signal: np.ndarray,
    sampling_rate: float = 1000.0,
    Pth: float = 5.0,
) -> np.ndarray:
    return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)["rpeaks"]


@register_atom(witness_asi_signal_segmenter)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda Pth: isinstance(Pth, (float, int, np.number)), "Pth must be numeric")
@icontract.ensure(lambda result: result is not None, "ASI_signal_segmenter output must not be None")
def asi_signal_segmenter(signal: np.ndarray, sampling_rate: float, Pth: float) -> np.ndarray:
    return ASI_segmenter(signal=signal, sampling_rate=sampling_rate, Pth=Pth)["rpeaks"]


@register_atom(witness_christovqrsdetect)
@icontract.require(lambda signal: np.isfinite(signal).all(), "signal must be finite")
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim == 1, "result must be 1-D")
def christovqrsdetect(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _christov_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]


@register_atom(witness_christov_qrs_segmenter)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "christov_qrs_segmenter output must not be None")
def christov_qrs_segmenter(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _christov_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]


@register_atom(witness_engzee_signal_segmentation)
@icontract.require(lambda signal: np.isfinite(signal).all(), "signal must be finite")
@icontract.require(lambda signal: signal.ndim == 1, "signal must be 1-D")
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be ndarray")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int)), "sampling_rate must be numeric")
@icontract.require(lambda sampling_rate: sampling_rate > 0, "sampling_rate must be positive")
@icontract.require(lambda threshold: isinstance(threshold, (float, int)), "threshold must be numeric")
@icontract.require(lambda threshold: 0.0 < threshold < 1.0, "threshold must be in (0, 1)")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "result must be ndarray")
@icontract.ensure(lambda result: result.ndim == 1, "result must be 1-D")
def engzee_signal_segmentation(signal: np.ndarray, sampling_rate: float, threshold: float) -> np.ndarray:
    return _engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)["rpeaks"]


@register_atom(witness_engzee_qrs_segmentation)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "engzee_qrs_segmentation output must not be None")
def engzee_qrs_segmentation(signal: np.ndarray, sampling_rate: float, threshold: float) -> np.ndarray:
    return _engzee_segmenter(signal=signal, sampling_rate=sampling_rate, threshold=threshold)["rpeaks"]


@register_atom(witness_gamboa_segmentation)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: result is not None, "gamboa_segmentation output must not be None")
def gamboa_segmentation(signal: np.ndarray, sampling_rate: float, tol: float) -> np.ndarray:
    return _gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)["rpeaks"]


@register_atom(witness_gamboa_segmenter)
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: result is not None, "gamboa_segmenter output must not be None")
def gamboa_segmenter(signal: np.ndarray, sampling_rate: float, tol: float) -> np.ndarray:
    return _gamboa_segmenter(signal=signal, sampling_rate=sampling_rate, tol=tol)["rpeaks"]


@register_atom(witness_hamilton_segmentation)
@icontract.require(lambda signal: isinstance(signal, np.ndarray), "signal must be a numpy array")
@icontract.ensure(lambda result: result is not None, "hamilton_segmentation output must not be None")
def hamilton_segmentation(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    return np.asarray(_hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"], dtype=int)


@register_atom(witness_hamilton_segmenter)  # type: ignore[untyped-decorator]
@icontract.require(lambda sampling_rate: isinstance(sampling_rate, (float, int, np.number)), "sampling_rate must be numeric")
@icontract.ensure(lambda result: result is not None, "hamilton_segmenter output must not be None")
def hamilton_segmenter(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)["rpeaks"]
