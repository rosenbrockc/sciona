"""BioSPPy ECG ZZ2018 atom wrappers for the signal-processing provider."""

from __future__ import annotations

import scipy.integrate as scipy_integrate
import numpy as np
import icontract

from sciona.ghost.registry import register_atom
from biosppy.signals.ecg import ZZ2018, bSQI, fSQI, kSQI

from .witnesses import (
    witness_calculatebeatagreementsqi,
    witness_calculatecompositesqi_zz2018,
    witness_calculatefrequencypowersqi,
    witness_calculatekurtosissqi,
)


def _ensure_scipy_trapz() -> None:
    """Compat shim for BioSPPy on SciPy versions without integrate.trapz."""
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    if not hasattr(scipy_integrate, "trapz"):
        scipy_integrate.trapz = np.trapz  # type: ignore[attr-defined]


@register_atom(witness_calculatecompositesqi_zz2018)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateCompositeSQI_ZZ2018 output must not be None")
def calculatecompositesqi_zz2018(
    signal: np.ndarray,
    detector_1: np.ndarray,
    detector_2: np.ndarray,
    fs: float,
    search_window: int,
    nseg: int,
    mode: str,
) -> float:
    """Calculate the composite ZZ2018 ECG quality score."""
    _ensure_scipy_trapz()
    return ZZ2018(
        signal=signal,
        detector_1=detector_1,
        detector_2=detector_2,
        fs=fs,
        search_window=search_window,
        nseg=nseg,
        mode=mode,
    )


@register_atom(witness_calculatebeatagreementsqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateBeatAgreementSQI output must not be None")
def calculatebeatagreementsqi(
    detector_1: np.ndarray,
    detector_2: np.ndarray,
    fs: float,
    mode: str,
    search_window: int,
) -> float:
    """Calculate the beat-agreement SQI."""
    return bSQI(detector_1=detector_1, detector_2=detector_2, fs=fs, mode=mode, search_window=search_window)


@register_atom(witness_calculatefrequencypowersqi)
@icontract.require(lambda fs: isinstance(fs, (float, int, np.number)), "fs must be numeric")
@icontract.ensure(lambda result: result is not None, "CalculateFrequencyPowerSQI output must not be None")
def calculatefrequencypowersqi(
    ecg_signal: np.ndarray,
    fs: float,
    nseg: int,
    num_spectrum: np.ndarray,
    dem_spectrum: np.ndarray,
    mode: str,
) -> float:
    """Calculate the frequency-power SQI."""
    _ensure_scipy_trapz()
    return fSQI(
        ecg_signal=ecg_signal,
        fs=fs,
        nseg=nseg,
        num_spectrum=num_spectrum,
        dem_spectrum=dem_spectrum,
        mode=mode,
    )


@register_atom(witness_calculatekurtosissqi)
@icontract.require(lambda signal: signal is not None, "signal cannot be None")
@icontract.require(lambda fisher: fisher is not None, "fisher cannot be None")
@icontract.ensure(lambda result: result is not None, "CalculateKurtosisSQI output must not be None")
def calculatekurtosissqi(signal: np.ndarray, fisher: bool) -> float:
    """Calculate the kurtosis SQI."""
    return kSQI(signal=signal, fisher=fisher)
