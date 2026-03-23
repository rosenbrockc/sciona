"""Runtime atoms for Signal Transform expansion rules.

Provides deterministic, pure functions for spectral analysis
quality diagnostics:

  - Window leakage analysis (spectral leakage from windowing)
  - Spectral aliasing detection (frequency-domain aliasing)
  - Parseval energy validation (energy conservation across transforms)
  - Inverse reconstruction quality check (round-trip fidelity)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Window leakage analysis
# ---------------------------------------------------------------------------


def analyze_window_leakage(
    windowed: np.ndarray,
    original: np.ndarray,
) -> tuple[float, bool]:
    """Analyze spectral leakage introduced by the window function.

    Compares windowed and original signal energy to estimate how much
    energy is redistributed by the window.  High leakage ratios indicate
    the window attenuates too much of the signal.

    Args:
        windowed: windowed signal array.
        original: original signal before windowing.

    Returns:
        (leakage_ratio, is_excessive) where leakage_ratio is
        1 - (windowed_energy / original_energy) and is_excessive is
        True if leakage_ratio > 0.5.
    """
    w = np.asarray(windowed, dtype=np.float64).ravel()
    o = np.asarray(original, dtype=np.float64).ravel()

    orig_energy = float(np.sum(o ** 2))
    if orig_energy == 0:
        return 0.0, False

    win_energy = float(np.sum(w ** 2))
    ratio = 1.0 - (win_energy / orig_energy)
    ratio = max(0.0, min(1.0, ratio))
    return ratio, ratio > 0.5


# ---------------------------------------------------------------------------
# Spectral aliasing detection
# ---------------------------------------------------------------------------


def detect_spectral_aliasing(
    spectrum: np.ndarray,
    nyquist_fraction: float = 0.9,
) -> tuple[float, bool]:
    """Detect potential aliasing by checking energy near Nyquist.

    If a significant fraction of spectral energy lies above
    ``nyquist_fraction`` of the Nyquist frequency, the signal may
    be aliased.

    Args:
        spectrum: 1-D complex or real spectrum from forward transform.
        nyquist_fraction: fraction of Nyquist above which energy is
            considered alias-prone (default 0.9).

    Returns:
        (alias_energy_fraction, has_aliasing) where alias_energy_fraction
        is the fraction of total spectral energy near Nyquist.
    """
    s = np.asarray(spectrum, dtype=np.complex128).ravel()
    n = len(s)
    if n < 2:
        return 0.0, False

    power = np.abs(s) ** 2
    total = float(np.sum(power))
    if total == 0:
        return 0.0, False

    cutoff = int(n * nyquist_fraction)
    alias_energy = float(np.sum(power[cutoff:]))
    fraction = alias_energy / total
    return fraction, fraction > 0.1


# ---------------------------------------------------------------------------
# Parseval energy validation
# ---------------------------------------------------------------------------


def validate_parseval_energy(
    time_domain: np.ndarray,
    freq_domain: np.ndarray,
) -> tuple[float, bool]:
    """Validate energy conservation between time and frequency domains.

    Parseval's theorem states that total energy is preserved across
    the DFT.  Significant deviation indicates implementation errors
    or numeric issues.

    Args:
        time_domain: time-domain signal.
        freq_domain: frequency-domain spectrum.

    Returns:
        (relative_error, is_valid) where relative_error is
        |E_time - E_freq| / max(E_time, E_freq) and is_valid is True
        if relative_error < 1e-6.
    """
    t = np.asarray(time_domain, dtype=np.float64).ravel()
    f = np.asarray(freq_domain, dtype=np.complex128).ravel()

    e_time = float(np.sum(t ** 2))
    e_freq = float(np.sum(np.abs(f) ** 2)) / len(f) if len(f) > 0 else 0.0

    denom = max(e_time, e_freq)
    if denom == 0:
        return 0.0, True

    err = abs(e_time - e_freq) / denom
    return err, err < 1e-6


# ---------------------------------------------------------------------------
# Inverse reconstruction quality
# ---------------------------------------------------------------------------


def check_inverse_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[float, bool]:
    """Check round-trip reconstruction quality of forward+inverse transform.

    Args:
        original: original time-domain signal.
        reconstructed: signal after forward → inverse transform.

    Returns:
        (relative_error, is_faithful) where relative_error is
        ||original - reconstructed|| / ||original|| and is_faithful
        is True if relative_error < 1e-10.
    """
    o = np.asarray(original, dtype=np.float64).ravel()
    r = np.asarray(reconstructed, dtype=np.float64).ravel()

    orig_norm = float(np.linalg.norm(o))
    if orig_norm == 0:
        return 0.0, True

    err = float(np.linalg.norm(o - r)) / orig_norm
    return err, err < 1e-10
