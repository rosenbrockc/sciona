"""Registry for signal transform primitives and expansion atoms."""

from __future__ import annotations

SIGNAL_TRANSFORM_DECLARATIONS = {
    "analyze_window_leakage": (
        "sciona.expansion_atoms.runtime_signal_transform.analyze_window_leakage",
        "ndarray, ndarray -> tuple[float, bool]",
        "Analyze spectral leakage introduced by the window function.",
    ),
    "detect_spectral_aliasing": (
        "sciona.expansion_atoms.runtime_signal_transform.detect_spectral_aliasing",
        "ndarray, float -> tuple[float, bool]",
        "Detect potential aliasing by checking energy near Nyquist.",
    ),
    "validate_parseval_energy": (
        "sciona.expansion_atoms.runtime_signal_transform.validate_parseval_energy",
        "ndarray, ndarray -> tuple[float, bool]",
        "Validate energy conservation between time and frequency domains.",
    ),
    "check_inverse_reconstruction": (
        "sciona.expansion_atoms.runtime_signal_transform.check_inverse_reconstruction",
        "ndarray, ndarray -> tuple[float, bool]",
        "Check round-trip reconstruction quality of forward+inverse transform.",
    ),
}
