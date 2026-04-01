"""Runtime atoms for Baseline Analysis expansion rules.

Provides deterministic, pure functions for baseline analysis pipeline
diagnostics:

  - Onset coverage analysis (onset density relative to signal length)
  - Padding saturation detection (fraction of output that is padding)
  - Normalization clipping monitoring (fraction of values at ceiling)
  - Component balance validation (entropy of component contributions)
"""

from __future__ import annotations

import numpy as np


def check_onset_coverage(
    fit_results: list,
    signal_length: int,
) -> tuple[float, bool]:
    """Check that fit step produces sufficient onset detections."""
    if signal_length <= 0:
        return 0.0, False
    density = len(fit_results) / signal_length
    return density, density >= 1e-4


def detect_padding_saturation(
    padded: np.ndarray,
    original_length: int,
) -> tuple[float, bool]:
    """Detect excessive padding in an output signal."""
    padded_arr = np.asarray(padded, dtype=np.float64).ravel()
    if padded_arr.size == 0 or original_length <= 0:
        return 0.0, False
    padding_len = max(0, padded_arr.size - original_length)
    fraction = padding_len / padded_arr.size
    return fraction, fraction > 0.5


def monitor_normalization_clipping(
    normalized: np.ndarray,
) -> tuple[float, bool]:
    """Monitor fraction of normalized values clipped at 1.0."""
    arr = np.asarray(normalized, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, False
    clipped = float(np.mean(np.isclose(arr, 1.0)))
    return clipped, clipped > 0.1


def validate_component_balance(
    component_outputs: list[np.ndarray],
) -> tuple[float, bool]:
    """Validate that component contributions are reasonably balanced."""
    if len(component_outputs) <= 1:
        return 1.0, True

    energies: list[float] = []
    for output in component_outputs:
        arr = np.asarray(output, dtype=np.float64).ravel()
        energies.append(float(np.sum(arr ** 2)) if arr.size > 0 else 0.0)

    total = sum(energies)
    if total == 0.0:
        return 1.0, True

    probs = [energy / total for energy in energies]
    entropy = 0.0
    for prob in probs:
        if prob > 0.0:
            entropy -= prob * np.log2(prob)
    max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 1.0
    return float(normalized_entropy), normalized_entropy >= 0.5
