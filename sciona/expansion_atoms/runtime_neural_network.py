"""Runtime atoms for Neural Network expansion rules.

Provides deterministic, pure functions for neural network training
pipeline diagnostics:

  - Gradient explosion detection (max gradient norm)
  - Activation statistics analysis (dead neuron fraction)
  - Loss convergence monitoring (plateau detection)
  - Weight distribution checking (layer norm balance)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Gradient explosion detection
# ---------------------------------------------------------------------------


def detect_gradient_explosion(
    gradients: np.ndarray,
) -> tuple[float, bool]:
    """Detect exploding gradients by checking max gradient norm.

    Args:
        gradients: 2D array (n_layers, n_params) of gradient vectors,
                   or 1D array for a single gradient.

    Returns:
        (max_norm, is_exploding) where is_exploding is True if
        max_norm > 100.0.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if g.size == 0:
        return 0.0, False

    if g.ndim == 1:
        g = g.reshape(1, -1)

    norms = np.linalg.norm(g, axis=1)
    max_norm = float(np.max(norms))
    return max_norm, max_norm > 100.0


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------


def analyze_activation_statistics(
    activations: np.ndarray,
) -> tuple[float, bool]:
    """Analyze activation statistics to detect dead neurons.

    A neuron is considered dead if its activation is exactly zero.

    Args:
        activations: 2D array (n_samples, n_neurons) of activation values.

    Returns:
        (dead_fraction, has_dead_neurons) where has_dead_neurons is True
        if dead_fraction > 0.5.
    """
    a = np.asarray(activations, dtype=np.float64)
    if a.size == 0:
        return 0.0, False

    dead_fraction = float(np.mean(a == 0.0))
    return dead_fraction, dead_fraction > 0.5


# ---------------------------------------------------------------------------
# Loss convergence monitoring
# ---------------------------------------------------------------------------


def monitor_loss_convergence(
    loss_history: np.ndarray,
) -> tuple[float, bool]:
    """Monitor loss convergence to detect plateaus.

    Computes the relative change in loss over a recent window.

    Args:
        loss_history: 1D array of loss values over training steps.

    Returns:
        (plateau_ratio, is_plateaued) where is_plateaued is True
        if plateau_ratio < 1e-6.
    """
    h = np.asarray(loss_history, dtype=np.float64).ravel()
    if h.size < 2:
        return 1.0, False

    # Relative change between last and second-to-last
    recent = h[-1]
    prev = h[-2]

    if prev == 0.0:
        if recent == 0.0:
            return 0.0, True
        return float("inf"), False

    plateau_ratio = float(abs(recent - prev) / abs(prev))
    return plateau_ratio, plateau_ratio < 1e-6


# ---------------------------------------------------------------------------
# Weight distribution check
# ---------------------------------------------------------------------------


def check_weight_distribution(
    weights: np.ndarray,
) -> tuple[float, bool]:
    """Check weight distribution balance across layers.

    Computes the ratio of max to min layer norm. A large ratio
    indicates imbalanced weight initialization or training.

    Args:
        weights: 2D array (n_layers, n_params) of weight vectors,
                 or 1D array for a single layer.

    Returns:
        (norm_ratio, is_balanced) where is_balanced is True if
        norm_ratio <= 100.0.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 1.0, True

    if w.ndim == 1:
        return 1.0, True

    norms = np.linalg.norm(w, axis=1)
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))

    if min_norm == 0.0:
        if max_norm == 0.0:
            return 1.0, True
        return float("inf"), False

    ratio = max_norm / min_norm
    return ratio, ratio <= 100.0
