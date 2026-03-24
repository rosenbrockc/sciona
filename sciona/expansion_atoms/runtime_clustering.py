"""Runtime atoms for Clustering expansion rules.

Provides deterministic, pure functions for clustering pipeline
diagnostics:

  - Cluster balance analysis (max/min size ratio)
  - Assignment stability monitoring (fraction of points changing)
  - Empty cluster detection (clusters with zero members)
  - Separation validation (inter/intra distance ratio)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cluster balance analysis
# ---------------------------------------------------------------------------


def analyze_cluster_balance(
    cluster_sizes: np.ndarray,
) -> tuple[float, bool]:
    """Analyze cluster size balance.

    A large max/min ratio indicates highly imbalanced clusters.

    Args:
        cluster_sizes: 1D array of cluster sizes (counts).

    Returns:
        (imbalance_ratio, is_balanced) where is_balanced is True if
        imbalance_ratio <= 10.0.
    """
    sizes = np.asarray(cluster_sizes, dtype=np.float64).ravel()
    if sizes.size == 0:
        return 1.0, True

    # Filter out zero-size clusters for ratio computation
    nonzero = sizes[sizes > 0]
    if nonzero.size == 0:
        return float("inf"), False

    if nonzero.size == 1:
        return 1.0, True

    ratio = float(np.max(nonzero) / np.min(nonzero))
    return ratio, ratio <= 10.0


# ---------------------------------------------------------------------------
# Assignment stability
# ---------------------------------------------------------------------------


def monitor_assignment_stability(
    prev_assignments: np.ndarray,
    curr_assignments: np.ndarray,
) -> tuple[float, bool]:
    """Monitor clustering assignment stability between iterations.

    Args:
        prev_assignments: 1D array of previous cluster assignments.
        curr_assignments: 1D array of current cluster assignments.

    Returns:
        (change_fraction, is_stable) where is_stable is True if
        change_fraction <= 0.01.
    """
    prev = np.asarray(prev_assignments, dtype=np.int64).ravel()
    curr = np.asarray(curr_assignments, dtype=np.int64).ravel()

    if prev.size == 0 or curr.size == 0:
        return 0.0, True

    n = min(len(prev), len(curr))
    change_fraction = float(np.mean(prev[:n] != curr[:n]))
    return change_fraction, change_fraction <= 0.01


# ---------------------------------------------------------------------------
# Empty cluster detection
# ---------------------------------------------------------------------------


def detect_empty_clusters(
    cluster_sizes: np.ndarray,
) -> tuple[int, bool]:
    """Detect clusters with zero members.

    Args:
        cluster_sizes: 1D array of cluster sizes (counts).

    Returns:
        (n_empty, has_empty) where has_empty is True if n_empty > 0.
    """
    sizes = np.asarray(cluster_sizes, dtype=np.float64).ravel()
    if sizes.size == 0:
        return 0, False

    n_empty = int(np.sum(sizes == 0))
    return n_empty, n_empty > 0


# ---------------------------------------------------------------------------
# Separation validation
# ---------------------------------------------------------------------------


def validate_separation(
    inter_distances: np.ndarray,
    intra_distances: np.ndarray,
) -> tuple[float, bool]:
    """Validate cluster separation quality.

    Computes the ratio of mean inter-cluster distance to mean
    intra-cluster distance. A ratio >= 1.0 indicates well-separated
    clusters.

    Args:
        inter_distances: 1D array of inter-cluster distances.
        intra_distances: 1D array of intra-cluster distances.

    Returns:
        (separation_ratio, is_well_separated) where is_well_separated
        is True if separation_ratio >= 1.0.
    """
    inter = np.asarray(inter_distances, dtype=np.float64).ravel()
    intra = np.asarray(intra_distances, dtype=np.float64).ravel()

    if inter.size == 0 or intra.size == 0:
        return 1.0, True

    mean_intra = float(np.mean(intra))
    mean_inter = float(np.mean(inter))

    if mean_intra == 0.0:
        if mean_inter == 0.0:
            return 1.0, True
        return float("inf"), True

    ratio = mean_inter / mean_intra
    return ratio, ratio >= 1.0
