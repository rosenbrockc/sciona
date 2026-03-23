"""Runtime atoms for Graph Optimization expansion rules.

Provides deterministic, pure functions for graph optimization
quality diagnostics and structural pre-checks:

  - Negative edge weight detection (Dijkstra safety check)
  - Relaxation convergence monitoring (early termination detection)
  - Distance overflow detection (numeric stability check)
  - Graph density analysis (algorithm selection guidance)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Negative edge weight detection
# ---------------------------------------------------------------------------


def detect_negative_weights(
    edge_weights: np.ndarray,
) -> tuple[int, float]:
    """Detect negative edge weights that invalidate Dijkstra's algorithm.

    Dijkstra assumes non-negative weights; negative edges produce
    incorrect shortest paths.  Bellman-Ford should be used instead.

    Args:
        edge_weights: 1-D array of edge weights.

    Returns:
        (n_negative, min_weight) where n_negative is the count of
        negative-weight edges and min_weight is the most negative weight.
    """
    weights = np.asarray(edge_weights, dtype=np.float64).ravel()

    if len(weights) == 0:
        return 0, 0.0

    mask = weights < 0
    n_negative = int(np.sum(mask))
    min_weight = float(np.min(weights))

    return n_negative, min_weight


# ---------------------------------------------------------------------------
# Relaxation convergence monitoring
# ---------------------------------------------------------------------------


def monitor_relaxation_convergence(
    distance_snapshots: np.ndarray,
) -> tuple[int, bool]:
    """Monitor whether edge relaxation has converged.

    Compares successive distance snapshots to detect when no further
    updates occur.  Early convergence means remaining relaxation
    iterations can be skipped.

    Args:
        distance_snapshots: 2-D array of shape (n_iterations, n_nodes),
            each row is the distance array after one relaxation pass.

    Returns:
        (converged_at, has_converged) where converged_at is the first
        iteration with no change (-1 if never converged), and
        has_converged indicates whether convergence was reached.
    """
    snaps = np.asarray(distance_snapshots, dtype=np.float64)

    if snaps.ndim == 1:
        snaps = snaps.reshape(1, -1)

    if len(snaps) < 2:
        return -1, False

    for i in range(1, len(snaps)):
        if np.allclose(snaps[i], snaps[i - 1], equal_nan=True):
            return i, True

    return -1, False


# ---------------------------------------------------------------------------
# Distance overflow detection
# ---------------------------------------------------------------------------


def detect_distance_overflow(
    distances: np.ndarray,
    overflow_threshold: float = 1e15,
) -> tuple[int, float]:
    """Detect numeric overflow in distance computations.

    When edge weights are large, repeated additions during relaxation
    can overflow float64.  This check flags nodes whose distances
    exceed the threshold.

    Args:
        distances: 1-D array of node distances.
        overflow_threshold: maximum safe distance value.

    Returns:
        (n_overflow, max_distance) where n_overflow is the count of
        nodes with distance exceeding the threshold.
    """
    dists = np.asarray(distances, dtype=np.float64).ravel()

    if len(dists) == 0:
        return 0, 0.0

    finite = dists[np.isfinite(dists)]
    if len(finite) == 0:
        return int(len(dists)), float("inf")

    n_overflow = int(np.sum(np.abs(finite) > overflow_threshold))
    max_dist = float(np.max(np.abs(finite)))

    return n_overflow, max_dist


# ---------------------------------------------------------------------------
# Graph density analysis
# ---------------------------------------------------------------------------


def analyze_graph_density(
    n_nodes: int,
    n_edges: int,
) -> tuple[float, str]:
    """Analyze graph density for algorithm selection guidance.

    Sparse graphs (density < 0.1) favor adjacency-list algorithms
    (Dijkstra with heap, Bellman-Ford).  Dense graphs (density > 0.5)
    may benefit from Floyd-Warshall's matrix approach.

    Args:
        n_nodes: number of nodes in the graph.
        n_edges: number of edges in the graph.

    Returns:
        (density, recommendation) where density is |E| / |V|² and
        recommendation is one of "sparse", "moderate", or "dense".
    """
    n = int(n_nodes)
    m = int(n_edges)

    if n <= 1:
        return 0.0, "sparse"

    max_edges = n * (n - 1)  # directed graph
    if max_edges == 0:
        return 0.0, "sparse"

    density = m / max_edges

    if density < 0.1:
        recommendation = "sparse"
    elif density > 0.5:
        recommendation = "dense"
    else:
        recommendation = "moderate"

    return density, recommendation
