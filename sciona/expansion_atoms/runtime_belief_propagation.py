"""Runtime atoms for Belief Propagation expansion rules.

Provides deterministic, pure functions for belief propagation
quality diagnostics:

  - Message convergence monitoring (iterative convergence check)
  - Belief normalization validation (probability sum-to-one check)
  - Message damping analysis (oscillation detection)
  - Graph cycle detection (loopy BP reliability assessment)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Message convergence monitoring
# ---------------------------------------------------------------------------


def monitor_message_convergence(
    message_deltas: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[float, bool]:
    """Monitor convergence of message passing iterations.

    Tracks the maximum message change between iterations.
    Convergence is reached when changes are below tolerance.

    Args:
        message_deltas: 1-D array of max message changes per iteration.
        tolerance: convergence threshold.

    Returns:
        (final_delta, has_converged) where final_delta is the last
        message change and has_converged is True if < tolerance.
    """
    d = np.asarray(message_deltas, dtype=np.float64).ravel()
    if len(d) == 0:
        return 0.0, True

    final = float(d[-1])
    return final, final < tolerance


# ---------------------------------------------------------------------------
# Belief normalization validation
# ---------------------------------------------------------------------------


def validate_belief_normalization(
    beliefs: np.ndarray,
    tolerance: float = 1e-8,
) -> tuple[float, bool]:
    """Validate that beliefs (marginals) are properly normalized.

    Each variable's belief should sum to 1.0.  Deviations indicate
    numerical issues in message passing.

    Args:
        beliefs: (n_variables, n_states) array of marginal beliefs.
        tolerance: max acceptable deviation from 1.0.

    Returns:
        (max_deviation, is_normalized) where max_deviation is the
        worst sum-to-one violation.
    """
    b = np.asarray(beliefs, dtype=np.float64)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if b.shape[0] == 0:
        return 0.0, True

    sums = np.sum(b, axis=1)
    deviations = np.abs(sums - 1.0)
    max_dev = float(np.max(deviations))
    return max_dev, max_dev < tolerance


# ---------------------------------------------------------------------------
# Message damping analysis
# ---------------------------------------------------------------------------


def analyze_message_damping(
    message_history: np.ndarray,
) -> tuple[float, bool]:
    """Analyze whether messages oscillate, suggesting damping is needed.

    Oscillating messages indicate the graph structure (cycles) is
    causing instability in belief propagation.

    Args:
        message_history: (n_iterations, n_messages) array of message
            values per iteration.

    Returns:
        (oscillation_score, needs_damping) where oscillation_score is
        the fraction of messages that changed sign between consecutive
        updates and needs_damping is True if > 0.1.
    """
    h = np.asarray(message_history, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(-1, 1)

    n_iters = h.shape[0]
    if n_iters < 3:
        return 0.0, False

    # Check for sign changes in differences
    diffs = np.diff(h, axis=0)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    total_possible = (n_iters - 2) * h.shape[1]

    if total_possible == 0:
        return 0.0, False

    score = float(sign_changes) / total_possible
    return score, score > 0.1


# ---------------------------------------------------------------------------
# Graph cycle detection
# ---------------------------------------------------------------------------


def detect_graph_cycles(
    adjacency: np.ndarray,
) -> tuple[int, bool]:
    """Detect cycles in the factor graph.

    Belief propagation is exact on trees but only approximate on
    graphs with cycles (loopy BP).  More cycles means less reliable.

    Args:
        adjacency: (n, n) factor graph adjacency matrix.

    Returns:
        (n_extra_edges, is_tree) where n_extra_edges is the number
        of edges beyond a spanning tree (|E| - |V| + 1 for connected)
        and is_tree is True if n_extra_edges == 0.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return 0, True

    n = A.shape[0]
    if n == 0:
        return 0, True

    # Count edges (undirected)
    n_edges = int(np.sum(A != 0)) // 2

    # Count components via BFS
    visited = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for nb in np.nonzero(A[node] != 0)[0]:
                if not visited[nb]:
                    stack.append(nb)

    # For a forest: edges = vertices - components
    extra = n_edges - (n - n_components)
    return max(extra, 0), extra == 0
