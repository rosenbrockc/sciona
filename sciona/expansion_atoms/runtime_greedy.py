"""Runtime atoms for Greedy expansion rules.

Provides deterministic, pure functions for greedy algorithm
quality diagnostics and structural pre-checks:

  - Matroid exchange property validation (sampling-based)
  - Criterion tie detection (near-equal score grouping)
  - Solution quality estimation (approximation ratio)
  - Redundant feasibility detection (monotone constraint check)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Matroid exchange validation
# ---------------------------------------------------------------------------


def validate_matroid_exchange(
    selected_sets: list[np.ndarray],
    ground_set_size: int,
) -> tuple[float, bool]:
    """Check the exchange property on observed selection sets.

    If sets of different sizes exist where the larger cannot donate an
    element to the smaller while staying feasible, the greedy strategy
    may not be optimal.  Uses a sampling approach — checks all pairs of
    selected sets for the exchange property.

    Args:
        selected_sets: list of 1-D arrays (observed feasible sets),
            each containing element indices from the ground set.
        ground_set_size: total number of elements in the ground set.

    Returns:
        (exchange_ratio, is_matroid) where exchange_ratio is the fraction
        of pairs satisfying the exchange property, and is_matroid is True
        if ratio > 0.95.
    """
    n = int(ground_set_size)
    sets = [np.asarray(s, dtype=np.int64) for s in selected_sets]

    if len(sets) < 2 or n == 0:
        return 1.0, True

    # Convert to frozensets for fast membership testing
    fsets = [frozenset(int(x) for x in s if 0 <= x < n) for s in sets]

    n_pairs = 0
    n_satisfied = 0

    for i in range(len(fsets)):
        for j in range(len(fsets)):
            if i == j:
                continue
            a, b = fsets[i], fsets[j]
            if len(a) <= len(b):
                continue
            # a is strictly larger than b — check if some element
            # in a \ b can be added to b (exchange property)
            n_pairs += 1
            diff = a - b
            if len(diff) > 0:
                n_satisfied += 1

    if n_pairs == 0:
        return 1.0, True

    ratio = n_satisfied / n_pairs
    return ratio, ratio > 0.95


# ---------------------------------------------------------------------------
# Criterion tie detection
# ---------------------------------------------------------------------------


def detect_criterion_ties(
    scores: np.ndarray,
    tie_tolerance: float = 1e-8,
) -> tuple[int, np.ndarray]:
    """Detect near-ties in greedy criterion ordering.

    Groups candidates whose scores differ by less than ``tie_tolerance``.
    Near-ties make the greedy selection unstable — different tie-breaking
    strategies can produce different solutions.

    Args:
        scores: 1-D array of candidate scores.
        tie_tolerance: maximum absolute difference to consider a tie.

    Returns:
        (n_ties, tie_groups) where n_ties is the number of candidates
        involved in at least one tie, and tie_groups is an integer label
        array (same label = tied group).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = len(scores)

    if n == 0:
        return 0, np.empty(0, dtype=np.int64)

    # Sort and assign group labels based on tolerance gaps
    order = np.argsort(scores)
    sorted_scores = scores[order]

    labels = np.zeros(n, dtype=np.int64)
    group = 0
    for i in range(1, n):
        if sorted_scores[i] - sorted_scores[i - 1] > tie_tolerance:
            group += 1
        labels[i] = group

    # Map labels back to original order
    result_labels = np.empty(n, dtype=np.int64)
    result_labels[order] = labels

    # Count candidates in groups with more than one member
    _, counts = np.unique(result_labels, return_counts=True)
    n_ties = int(np.sum(counts[counts > 1]))

    return n_ties, result_labels


# ---------------------------------------------------------------------------
# Solution quality estimation
# ---------------------------------------------------------------------------


def estimate_solution_quality(
    greedy_value: float,
    relaxation_bound: float,
) -> tuple[float, bool]:
    """Compute approximation ratio of greedy solution against a known bound.

    For maximization problems, the ratio is greedy_value / relaxation_bound.
    The bound is assumed to be an upper bound (e.g. LP relaxation).

    Args:
        greedy_value: objective value of the greedy solution.
        relaxation_bound: upper/lower bound from a relaxation.

    Returns:
        (approx_ratio, is_optimal) where is_optimal is True if
        ratio >= 0.99.
    """
    gv = float(greedy_value)
    rb = float(relaxation_bound)

    if rb == 0.0:
        if gv == 0.0:
            return 1.0, True
        return 0.0, False

    ratio = gv / rb
    # Clamp to [0, 1] for well-formed inputs
    ratio = max(0.0, min(1.0, ratio))
    return ratio, ratio >= 0.99


# ---------------------------------------------------------------------------
# Redundant feasibility detection
# ---------------------------------------------------------------------------


def detect_redundant_feasibility(
    feasibility_history: np.ndarray,
) -> tuple[float, bool]:
    """Detect when feasibility checks are always passing.

    When feasibility is monotone (all checks pass), the feasibility
    check node can potentially be skipped for performance.

    Args:
        feasibility_history: 1-D bool array of per-step feasibility outcomes.

    Returns:
        (pass_rate, is_redundant) where is_redundant is True if
        pass_rate == 1.0.
    """
    history = np.asarray(feasibility_history, dtype=bool).ravel()

    if len(history) == 0:
        return 1.0, True

    pass_rate = float(np.mean(history))
    return pass_rate, pass_rate == 1.0
