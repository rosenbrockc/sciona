"""Runtime atoms for Combinatorial Optimization expansion rules.

Provides deterministic, pure functions for discrete optimization
quality diagnostics:

  - Branching factor analysis (exponential blowup detection)
  - Bound tightness monitoring (upper-lower gap tracking)
  - Symmetry detection (search space redundancy)
  - Pruning effectiveness checking (subtree elimination rate)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Branching factor analysis
# ---------------------------------------------------------------------------


def analyze_branching_factor(
    child_counts: np.ndarray,
) -> tuple[float, bool]:
    """Analyze the effective branching factor of a search tree.

    A high mean branching factor indicates exponential blowup in
    the search space.

    Args:
        child_counts: 1D array of child counts per explored node.

    Returns:
        (mean_branching, is_manageable) where is_manageable is True
        if mean_branching <= 10.
    """
    counts = np.asarray(child_counts, dtype=np.float64).ravel()
    if counts.size == 0:
        return 0.0, True

    mean_branching = float(np.mean(counts))
    return mean_branching, mean_branching <= 10.0


# ---------------------------------------------------------------------------
# Bound tightness monitoring
# ---------------------------------------------------------------------------


def monitor_bound_tightness(
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
) -> tuple[float, bool]:
    """Monitor the gap between upper and lower bounds over time.

    Computes (UB - LB) / max(|UB|, 1) for the latest bounds.
    A large gap indicates loose bounds.

    Args:
        upper_bounds: 1D array of upper bound values over iterations.
        lower_bounds: 1D array of lower bound values over iterations.

    Returns:
        (gap_ratio, is_tight) where is_tight is True if
        gap_ratio <= 0.5.
    """
    ub = np.asarray(upper_bounds, dtype=np.float64).ravel()
    lb = np.asarray(lower_bounds, dtype=np.float64).ravel()

    if ub.size == 0 or lb.size == 0:
        return 0.0, True

    n = min(len(ub), len(lb))
    latest_ub = float(ub[n - 1])
    latest_lb = float(lb[n - 1])

    denom = max(abs(latest_ub), 1.0)
    gap_ratio = (latest_ub - latest_lb) / denom
    return gap_ratio, gap_ratio <= 0.5


# ---------------------------------------------------------------------------
# Symmetry detection
# ---------------------------------------------------------------------------


def detect_symmetry(
    candidate_pairs: np.ndarray,
    equivalence_count: int,
) -> tuple[float, bool]:
    """Detect symmetry in the search space.

    Computes the fraction of candidate solution pairs that are
    equivalent (symmetric). High symmetry suggests symmetry-breaking
    constraints should be added.

    Args:
        candidate_pairs: 1D array (or 2D) of candidate representations.
        equivalence_count: Number of equivalent (symmetric) pairs found.

    Returns:
        (symmetry_fraction, has_symmetry) where has_symmetry is True
        if symmetry_fraction > 0.3.
    """
    pairs = np.asarray(candidate_pairs, dtype=np.float64)
    total = max(pairs.size, 1)
    eq = max(0, int(equivalence_count))

    fraction = eq / total
    return fraction, fraction > 0.3


# ---------------------------------------------------------------------------
# Pruning effectiveness checking
# ---------------------------------------------------------------------------


def check_pruning_effectiveness(
    total_nodes: int,
    pruned_nodes: int,
) -> tuple[float, bool]:
    """Check the effectiveness of pruning in branch-and-bound.

    Computes pruned / total. Effective pruning should eliminate
    at least 10% of explored nodes.

    Args:
        total_nodes: Total number of nodes explored.
        pruned_nodes: Number of nodes pruned.

    Returns:
        (pruning_rate, is_effective) where is_effective is True
        if pruning_rate >= 0.1.
    """
    total = max(int(total_nodes), 1)
    pruned = max(0, min(int(pruned_nodes), total))

    rate = pruned / total
    return rate, rate >= 0.1
