"""Runtime atoms for Divide and Conquer expansion rules.

Provides deterministic, pure functions for divide-and-conquer
quality diagnostics and structural pre-checks:

  - Split balance analysis (detect lopsided partitions)
  - Recursion depth monitoring (detect excessive depth)
  - Merge cost profiling (detect merge-dominated workloads)
  - Subproblem overlap detection (suggest DP when subproblems repeat)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Split balance analysis
# ---------------------------------------------------------------------------


def measure_split_balance(
    left_sizes: np.ndarray,
    right_sizes: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Measure the balance of divide-and-conquer splits.

    Computes the split ratio min(left, right) / max(left, right) for
    each recursion level.  A perfectly balanced split has ratio 1.0;
    worst-case (e.g. quicksort with bad pivot) approaches 0.0.

    Args:
        left_sizes: 1-D array of left subproblem sizes per recursion level.
        right_sizes: 1-D array of right subproblem sizes per recursion level.

    Returns:
        (mean_balance, per_level_balance) where mean_balance is the average
        split ratio and per_level_balance[k] is the ratio at level k.
    """
    left = np.asarray(left_sizes, dtype=np.float64).ravel()
    right = np.asarray(right_sizes, dtype=np.float64).ravel()

    n = min(len(left), len(right))
    if n == 0:
        return 1.0, np.empty(0, dtype=np.float64)

    left = left[:n]
    right = right[:n]

    mins = np.minimum(left, right)
    maxs = np.maximum(left, right)

    # Avoid division by zero for empty subproblems
    ratios = np.where(maxs > 0, mins / maxs, 1.0)
    mean_balance = float(np.mean(ratios))

    return mean_balance, ratios


# ---------------------------------------------------------------------------
# Recursion depth monitoring
# ---------------------------------------------------------------------------


def check_recursion_depth(
    actual_depth: int,
    input_size: int,
) -> tuple[float, bool]:
    """Check whether recursion depth is excessive relative to input size.

    For balanced D&C, expected depth is O(log2(n)).  Depth significantly
    exceeding 2 * log2(n) suggests unbalanced splits or missing base case.

    Args:
        actual_depth: observed maximum recursion depth.
        input_size: size of the original input.

    Returns:
        (depth_ratio, is_excessive) where depth_ratio is
        actual_depth / (2 * log2(n)) and is_excessive is True
        if depth_ratio > 1.0.
    """
    depth = int(actual_depth)
    n = int(input_size)

    if n <= 1:
        return 0.0, False

    expected_max = 2.0 * np.log2(max(n, 2))
    ratio = depth / expected_max
    return ratio, ratio > 1.0


# ---------------------------------------------------------------------------
# Merge cost profiling
# ---------------------------------------------------------------------------


def profile_merge_cost(
    merge_times: np.ndarray,
    total_times: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Profile the fraction of total time spent in merge operations.

    When merge dominates (e.g. > 50% of total time at most levels),
    optimizing the merge step yields the biggest speedup.

    Args:
        merge_times: 1-D array of merge durations per recursion level.
        total_times: 1-D array of total durations per recursion level.

    Returns:
        (mean_merge_fraction, per_level_fraction) where
        mean_merge_fraction is the overall merge time fraction.
    """
    merge = np.asarray(merge_times, dtype=np.float64).ravel()
    total = np.asarray(total_times, dtype=np.float64).ravel()

    n = min(len(merge), len(total))
    if n == 0:
        return 0.0, np.empty(0, dtype=np.float64)

    merge = merge[:n]
    total = total[:n]

    safe_total = np.where(total > 0, total, 1.0)
    fractions = np.where(total > 0, merge / safe_total, 0.0)
    mean_fraction = float(np.mean(fractions))

    return mean_fraction, fractions


# ---------------------------------------------------------------------------
# Subproblem overlap detection
# ---------------------------------------------------------------------------


def detect_subproblem_overlap(
    subproblem_hashes: np.ndarray,
) -> tuple[float, int]:
    """Detect repeated subproblems suggesting DP would be more efficient.

    Counts how many subproblem hashes appear more than once.  High
    overlap means the problem has optimal substructure with reuse,
    and memoization / DP should be considered instead of pure D&C.

    Args:
        subproblem_hashes: 1-D array of integer hashes for each
            subproblem encountered during recursion.

    Returns:
        (overlap_ratio, n_duplicates) where overlap_ratio is the
        fraction of subproblems that are duplicates and n_duplicates
        is the absolute count of repeated entries.
    """
    hashes = np.asarray(subproblem_hashes, dtype=np.int64).ravel()

    if len(hashes) == 0:
        return 0.0, 0

    unique, counts = np.unique(hashes, return_counts=True)
    n_duplicates = int(np.sum(counts[counts > 1]) - len(counts[counts > 1]))
    overlap_ratio = n_duplicates / len(hashes)

    return overlap_ratio, n_duplicates
