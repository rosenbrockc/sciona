"""Runtime atoms for Sorting expansion rules.

Provides deterministic, pure functions for sorting algorithm
quality diagnostics and structural pre-checks:

  - Presortedness detection (measure existing order to pick adaptive algorithm)
  - Comparison count analysis (detect excessive comparisons)
  - Swap count analysis (detect excessive data movement)
  - Stability validation (check whether equal-key order is preserved)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Presortedness detection
# ---------------------------------------------------------------------------


def measure_presortedness(
    data: np.ndarray,
) -> tuple[float, int]:
    """Measure how sorted the input already is.

    Counts the number of inversions (pairs where data[i] > data[j]
    for i < j) using a simple O(n) adjacent-pair approximation.
    Fully sorted → 0.0, fully reversed → 1.0.

    Args:
        data: 1-D array of comparable values.

    Returns:
        (disorder_ratio, n_adjacent_inversions) where disorder_ratio
        is the fraction of adjacent pairs that are out of order.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data)

    if n <= 1:
        return 0.0, 0

    inversions = int(np.sum(data[:-1] > data[1:]))
    ratio = inversions / (n - 1)

    return ratio, inversions


# ---------------------------------------------------------------------------
# Comparison count analysis
# ---------------------------------------------------------------------------


def analyze_comparison_count(
    n_comparisons: int,
    n_elements: int,
) -> tuple[float, bool]:
    """Check whether comparison count is excessive.

    Optimal comparison-based sorting uses O(n log n) comparisons.
    A count significantly exceeding 2 * n * log2(n) suggests a
    degenerate case (e.g. quadratic fallback).

    Args:
        n_comparisons: observed number of comparisons.
        n_elements: number of elements being sorted.

    Returns:
        (comparison_ratio, is_excessive) where comparison_ratio is
        n_comparisons / (2 * n * log2(n)) and is_excessive is True
        if ratio > 1.0.
    """
    comps = int(n_comparisons)
    n = int(n_elements)

    if n <= 1:
        return 0.0, False

    expected = 2.0 * n * np.log2(max(n, 2))
    ratio = comps / expected
    return ratio, ratio > 1.0


# ---------------------------------------------------------------------------
# Swap count analysis
# ---------------------------------------------------------------------------


def analyze_swap_count(
    n_swaps: int,
    n_elements: int,
) -> tuple[float, bool]:
    """Check whether swap/move count is excessive.

    For comparison-based sorting, O(n log n) swaps is typical.
    Exceeding 2 * n * log2(n) suggests excessive data movement
    (e.g. insertion sort on reverse-sorted input).

    Args:
        n_swaps: observed number of swaps/moves.
        n_elements: number of elements being sorted.

    Returns:
        (swap_ratio, is_excessive) where swap_ratio is
        n_swaps / (2 * n * log2(n)) and is_excessive is True
        if ratio > 1.0.
    """
    swaps = int(n_swaps)
    n = int(n_elements)

    if n <= 1:
        return 0.0, False

    expected = 2.0 * n * np.log2(max(n, 2))
    ratio = swaps / expected
    return ratio, ratio > 1.0


# ---------------------------------------------------------------------------
# Stability validation
# ---------------------------------------------------------------------------


def validate_stability(
    keys: np.ndarray,
    original_indices: np.ndarray,
    sorted_indices: np.ndarray,
) -> tuple[int, bool]:
    """Check whether a sort preserves the relative order of equal keys.

    A stable sort maintains the original order among elements with
    equal keys.  This check counts violations.

    Args:
        keys: 1-D array of sort keys.
        original_indices: 1-D array of original positions (0..n-1).
        sorted_indices: 1-D array giving the order after sorting
            (sorted_indices[k] = original index of k-th sorted element).

    Returns:
        (n_violations, is_stable) where n_violations is the count of
        adjacent equal-key pairs whose original order was reversed.
    """
    keys = np.asarray(keys, dtype=np.float64).ravel()
    orig = np.asarray(original_indices, dtype=np.int64).ravel()
    srt = np.asarray(sorted_indices, dtype=np.int64).ravel()

    n = len(srt)
    if n <= 1:
        return 0, True

    violations = 0
    for i in range(n - 1):
        idx_a, idx_b = int(srt[i]), int(srt[i + 1])
        if 0 <= idx_a < len(keys) and 0 <= idx_b < len(keys):
            if keys[idx_a] == keys[idx_b] and idx_a > idx_b:
                violations += 1

    return violations, violations == 0
