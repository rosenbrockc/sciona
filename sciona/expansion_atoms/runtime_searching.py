"""Runtime atoms for Searching expansion rules.

Provides deterministic, pure functions for searching algorithm
quality diagnostics and structural pre-checks:

  - Sorted order validation (binary search precondition)
  - Distribution uniformity analysis (interpolation search guidance)
  - Midpoint overflow detection (integer overflow in index arithmetic)
  - Iteration count analysis (detect excessive search iterations)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Sorted order validation
# ---------------------------------------------------------------------------


def validate_sorted_order(
    data: np.ndarray,
) -> tuple[int, bool]:
    """Validate that input data is sorted in non-decreasing order.

    Binary search and interpolation search require sorted input.
    Unsorted input produces silently incorrect results.

    Args:
        data: 1-D array of search data.

    Returns:
        (n_violations, is_sorted) where n_violations is the count of
        adjacent pairs that are out of order.
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if len(data) <= 1:
        return 0, True

    violations = int(np.sum(data[:-1] > data[1:]))
    return violations, violations == 0


# ---------------------------------------------------------------------------
# Distribution uniformity analysis
# ---------------------------------------------------------------------------


def analyze_distribution_uniformity(
    data: np.ndarray,
) -> tuple[float, str]:
    """Analyze how uniformly distributed the search data is.

    Interpolation search is O(log log n) for uniform data but
    degrades to O(n) for highly skewed distributions.

    Args:
        data: 1-D sorted array of search data.

    Returns:
        (uniformity_score, recommendation) where uniformity_score
        is in [0, 1] (1 = perfectly uniform) and recommendation is
        "interpolation" or "binary".
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if len(data) <= 2:
        return 1.0, "binary"

    # Compute gaps between consecutive elements
    gaps = np.diff(data)

    if np.all(gaps == 0):
        return 0.0, "binary"

    nonzero_gaps = gaps[gaps > 0]
    if len(nonzero_gaps) == 0:
        return 0.0, "binary"

    # Coefficient of variation of gaps: lower = more uniform
    mean_gap = np.mean(nonzero_gaps)
    std_gap = np.std(nonzero_gaps)

    if mean_gap == 0:
        return 0.0, "binary"

    cv = std_gap / mean_gap
    # Map CV to uniformity score: cv=0 → 1.0, cv≥2 → 0.0
    uniformity = max(0.0, 1.0 - cv / 2.0)

    recommendation = "interpolation" if uniformity > 0.7 else "binary"
    return uniformity, recommendation


# ---------------------------------------------------------------------------
# Midpoint overflow detection
# ---------------------------------------------------------------------------


def detect_midpoint_overflow(
    lo: int,
    hi: int,
) -> tuple[bool, int]:
    """Detect potential integer overflow in midpoint calculation.

    The naive (lo + hi) / 2 overflows when lo + hi exceeds the
    integer range.  The safe form is lo + (hi - lo) // 2.

    Args:
        lo: lower bound index.
        hi: upper bound index.

    Returns:
        (would_overflow, safe_mid) where would_overflow is True if
        lo + hi would exceed int64 range, and safe_mid is the
        correctly computed midpoint.
    """
    lo_val = int(lo)
    hi_val = int(hi)

    max_int64 = np.iinfo(np.int64).max
    would_overflow = lo_val > 0 and hi_val > max_int64 - lo_val

    safe_mid = lo_val + (hi_val - lo_val) // 2
    return would_overflow, safe_mid


# ---------------------------------------------------------------------------
# Iteration count analysis
# ---------------------------------------------------------------------------


def analyze_iteration_count(
    n_iterations: int,
    n_elements: int,
) -> tuple[float, bool]:
    """Check whether search iteration count is excessive.

    Binary search should complete in O(log2 n) iterations.
    Exceeding 2 * log2(n) suggests a bug or degenerate case.

    Args:
        n_iterations: observed number of search iterations.
        n_elements: size of the search space.

    Returns:
        (iteration_ratio, is_excessive) where iteration_ratio is
        n_iterations / (2 * log2(n)) and is_excessive is True
        if ratio > 1.0.
    """
    iters = int(n_iterations)
    n = int(n_elements)

    if n <= 1:
        return 0.0, False

    expected = 2.0 * np.log2(max(n, 2))
    ratio = iters / expected
    return ratio, ratio > 1.0
