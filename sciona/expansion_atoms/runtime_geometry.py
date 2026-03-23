"""Runtime atoms for Geometry expansion rules.

Provides deterministic, pure functions for computational geometry
quality diagnostics and structural pre-checks:

  - Collinearity detection (degenerate point configurations)
  - Numeric precision analysis (floating-point predicate robustness)
  - Duplicate point detection (preprocessing quality)
  - Convexity validation (output structure correctness)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Collinearity detection
# ---------------------------------------------------------------------------


def detect_collinear_points(
    points: np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[int, float]:
    """Detect collinear point triples that cause degenerate geometry.

    Many geometric algorithms (convex hull, Delaunay) fail or produce
    degenerate results when many input points are collinear.

    Args:
        points: (n, 2) array of 2D point coordinates.
        tolerance: cross-product threshold for collinearity.

    Returns:
        (n_collinear_triples, collinear_fraction) where
        collinear_fraction is the ratio of collinear triples sampled.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    n = len(pts)
    if n < 3:
        return 0, 0.0

    # Sample up to 1000 random triples for efficiency
    max_samples = min(1000, n * (n - 1) * (n - 2) // 6)
    rng = np.random.RandomState(42)  # deterministic
    count = 0
    for _ in range(max_samples):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross < tolerance:
            count += 1

    fraction = count / max_samples if max_samples > 0 else 0.0
    return count, fraction


# ---------------------------------------------------------------------------
# Numeric precision analysis
# ---------------------------------------------------------------------------


def analyze_numeric_precision(
    points: np.ndarray,
) -> tuple[float, bool]:
    """Analyze whether point coordinates risk floating-point issues.

    When the ratio of max coordinate magnitude to minimum inter-point
    distance is very large, geometric predicates (orientation, in-circle)
    may produce incorrect results.

    Args:
        points: (n, 2) array of 2D point coordinates.

    Returns:
        (condition_number, is_risky) where condition_number is
        max_coord / min_distance and is_risky is True if > 1e10.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    n = len(pts)
    if n < 2:
        return 0.0, False

    max_coord = float(np.max(np.abs(pts)))
    if max_coord == 0:
        return 0.0, False

    # Sample pairwise distances for efficiency
    rng = np.random.RandomState(42)
    n_samples = min(1000, n * (n - 1) // 2)
    min_dist = float("inf")
    for _ in range(n_samples):
        i, j = rng.choice(n, 2, replace=False)
        d = np.linalg.norm(pts[i] - pts[j])
        if d > 0 and d < min_dist:
            min_dist = d

    if min_dist == float("inf") or min_dist == 0:
        return float("inf"), True

    cond = max_coord / min_dist
    return cond, cond > 1e10


# ---------------------------------------------------------------------------
# Duplicate point detection
# ---------------------------------------------------------------------------


def detect_duplicate_points(
    points: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[int, float]:
    """Detect duplicate or near-duplicate points.

    Duplicate points can cause division-by-zero in geometric
    computations and degenerate output structures.

    Args:
        points: (n, 2) array of 2D point coordinates.
        tolerance: distance threshold for considering points duplicate.

    Returns:
        (n_duplicates, duplicate_fraction) where duplicate_fraction
        is the fraction of points that have a near-duplicate.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    n = len(pts)
    if n < 2:
        return 0, 0.0

    # Sort by x then y for efficient sweep
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[order]

    dup_count = 0
    for i in range(1, n):
        if np.linalg.norm(sorted_pts[i] - sorted_pts[i - 1]) < tolerance:
            dup_count += 1

    fraction = dup_count / n
    return dup_count, fraction


# ---------------------------------------------------------------------------
# Convexity validation
# ---------------------------------------------------------------------------


def validate_convexity(
    hull_points: np.ndarray,
) -> tuple[int, bool]:
    """Validate that a polygon is convex by checking cross-product signs.

    All consecutive edge cross-products should have the same sign
    for a convex polygon.

    Args:
        hull_points: (m, 2) array of hull vertices in order.

    Returns:
        (n_violations, is_convex) where n_violations is the count of
        consecutive edges with inconsistent cross-product sign.
    """
    pts = np.asarray(hull_points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)

    m = len(pts)
    if m < 3:
        return 0, True

    violations = 0
    sign = 0
    for i in range(m):
        a = pts[i]
        b = pts[(i + 1) % m]
        c = pts[(i + 2) % m]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            current_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = current_sign
            elif current_sign != sign:
                violations += 1

    return violations, violations == 0
