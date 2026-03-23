"""Runtime atoms for Dynamic Programming expansion rules.

Provides deterministic, pure functions for DP table optimization
and structural diagnostics:

  - Table sparsity detection (identify underutilized DP tables)
  - Infeasible state pruning (skip states that violate constraints)
  - Table compression (retain only recent rows when reuse distance is bounded)
  - Subproblem overlap validation (verify memoization is justified)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Table sparsity detection
# ---------------------------------------------------------------------------


def detect_table_sparsity(
    table: np.ndarray,
    fill_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute fraction of DP table cells that are actually filled/used.

    Treats NaN and 0.0 as unfilled by default.  If *fill_mask* is provided,
    it is used directly instead of inferring fill status.

    Args:
        table: N-dimensional DP table (any numeric dtype).
        fill_mask: Optional boolean array of same shape as *table*.
            True means the cell is filled.  When ``None``, cells that
            are NaN or exactly 0.0 are treated as unfilled.

    Returns:
        (density, sparse_indices) where *density* is the fraction of
        filled cells and *sparse_indices* contains the flat indices of
        filled cells.
    """
    table = np.asarray(table, dtype=np.float64)

    if fill_mask is not None:
        mask = np.asarray(fill_mask, dtype=bool)
    else:
        not_nan = ~np.isnan(table)
        not_zero = table != 0.0
        mask = not_nan & not_zero

    total = mask.size
    if total == 0:
        return 0.0, np.empty(0, dtype=np.int64)

    filled_flat = np.flatnonzero(mask)
    density = float(len(filled_flat) / total)
    return density, filled_flat.astype(np.int64)


# ---------------------------------------------------------------------------
# Infeasible state pruning
# ---------------------------------------------------------------------------


def prune_infeasible_states(
    table_shape: tuple[int, ...],
    constraints: np.ndarray,
    state_bounds: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Build a feasibility mask over the DP state space given bound constraints.

    For each cell in the table, the cell's multi-dimensional index is checked
    against per-dimension bounds.  A cell is infeasible if any of its index
    components falls outside the corresponding [lower, upper] range defined
    by *state_bounds*.

    Args:
        table_shape: Shape of the DP table (n_dims dimensions).
        constraints: shape (n_dims, 2) — each row is [lower, upper] bound
            for a constraint (unused, reserved for future constraint types).
        state_bounds: shape (n_dims, 2) — each row is [lower, upper] bound
            for the state variable along that dimension.

    Returns:
        (feasible_mask, n_pruned) where *feasible_mask* is a bool array of
        shape *table_shape* (True = feasible) and *n_pruned* is the count
        of infeasible cells.
    """
    state_bounds = np.asarray(state_bounds, dtype=np.float64)
    n_dims = len(table_shape)

    feasible = np.ones(table_shape, dtype=bool)

    for dim in range(min(n_dims, len(state_bounds))):
        lower = state_bounds[dim, 0]
        upper = state_bounds[dim, 1]
        # Build index array for this dimension
        idx = np.arange(table_shape[dim])
        # Reshape for broadcasting: size along dim, 1 elsewhere
        shape = [1] * n_dims
        shape[dim] = table_shape[dim]
        idx = idx.reshape(shape)
        feasible &= (idx >= lower) & (idx <= upper)

    n_pruned = int(np.sum(~feasible))
    return feasible, n_pruned


# ---------------------------------------------------------------------------
# Table compression
# ---------------------------------------------------------------------------


def compress_dp_table(
    table: np.ndarray,
    reuse_distance: int,
) -> tuple[np.ndarray, float]:
    """Retain only the most recent *reuse_distance* rows of a DP table.

    Many DP recurrences only look back a bounded number of rows (e.g. 1
    for Fibonacci, 2 for edit distance).  Discarding older rows saves
    memory proportional to 1 - reuse_distance / n_rows.

    Args:
        table: N-dimensional DP table; compression is along axis 0.
        reuse_distance: Number of most-recent rows to retain.

    Returns:
        (compressed_table, memory_saved_ratio) where *memory_saved_ratio*
        is the fraction of table memory that was discarded.
    """
    table = np.asarray(table)
    n_rows = table.shape[0]

    if reuse_distance <= 0 or reuse_distance >= n_rows:
        return table, 0.0

    compressed = table[-reuse_distance:]
    memory_saved_ratio = float(1.0 - reuse_distance / n_rows)
    return compressed, memory_saved_ratio


# ---------------------------------------------------------------------------
# Subproblem overlap validation
# ---------------------------------------------------------------------------


def validate_subproblem_overlap(
    call_counts: np.ndarray,
) -> tuple[float, bool]:
    """Check whether subproblems are reused enough to justify memoization.

    A low reuse ratio indicates that divide-and-conquer (without memoization)
    might be more appropriate than DP.

    Args:
        call_counts: 1-D array where ``call_counts[i]`` is the number of
            times subproblem *i* was evaluated.

    Returns:
        (reuse_ratio, has_overlap) where *reuse_ratio* is the mean number
        of evaluations per unique subproblem and *has_overlap* is True if
        the ratio exceeds 1.5.
    """
    call_counts = np.asarray(call_counts, dtype=np.float64)

    if len(call_counts) == 0:
        return 0.0, False

    # Only count subproblems that were actually called at least once
    active = call_counts[call_counts > 0]
    if len(active) == 0:
        return 0.0, False

    reuse_ratio = float(np.mean(active))
    has_overlap = reuse_ratio > 1.5
    return reuse_ratio, has_overlap
