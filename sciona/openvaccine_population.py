"""Explicit population weights and aligned reversal for OpenVaccine training.

Cluster membership and proximity factors must be computed from the caller's
reviewed population definition. Counts cover that population, not each batch.
No historical clustering algorithm or proximity formula is inferred here.
"""
import numpy as np

from sciona.openvaccine_models import validate_inputs


def population_weights(cluster_ids, proximity_factors):
    ids, factors = np.asarray(cluster_ids), np.asarray(proximity_factors)
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'iu':
        raise ValueError('Nonempty integer cluster membership required')
    if factors.shape != ids.shape or factors.dtype != np.float32:
        raise ValueError('Float32 proximity factors must match population')
    if not np.isfinite(factors).all() or (factors < 0).any() or not (factors > 0).any():
        raise ValueError('Finite nonnegative factors with positive support required')
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    values = factors.astype(np.float64)/np.sqrt(counts[inverse])
    result = values.astype(np.float32)
    if not (result > 0).any() or not np.isfinite(result.sum()):
        raise ValueError('Population weights not representable with finite positive sum')
    return result


def reverse_training_batch(nodes, adjacency, targets, sample_weights, reverse_flags):
    nodes, adjacency = validate_inputs(nodes, adjacency)
    targets, weights, flags = map(np.asarray, (targets, sample_weights, reverse_flags))
    if targets.dtype != np.float32 or targets.shape != (len(nodes),nodes.shape[1]-2,5) or np.isinf(targets).any():
        raise ValueError('Float32 targets must match outputs; only NaN masks permitted')
    if (weights.dtype != np.float32 or weights.shape != (len(nodes),)
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError('Finite nonnegative batch weights with positive sum required')
    if flags.dtype != np.bool_ or flags.shape != (len(nodes),):
        raise ValueError('One boolean reversal decision per example required')
    n, a, t = nodes.copy(), adjacency.copy(), targets.copy()
    n[flags] = nodes[flags,::-1,:]
    a[flags] = adjacency[flags,::-1,::-1,:]
    t[flags] = targets[flags,::-1,:]
    # Boundary tokens move with features, matching source reverse_input.
    return n, a, t, weights.copy()
