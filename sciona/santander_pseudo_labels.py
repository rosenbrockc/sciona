"""Deterministic confidence-tail selection for pending pseudo-label retraining."""
import numpy as np


def select_pseudo_labels(scores, *, positives, negatives):
    """Select disjoint highest/lowest score tails without changing input order.

Returns selected query positions and labels, in original query order. Scores
may be probabilities or rank scores but must be finite real numbers. Equal-score
boundaries are rejected: the source does not define a tie-breaking rule. Counts
are explicit and never silently reduced for small populations. This operation
alone neither fits models nor regenerates features.
"""
    values = np.asarray(scores)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Expected a finite one-dimensional score vector')
    if any(type(n) is not int or n < 0 for n in (positives, negatives)) or positives + negatives == 0:
        raise ValueError('Expected nonnegative counts with a nonempty selection')
    if positives + negatives > len(values):
        raise ValueError('Requested tails exceed query population')
    order = np.argsort(values, kind='stable')
    for boundary in (negatives, len(values)-positives):
        if 0 < boundary < len(values) and values[order[boundary-1]] == values[order[boundary]]:
            raise ValueError('Ambiguous equal-score selection boundary')
    labels = np.full(len(values), -1, dtype=np.int64)
    if negatives: labels[order[:negatives]] = 0
    if positives: labels[order[len(values)-positives:]] = 1
    positions = np.flatnonzero(labels >= 0)
    return positions, labels[positions]
