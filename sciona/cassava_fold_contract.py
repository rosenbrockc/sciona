"""Runtime-only identity and five-fold alignment contract for the corrected CDG."""
from dataclasses import dataclass, field
from collections.abc import Mapping
import numpy as np


def _keys(values):
    if (not isinstance(values, (list, tuple)) or not values
            or not all(isinstance(v, str) and v for v in values) or len(set(values)) != len(values)):
        raise ValueError('Nonempty unique runtime sample keys required')
    return tuple(values)


@dataclass(frozen=True, repr=False)
class FoldPlan:
    keys: tuple = field(repr=False)
    labels: tuple = field(repr=False)
    assignments: tuple = field(repr=False)


def build_plan(sample_keys, labels, assignments):
    """Normalize external fold identifiers to 0..4 before this boundary.

Keys stay private runtime values. Each row has one held-out assignment;
training uses its complement. Identity aliases for the same physical image
must be resolved by ingestion, not guessed from opaque keys here.
"""
    keys = _keys(sample_keys)
    for values in (labels, assignments):
        if (not isinstance(values, (list, tuple)) or len(values) != len(keys)
                or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 5 for v in values)):
            raise ValueError('Aligned integer class and fold assignments from zero through four required')
    if set(assignments) != set(range(5)):
        raise ValueError('All five held-out folds must be nonempty')
    for fold in range(5):
        if {label for label, assigned in zip(labels, assignments) if assigned != fold} != set(range(5)):
            raise ValueError('Every training complement must contain all five classes')
    return FoldPlan(keys, tuple(labels), tuple(assignments))


def members(plan, fold):
    """Revalidate caller-constructed plans and return disjoint positional tuples."""
    if not isinstance(plan, FoldPlan) or isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 5:
        raise ValueError('Valid fold plan and ordinal required')
    plan = build_plan(plan.keys, plan.labels, plan.assignments)
    return (tuple(i for i, assigned in enumerate(plan.assignments) if assigned != fold),
            tuple(i for i, assigned in enumerate(plan.assignments) if assigned == fold))


def average_folds(expected_keys, predictions):
    """Align all five fold probability matrices by identity, then source mean.

Predictions maps ordinal fold to (runtime keys, probability matrix). Preserve
float32/64 arithmetic, and reject missing/extra rows or mismatched precision.
"""
    expected = _keys(expected_keys)
    if (not isinstance(predictions, Mapping) or set(predictions) != set(range(5))
            or any(type(key) is not int for key in predictions)):
        raise ValueError('Exactly five uniquely numbered fold outputs required')
    matrices = []
    dtype = None
    for fold in range(5):
        keys, values = predictions[fold]
        keys = _keys(keys)
        array = np.asarray(values)
        if (set(keys) != set(expected) or array.shape != (len(keys), 5)
                or array.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or (dtype is not None and dtype != array.dtype)
                or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
                or not np.allclose(array.sum(-1), 1., atol=1e-6)):
            raise ValueError('Complete aligned finite normalized fold probabilities required')
        dtype = array.dtype
        positions = {key: i for i, key in enumerate(keys)}
        matrices.append(array[[positions[key] for key in expected]])
    return np.stack(matrices).mean(axis=0)
