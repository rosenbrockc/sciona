"""Final rank-blend operation for the unfinished Santander ensemble."""
import numpy as np
from scipy.stats import rankdata


def blend_neural_tree(neural_scores, tree_scores):
    """Blend average-tie ranks with recovered neural:tree weights 2.1:1.

Outputs remain rank scores, not probabilities. Each input must represent the
same query rows in the same order. The operation does not fit missing branches.
"""
    arrays = [np.asarray(x) for x in (neural_scores, tree_scores)]
    if any(a.ndim != 1 or not len(a) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Expected nonempty finite real score vectors')
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('Branch query populations differ')
    return (2.1*rankdata(arrays[0],method='average')+rankdata(arrays[1],method='average'))/3.1
