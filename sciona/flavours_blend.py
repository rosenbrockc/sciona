"""Independent evaluation of the author notebook's mathematical blend.

The two intermediate groups use classifier1/2 and classifier3/4/5/neural,
respectively. Return the unnormalized source score, not a calibrated probability.
"""
import numpy as np


def blend(classifier_scores, neural_scores):
    if set(classifier_scores) != {'xgb1', 'xgb2', 'xgb3', 'xgb4', 'xgb5'}:
        raise ValueError('All five named classifier score vectors are required')
    arrays = [np.asarray(classifier_scores['xgb'+str(i)], dtype=np.float64) for i in range(1, 6)]
    arrays.append(np.asarray(neural_scores, dtype=np.float64))
    if arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError('Nonempty one-dimensional scores required')
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all()
           or np.any((a < 0) | (a > 1)) for a in arrays):
        raise ValueError('Aligned finite component probabilities in [0, 1] required')
    a, b, c, d, e, neural = arrays
    first = np.sqrt(a) * np.square(b) / 2
    second = (c * d**.85 * e**.01 + .85 * d * neural**900 + 2 * e**1000) / 3.85
    return .5 * second**3.9 + .2 * first**.6 + .0001 * first**.01 * second**.2
