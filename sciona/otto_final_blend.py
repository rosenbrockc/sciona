"""Documented final blend of the Otto winner's second-level model bags.

Inputs are aligned nine-class probabilities from every fitted run. This is only
the final numerical stage; it does not train or qualify the upstream learners.
"""
import numpy as np


def _bag(values, runs):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError,TypeError,OverflowError):
        raise ValueError('Finite aligned probability bags required') from None
    if array.ndim != 3 or array.shape[0] != runs or array.shape[1] < 1 or array.shape[2] != 9:
        raise ValueError('Full run count and nine-class probabilities required')
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError('Probability values must be in [0,1]')
    if not np.allclose(array.sum(axis=2),1.,rtol=0,atol=1e-8):
        raise ValueError('Each run must supply normalized class probabilities')
    return array.mean(axis=0)


def blend(xgboost, neural, adaboost_extratrees):
    tree = _bag(xgboost,250)
    network = _bag(neural,600)
    extra = _bag(adaboost_extratrees,250)
    if tree.shape != network.shape or tree.shape != extra.shape:
        raise ValueError('Prediction rows must align across model families')
    raw = .85 * np.power(tree,.65) * np.power(network,.35) + .15 * extra
    # Preserve the published formula before providing conventional normalized
    # multiclass probabilities; never normalize the geometric branch alone.
    probability = raw / raw.sum(axis=1,keepdims=True)
    return dict(raw_scores=raw.tolist(),probabilities=probability.tolist(),classes=np.argmax(probability,axis=1).tolist())
