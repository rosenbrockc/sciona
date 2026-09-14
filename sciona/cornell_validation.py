"""Source Cornell validation: per-record F1 after max pooling two clips."""
import numpy as np


def clip_f1(probabilities,targets):
    pred=np.asarray(probabilities);truth=np.asarray(targets)
    if pred.ndim!=3 or pred.shape[1:]!=(2,264) or pred.shape[0]==0 or truth.shape!=pred.shape:
        raise ValueError('Expected nonempty records with two clips and264classes')
    if pred.dtype.kind not in 'fiu' or truth.dtype.kind not in 'fiu':
        raise ValueError('Expected numeric predictions and targets')
    # Float32 attention-weight summation can exceed one by a few ULPs.
    tolerance=8*np.finfo(np.float32).eps
    if not np.isfinite(pred).all() or np.any((pred < -tolerance)|(pred > 1+tolerance)):
        raise ValueError('Expected finite probabilities')
    if not np.isin(truth,[0,1]).all() or not np.array_equal(truth[:,0],truth[:,1]):
        raise ValueError('Expected repeated binary record targets')
    predicted=pred.max(axis=1)>=.5;actual=truth[:,0].astype(bool)
    tp=(predicted&actual).sum(axis=1)
    size=predicted.sum(axis=1)+actual.sum(axis=1)
    scores=np.ones(len(pred))
    np.divide(2*tp,size,out=scores,where=size!=0)
    return float(scores.mean())
