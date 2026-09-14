"""Independent pinball and normalized weighted scaled pinball objectives."""
import numpy as np


def _loss(actual,predicted,quantile):
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    if (actual.shape!=predicted.shape or not actual.size or actual.dtype.kind not in 'iuf'
            or predicted.dtype.kind not in 'iuf' or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError('Aligned nonempty finite real arrays required')
    if isinstance(quantile,(bool,np.bool_)) or not isinstance(quantile,(int,float,np.integer,np.floating)) or not 0<quantile<1:
        raise ValueError('Quantile must lie strictly between zero and one')
    error=actual.astype(float)-predicted.astype(float)
    return np.maximum(quantile*error,(quantile-1)*error)


def pinball(actual,predicted,quantile=.5):
    return float(np.mean(_loss(actual,predicted,quantile)))


def weighted_scaled(actual,predicted,weights,scale,quantile=.5):
    loss=_loss(actual,predicted,quantile)
    weights,scale=np.asarray(weights),np.asarray(scale)
    if (weights.shape!=loss.shape or scale.shape!=loss.shape or weights.dtype.kind not in 'iuf'
            or scale.dtype.kind not in 'iuf' or not np.isfinite(weights).all() or not np.isfinite(scale).all()
            or (weights<0).any() or (scale<=0).any() or not (weights>0).any()):
        raise ValueError('Aligned nonnegative weights and positive scales required')
    return float(np.mean(weights*loss/scale)/np.mean(weights))
