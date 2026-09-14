"""Independent hierarchy rescaling, expanding volatility and revenue weights."""
import numpy as np


def level_factors(levels):
    """Compute level population / base population before any base-level splits."""
    levels=np.asarray(levels)
    if levels.ndim!=1 or levels.dtype.kind not in 'iu' or not (levels==12).any():
        raise ValueError('Integer levels with a base population required')
    base=np.sum(levels==12)
    return np.array([np.sum(levels==value)/base for value in levels],dtype=float)


def statistics(history,revenue,levels):
    history,revenue,levels=np.asarray(history),np.asarray(revenue),np.asarray(levels)
    if (history.ndim!=2 or not all(history.shape) or revenue.shape!=history.shape
            or history.dtype.kind not in 'iuf' or revenue.dtype.kind not in 'iuf'
            or np.isinf(history).any() or np.isinf(revenue).any() or (history<0).any() or (revenue<0).any()
            or levels.shape!=(history.shape[1],) or levels.dtype.kind not in 'iu'):
        raise ValueError('Aligned nonnegative-or-missing histories/revenue and integer levels required')
    history,revenue=history.astype(float),revenue.astype(float)
    observations=np.cumsum(~np.isnan(history),axis=0)
    if observations.max()>np.iinfo(np.int16).max:raise ValueError('Observation count exceeds source storage range')
    differences=np.full(history.shape,np.nan);differences[1:]=np.abs(np.diff(history,axis=0))
    counts=np.cumsum(~np.isnan(differences),axis=0)
    totals=np.cumsum(np.nan_to_num(differences,nan=0),axis=0)
    volatility=np.divide(totals,counts,out=np.full(history.shape,np.nan),where=counts>0)
    trailing=np.full(history.shape,np.nan)
    for end in range(len(history)):
        sample=revenue[max(0,end-27):end+1];valid=~np.isnan(sample).all(axis=0)
        trailing[end,valid]=np.nansum(sample[:,valid],axis=0)
    weights=np.full(history.shape,np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        for level in np.unique(levels):
            selected=levels==level
            denominator=np.nansum(trailing[:,selected],axis=1)
            weights[:,selected]=trailing[:,selected]/denominator[:,None]
    return dict(observations=observations.astype(np.int16),volatility=volatility,weights=weights)
