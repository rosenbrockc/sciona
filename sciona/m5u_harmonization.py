"""Independent final level adjustment and restoration of hierarchy units."""
import numpy as np


def restore(predictions, levels, factors, quantiles, *, adjustment=.7):
    """Accept quantile x horizon x series values; return unrounded native units.

    Each series retains its level identifier. Only levels 1 through 9 share
    the median-total adjustment; every level has its input multiplier undone.
    """
    values=np.asarray(predictions)
    levels=np.asarray(levels)
    factors=np.asarray(factors)
    if (values.ndim!=3 or 0 in values.shape or values.dtype.kind not in 'iuf'
            or not np.isfinite(values).all() or levels.shape!=(values.shape[2],)
            or levels.dtype.kind not in 'iu' or (levels<1).any()
            or factors.shape!=levels.shape or factors.dtype.kind not in 'iuf'
            or not np.isfinite(factors).all() or (factors<=0).any()
            or not isinstance(quantiles,(list,tuple)) or len(quantiles)!=values.shape[0]
            or len(set(quantiles))!=len(quantiles) or .5 not in quantiles
            or any(isinstance(q,bool) or not isinstance(q,(int,float)) or not 0<q<1 for q in quantiles)
            or isinstance(adjustment,bool) or not isinstance(adjustment,(int,float))
            or not 0<=adjustment<=1):
        raise ValueError('Finite aligned forecasts, levels, positive factors and median quantile required')
    for level in np.unique(levels):
        if np.unique(factors[levels==level]).size!=1:
            raise ValueError('One multiplier per level required')
    result=values.astype(float,copy=True)/factors[None,None,:]
    selected=[level for level in np.unique(levels) if level<=9]
    if selected:
        median=quantiles.index(.5)
        totals=np.stack([result[median][:,levels==level].sum(axis=1) for level in selected])
        if not np.isfinite(totals).all() or (totals==0).any():
            raise ValueError('Finite nonzero median totals required for level adjustment')
        target=totals.mean(axis=0)
        for level,total in zip(selected,totals):
            correction=(1-adjustment)+adjustment*target/total
            result[:,:,levels==level]*=correction[None,:,None]
    if not np.isfinite(result).all():raise ValueError('Hierarchy restoration overflow')
    return result
