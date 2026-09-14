"""Independent grouped rank and timestamp-window operations for Amex features."""
import numpy as np
from scipy.stats import rankdata
from sciona.amex_numeric import _matrix


def _groups(groups,n):
    g=np.asarray(groups)
    if g.shape!=(n,) or g.dtype.kind not in 'iu':raise ValueError('Aligned integer grouping codes required')
    return g


def percentile_ranks(values,groups):
    x=_matrix(values);g=_groups(groups,len(x));out=np.full_like(x,np.nan)
    for group in np.unique(g):
        rows=np.flatnonzero(g==group)
        for col in range(x.shape[1]):
            valid=rows[~np.isnan(x[rows,col])]
            if len(valid):out[valid,col]=rankdata(x[valid,col],method='average')/len(valid)
    return out


def latest_window(times,groups,*,count):
    t=np.asarray(times)
    if t.ndim!=1 or not len(t) or t.dtype.kind not in 'iuf' or np.isinf(t).any():raise ValueError('Numeric time ordering with optional NaN required')
    g=_groups(groups,len(t))
    if type(count) is not int or count<1:raise ValueError('Positive window count required')
    selected=np.zeros(len(t),dtype=bool)
    for group in np.unique(g):
        rows=np.flatnonzero((g==group)&~np.isnan(t))
        selected[rows]=rankdata(-t[rows].astype(float),method='average')<=count
    return selected
