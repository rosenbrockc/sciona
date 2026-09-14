"""Independent RankGauss and swap corruption for the pending Porto realization.

RankGauss uses the entire declared population. Average ties and midpoint rank
positions provide finite endpoints; these choices are explicit, not a claim
about the author's unpublished implementation. No out-of-sample transform.
"""
import numpy as np
from scipy.special import erfinv
from scipy.stats import rankdata


def _matrix(values):
    try:array=np.asarray(values)
    except ValueError:raise ValueError('Rectangular finite real matrix required') from None
    if array.ndim!=2 or not all(array.shape) or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('Nonempty finite real matrix required')
    return array.astype(np.float64,copy=True)


def rank_gauss_population(values,*,binary_columns):
    x=_matrix(values)
    if type(binary_columns) is not list or any(type(i) is not int or not 0<=i<x.shape[1] for i in binary_columns) or len(set(binary_columns))!=len(binary_columns):
        raise ValueError('Explicit distinct binary column indices required')
    if binary_columns and not np.isin(x[:,binary_columns],[0.,1.]).all():raise ValueError('Declared binary features must contain only zero and one')
    binary=set(binary_columns);result=x.copy()
    for column in range(x.shape[1]):
        if column in binary:continue
        positions=(rankdata(x[:,column],method='average')-.5)/len(x)
        transformed=erfinv(2*positions-1)
        result[:,column]=transformed-transformed.mean()
    if not np.isfinite(result).all():raise ValueError('Nonfinite RankGauss output')
    return result


def swap_noise(values,reference,*,probability,rng):
    """Independently replace selected cells with same-column reference draws.

The caller owns the generator and the donor population. Self-donors are allowed;
selection probability does not imply that the numerical value changes.
"""
    x=_matrix(values);donors=_matrix(reference)
    if x.shape[1]!=donors.shape[1]:raise ValueError('Donor and input widths differ')
    if type(probability) not in (int,float) or not np.isfinite(probability) or not 0<=probability<=1 or not isinstance(rng,np.random.Generator):raise ValueError('Probability and explicit Generator required')
    if probability==0:return x
    mask=rng.random(x.shape)<probability
    rows=rng.integers(len(donors),size=x.shape)
    replacement=donors[rows,np.arange(x.shape[1])[None,:]]
    return np.where(mask,replacement,x)
