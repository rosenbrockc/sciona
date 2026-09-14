"""Explicit numeric representations described in the Otto winner write-up.

The caller chooses fitting population and standard-deviation convention. These
choices are not inferred as historical facts from the incomplete source recipe.
"""
from dataclasses import dataclass
import numpy as np


def representation(values, *, kind):
    try: x=np.asarray(values,dtype=np.float64)
    except (TypeError,ValueError,OverflowError):raise ValueError('Finite nonnegative feature matrix required') from None
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Finite nonnegative feature matrix required')
    if kind=='raw':return x.copy()
    if kind=='log1p':return np.log1p(x)
    if kind=='sqrt_offset':return np.sqrt(x+3/8)
    if kind=='raw_zero':return np.concatenate((x,(x==0).astype(float)),axis=1)
    if kind=='raw_zero_log':return np.concatenate((x,(x==0).astype(float),np.log1p(x)),axis=1)
    raise ValueError('Unsupported feature representation')


@dataclass(frozen=True,repr=False)
class Scaling:
    kind: str
    mean: np.ndarray
    scale: np.ndarray
    ddof: int

    def transform(self, values):
        x=representation(values,kind=self.kind)
        if x.shape[1]!=len(self.mean):raise ValueError('Feature width differs from fitted scaler')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            result=(x-self.mean)/self.scale
        if not np.isfinite(result).all():raise ValueError('Nonfinite standardized features')
        return result


def fit_scaling(reference, *, kind, ddof):
    if kind not in ('raw','log1p') or type(ddof) is not int or ddof not in (0,1):
        raise ValueError('Explicit raw/log1p scaling and ddof zero or one required')
    x=representation(reference,kind=kind)
    if len(x)<=ddof:raise ValueError('Insufficient fitting rows for scaling convention')
    with np.errstate(over='ignore',invalid='ignore'):
        mean=x.mean(axis=0);scale=x.std(axis=0,ddof=ddof)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite fitted scaling statistics')
    scale[scale==0]=1.
    mean.setflags(write=False);scale.setflags(write=False)
    return Scaling(kind,mean,scale,ddof)
