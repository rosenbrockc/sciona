"""Explicit-reference TF-IDF representation for Otto neighbor features.

Weighting choices are supplied, not inferred from the historical recipe.
"""
from dataclasses import dataclass
import numpy as np
from sciona.otto_preprocessing import representation


@dataclass(frozen=True,repr=False)
class Weighting:
    idf: np.ndarray
    norm: str
    sublinear_tf: bool

    def transform(self, values):
        x=representation(values,kind='raw')
        if x.shape[1]!=len(self.idf):raise ValueError('Feature width differs from fitted IDF')
        if self.sublinear_tf:
            if ((x>0)&(x<1)).any():raise ValueError('Sublinear TF requires nonzero counts at least one')
            mask=x>0
            x[mask]=1.+np.log(x[mask])
        with np.errstate(over='ignore',invalid='ignore'):
            weighted=x*self.idf
        if not np.isfinite(weighted).all():raise ValueError('Nonfinite weighted features')
        if self.norm!='none':
            # Scale before the norm to avoid squaring very large finite values.
            maxima=weighted.max(axis=1,keepdims=True)
            scaled=np.divide(weighted,maxima,out=np.zeros_like(weighted),where=maxima!=0)
            denominator=scaled.sum(axis=1,keepdims=True) if self.norm=='l1' else np.sqrt((scaled*scaled).sum(axis=1,keepdims=True))
            weighted=np.divide(scaled,denominator,out=np.zeros_like(scaled),where=denominator!=0)
        return weighted


def fit_weighting(reference,*,smooth_idf,sublinear_tf,norm):
    if type(smooth_idf) is not bool or type(sublinear_tf) is not bool or norm not in ('l1','l2','none'):
        raise ValueError('Explicit IDF, TF and normalization choices required')
    x=representation(reference,kind='raw')
    if sublinear_tf and ((x>0)&(x<1)).any():raise ValueError('Sublinear TF requires nonzero counts at least one')
    frequency=(x>0).sum(axis=0)
    if not smooth_idf and (frequency==0).any():
        raise ValueError('Unsmoothed IDF undefined for absent fitting features')
    smoothing=int(smooth_idf)
    idf=np.log((len(x)+smoothing)/(frequency+smoothing))+1.
    idf.setflags(write=False)
    return Weighting(idf,norm,sublinear_tf)
