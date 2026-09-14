"""Independent weighted-Gini and top-four-percent Amex checkpoint metric."""
import numpy as np


def score(labels,probabilities):
    y=np.asarray(labels);p=np.asarray(probabilities)
    if y.ndim!=1 or not len(y) or y.dtype.kind not in 'iu' or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:raise ValueError('Both integer binary classes required')
    if p.shape!=y.shape or p.dtype.kind not in 'iuf' or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Aligned finite probabilities required')
    order=np.argsort(p)[::-1];ranked=y[order];weights=np.where(ranked==0,20,1)
    selected=np.cumsum(weights)<=int(.04*weights.sum())
    top=ranked[selected].sum()/ranked.sum()
    def gini(order):
        a=y[order];w=np.where(a==0,20,1)
        uniform=np.cumsum(w/w.sum());lorenz=np.cumsum(a*w)/(a*w).sum()
        return np.sum((lorenz-uniform)*w)
    perfect=gini(np.argsort(y)[::-1])
    if perfect==0:raise ValueError('Undefined normalized Gini')
    return float(.5*(gini(order)/perfect+top))
