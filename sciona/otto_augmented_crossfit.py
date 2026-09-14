"""Five-fold augmented multiclass boosting and full-training query refits.

This boundary accepts raw features. Scaling and clustering are fitted inside
each excluding-fold model, followed by a full-training query refit.
"""
import numpy as np
from sciona.otto_augmented_boosting import fit_predict


def crossfit_boosting(training, labels, folds, training_ids, query, query_ids, *, seed, controls):
    x=np.asarray(training,dtype=np.float64);q=np.asarray(query,dtype=np.float64)
    y=np.asarray(labels);f=np.asarray(folds)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or x.shape[1]<1 or q.shape[1]!=x.shape[1]:
        raise ValueError('Aligned nonempty feature populations required')
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('Finite features required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Nine integer classes required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Exactly five aligned folds required')
    identities=set()
    for ids,count in [(training_ids,len(x)),(query_ids,len(q))]:
        if type(ids) is not list or len(ids)!=count or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque row identities required')
        for identity in ids:
            if identity in identities:raise ValueError('Repeated or overlapping row identities')
            identities.add(identity)
    # Reject deficient folds before any model fitting.
    for fold in range(5):
        if set(y[f!=fold].tolist())!=set(range(9)):
            raise ValueError('Every fitting fold requires every class')
    oof=np.empty((len(x),9),dtype=np.float64)
    coverage=np.zeros(len(x),dtype=np.int64)
    for fold in range(5):
        held=f==fold
        oof[held]=fit_predict(x[~held],y[~held],x[held],seed=seed,controls=controls)
        coverage[held]+=1
    if not np.all(coverage==1):raise ValueError('Incomplete out-of-fold coverage')
    predicted=fit_predict(x,y,q,seed=seed,controls=controls)
    return dict(oof=oof,query=predicted,folds=5,oof_rows=len(x),query_rows=len(q),model_fits=6)
