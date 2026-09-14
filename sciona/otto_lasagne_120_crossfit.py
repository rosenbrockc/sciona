"""Five-fold 120-run native Lasagne neural bagging and full-training query refits.

This boundary accepts raw nonnegative values. Each native fit applies the
raw/log representation and fits its standardization on references only.
"""
import numpy as np
from sciona.otto_lasagne_120 import fit_predict


def crossfit_lasagne(training, labels, folds, training_ids, query, query_ids, *, variant, seed, controls):
    x=np.asarray(training,dtype=np.float64);q=np.asarray(query,dtype=np.float64)
    y=np.asarray(labels);f=np.asarray(folds)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or x.shape[1]<1 or q.shape[1]!=x.shape[1]:
        raise ValueError('Aligned nonempty feature populations required')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or (x<0).any() or (q<0).any():
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
        oof[held]=fit_predict(x[~held],y[~held],x[held],variant=variant,seed=seed,controls=controls)
        coverage[held]+=1
    if not np.all(coverage==1):raise ValueError('Incomplete out-of-fold coverage')
    predicted=fit_predict(x,y,q,variant=variant,seed=seed,controls=controls)
    return dict(oof=oof,query=predicted,folds=5,oof_rows=len(x),query_rows=len(q),bag_runs=120,total_model_fits=720)
