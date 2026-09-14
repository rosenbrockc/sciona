"""Otto scaled-log and zero-indicator KNN families with explicit controls.

Historical k/metric assignments are unresolved. Fold-local standardization is
an independent leakage-controlled realization, not a recovered source setting.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sciona.otto_preprocessing import representation, fit_scaling


def predict(reference, labels, query, *, variant, neighbors, metric, ddof, chunk_size=128):
    x=representation(reference,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if type(neighbors) is not int or not 1<=neighbors<=len(x):
        raise ValueError('Neighbor count must fit reference population')
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Positive integer chunk size required')
    if variant not in ('scaled_log','raw_zero','raw_zero_log'):
        raise ValueError('Explicit supported KNN variant required')
    if metric not in ('euclidean','cityblock','braycurtis'):
        raise ValueError('Explicit supported metric required')
    if variant=='scaled_log':
        if metric=='braycurtis':raise ValueError('Bray-Curtis requires nonnegative representation')
        scaler=fit_scaling(x,kind='log1p',ddof=ddof)
        x=scaler.transform(x);q=scaler.transform(q)
    else:
        if ddof is not None:raise ValueError('Unscaled variants require ddof=None')
        x=representation(x,kind=variant);q=representation(q,kind=variant)
    scores=np.empty((len(q),9),dtype=float)
    for start in range(0,len(q),chunk_size):
        batch=q[start:start+chunk_size]
        distances=cdist(batch,x,metric=metric)
        if metric=='braycurtis':
            distances[np.ix_(np.all(batch==0,axis=1),np.all(x==0,axis=1))]=0.
        if not np.isfinite(distances).all():raise ValueError('Nonfinite neighbor distances')
        nearest=y[np.argsort(distances,axis=1,kind='stable')[:,:neighbors]]
        for label in range(9):scores[start:start+len(batch),label]=np.mean(nearest==label,axis=1)
    return scores


def crossfit(training,labels,folds,training_ids,query,query_ids,*,variant,neighbors,metric,ddof,chunk_size=128):
    x=representation(training,kind='raw');q=representation(query,kind='raw')
    y=np.asarray(labels);f=np.asarray(folds)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Exactly five aligned folds required')
    identities=set()
    for ids,count in [(training_ids,len(x)),(query_ids,len(q))]:
        if type(ids) is not list or len(ids)!=count or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque row identities required')
        for identity in ids:
            if identity in identities:raise ValueError('Repeated or overlapping row identities')
            identities.add(identity)
    for fold in range(5):
        if set(y[f!=fold].tolist())!=set(range(9)):
            raise ValueError('Each fitting fold needs every class')
        if type(neighbors) is not int or not 1<=neighbors<=np.sum(f!=fold):
            raise ValueError('Each fitting fold needs enough neighbors')
    controls=dict(variant=variant,neighbors=neighbors,metric=metric,ddof=ddof,chunk_size=chunk_size)
    oof=np.empty((len(x),9));coverage=np.zeros(len(x),dtype=int)
    for fold in range(5):
        held=f==fold
        oof[held]=predict(x[~held],y[~held],x[held],**controls)
        coverage[held]+=1
    if not np.all(coverage==1):raise ValueError('Incomplete OOF coverage')
    return dict(oof=oof,query=predict(x,y,q,**controls),folds=5,reference_fits=6)
