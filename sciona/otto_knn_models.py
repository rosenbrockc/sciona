"""Ten documented raw-feature KNN branches with unsmoothed class frequencies."""
import numpy as np
from scipy.spatial.distance import cdist

NEIGHBORS = (2,4,8,16,32,64,128,256,512,1024)


def predict(reference, labels, query, *, metric, chunk_size=128):
    x=np.asarray(reference,dtype=np.float64);q=np.asarray(query,dtype=np.float64);y=np.asarray(labels)
    if metric not in ('euclidean','cityblock','braycurtis'):
        raise ValueError('Explicit supported neighbor metric required')
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Positive integer chunk size required')
    if x.ndim!=2 or len(x)<1024 or q.ndim!=2 or len(q)<1 or x.shape[1]<1 or x.shape[1]!=q.shape[1]:
        raise ValueError('Aligned matrices and at least 1024 references required')
    if not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Finite features required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class integer labels required')
    if metric=='braycurtis' and ((x<0).any() or (q<0).any()):
        raise ValueError('Nonnegative Bray-Curtis features required')
    result=np.empty((len(q),len(NEIGHBORS),9))
    for start in range(0,len(q),chunk_size):
        batch=q[start:start+chunk_size]
        distances=cdist(batch,x,metric=metric)
        if metric=='braycurtis':
            distances[np.ix_(np.all(batch==0,axis=1),np.all(x==0,axis=1))]=0.
        if not np.isfinite(distances).all():raise ValueError('Nonfinite distances')
        # Equal-distance references use their supplied order. Preserve it across
        # folds; no claim of matching the historical library's tie behavior.
        order=np.argsort(distances,axis=1,kind='stable')[:,:1024]
        nearest=y[order]
        for column,k in enumerate(NEIGHBORS):
            for label in range(9):
                result[start:start+len(batch),column,label]=np.mean(nearest[:,:k]==label,axis=1)
    return result


def crossfit(training,labels,folds,training_ids,query,query_ids,*,metric,chunk_size=128):
    x=np.asarray(training,dtype=np.float64);q=np.asarray(query,dtype=np.float64)
    y=np.asarray(labels);f=np.asarray(folds)
    if x.ndim!=2 or q.ndim!=2 or not len(q) or x.shape[1]<1 or x.shape[1]!=q.shape[1]:
        raise ValueError('Aligned feature populations required')
    if y.shape!=(len(x),) or f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Five aligned integer folds required')
    identities=set()
    for ids,size in [(training_ids,len(x)),(query_ids,len(q))]:
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque identities required')
        for identity in ids:
            if identity in identities:raise ValueError('Duplicate or overlapping identities')
            identities.add(identity)
    for fold in range(5):
        if np.sum(f!=fold)<1024:raise ValueError('Each fitting fold needs 1024 references')
        if set(y[f!=fold].tolist())!=set(range(9)):raise ValueError('Each fitting fold needs every class')
    output=np.empty((len(x),10,9));coverage=np.zeros(len(x),dtype=int)
    for fold in range(5):
        held=f==fold
        output[held]=predict(x[~held],y[~held],x[held],metric=metric,chunk_size=chunk_size)
        coverage[held]+=1
    if not np.all(coverage==1):raise ValueError('Incomplete OOF coverage')
    return dict(oof=output,query=predict(x,y,q,metric=metric,chunk_size=chunk_size),neighbors=list(NEIGHBORS))
