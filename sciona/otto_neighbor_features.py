"""Class-specific nearest-distance features for explicit fitting populations.

Callers must pass the excluding-fold reference population for OOF rows. The
metric is explicit because the winner description uses several distance metrics.
"""
import numpy as np
from scipy.spatial.distance import cdist


def class_distances(reference, labels, query, *, metric, chunk_size=128):
    if metric not in ('euclidean','cityblock','braycurtis'):
        raise ValueError('Unsupported neighbor metric')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Positive integer query chunk size required')
    x=np.asarray(reference,dtype=np.float64);q=np.asarray(query,dtype=np.float64)
    y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]<1 or q.shape[1]!=x.shape[1] or len(q)<1:
        raise ValueError('Aligned nonempty feature matrices required')
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('Finite feature matrices required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class integer labels required')
    groups=[x[y==label] for label in range(9)]
    if any(len(group)<4 for group in groups):
        raise ValueError('Four fitting references per class required')
    if metric=='braycurtis' and ((x<0).any() or (q<0).any()):
        raise ValueError('Bray-Curtis branch requires nonnegative features')
    output=np.empty((len(q),3,9),dtype=np.float64)
    for start in range(0,len(q),chunk_size):
        batch=q[start:start+chunk_size]
        for label,group in enumerate(groups):
            distances=cdist(batch,group,metric=metric)
            if metric=='braycurtis':
                # Explicit identical-zero convention, avoiding SciPy's 0/0.
                distances[np.ix_(np.all(batch==0,axis=1),np.all(group==0,axis=1))]=0.
            if not np.isfinite(distances).all():
                raise ValueError('Nonfinite neighbor distances')
            nearest=np.sort(np.partition(distances,3,axis=1)[:,:4],axis=1)
            for column,k in enumerate((1,2,4)):
                output[start:start+len(batch),column,label]=nearest[:,:k].sum(axis=1)
    return output
