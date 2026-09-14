"""Independent grouped MCC threshold search and legacy random fold assignment."""
import numpy as np


def binary_labels(values):
    y=np.asarray(values)
    if y.ndim!=1 or not len(y) or y.dtype.kind not in 'iu' or not np.isin(y,[0,1]).all():raise ValueError('Nonempty integer binary labels required')
    return y.astype(np.int64)


def threshold(labels,probabilities):
    y=binary_labels(labels);p=np.asarray(probabilities)
    if p.shape!=y.shape or p.dtype.kind not in 'iuf' or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Aligned finite probabilities required')
    order=np.argsort(p);v=p[order];positive=y[order]
    starts=np.r_[0,np.flatnonzero(np.diff(v)!=0)+1]
    prior_positive=np.r_[0,np.cumsum(positive)][starts].astype(float)
    fn=prior_positive;tn=starts-fn;tp=float(y.sum())-fn;fp=float(len(y)-y.sum())-tn
    denominator=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    scores=np.divide(tp*tn-fp*fn,np.sqrt(denominator),out=np.zeros(len(starts)),where=denominator!=0)
    best=np.flatnonzero(scores==scores.max())[-1]
    return dict(threshold=float(v[starts[best]]),mcc=float(scores[best]))


def repeated_folds(labels,*,seed,repetitions):
    y=binary_labels(labels)
    if type(seed) is not int or type(repetitions) is not int or repetitions<1 or seed<0 or seed+repetitions-1>=2**32:raise ValueError('Bounded seed and positive repetitions required')
    if len(np.unique(y))!=2:raise ValueError('Both measurement classes required')
    plans=[]
    for iteration in range(repetitions):
        rng=np.random.RandomState(seed+iteration);assigned=np.empty(len(y),dtype=int)
        for label in (1,0):
            mask=y==label;assigned[mask]=rng.randint(0,5,size=mask.sum())
        for fold in range(5):
            validation=np.flatnonzero(assigned==fold);test=np.flatnonzero(assigned==(fold+1)%5)
            training=np.flatnonzero(~np.isin(assigned,[fold,(fold+1)%5]))
            if not len(validation) or not len(test) or len(np.unique(y[training]))!=2:raise ValueError('Random partition cannot train complete fold lifecycle')
            plans.append((training,validation,test))
    return plans
