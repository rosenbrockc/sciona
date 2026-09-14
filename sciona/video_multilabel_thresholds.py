"""Independent label-wise held-out F1 thresholds with explicit tie behavior."""
import numpy as np


def score_matrix(scores):
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim!=2 or not all(values.shape) or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Finite nonempty probability matrix required')
    return values


def label_matrix(labels,shape):
    if type(labels) is not list or any(type(row) is not list or any(type(v) is not int or v not in (0,1) for v in row) for row in labels):
        raise ValueError('Integer binary label matrix required')
    result=np.asarray(labels,dtype=np.int64)
    if result.shape!=shape:raise ValueError('Label matrix shape mismatch')
    if any(set(column)!={0,1} for column in result.T):raise ValueError('Each label requires both classes')
    return result


def calibrate(scores,labels):
    values=score_matrix(scores);targets=label_matrix(labels,values.shape)
    thresholds=[]
    for column,target in zip(values.T,targets.T):
        best_score=-1.;best_threshold=None
        # Every distinct >= decision, including predicting none. Ascending
        # iteration with >= updates implements the highest-threshold tie rule.
        candidates=np.append(np.unique(column),np.nextafter(column.max(),np.inf))
        for threshold in candidates:
            predicted=column>=threshold
            tp=int(np.count_nonzero(predicted & (target==1)))
            denominator=int(predicted.sum()+target.sum())
            f1=2*tp/denominator if denominator else 0.
            if f1>=best_score:best_score=f1;best_threshold=float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def decisions(scores,thresholds):
    values=score_matrix(scores);threshold=np.asarray(thresholds,dtype=np.float64)
    if threshold.shape!=(values.shape[1],) or not np.isfinite(threshold).all():raise ValueError('Aligned finite thresholds required')
    return (values>=threshold).astype(int).tolist()
