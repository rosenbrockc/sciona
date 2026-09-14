"""Discrete branch routing from Figure 2 of the Bengali solution paper.

Threshold selection, seen-class mapping, OOD scores and unseen-ensemble
predictions must be supplied by separately qualified pipeline stages.
"""
import math
import numpy as np


def route_predictions(ood_confidence,seen_predictions,unseen_predictions,*,seen_class_mask,threshold):
    confidence=np.asarray(ood_confidence)
    seen,unseen=np.asarray(seen_predictions),np.asarray(unseen_predictions)
    membership=np.asarray(seen_class_mask)
    if (isinstance(threshold,bool) or not isinstance(threshold,(int,float))
            or not math.isfinite(threshold) or not 0<=threshold<=1):
        raise ValueError('explicit finite OOD threshold in [0,1] required')
    if (confidence.ndim!=1 or not len(confidence) or confidence.dtype.kind not in 'fiu'
            or not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any()):
        raise ValueError('finite OOD confidence vector in [0,1] required')
    for prediction in (seen,unseen):
        if (prediction.shape!=confidence.shape or prediction.dtype.kind not in 'iu'
                or (prediction<0).any() or (prediction>=14784).any()):
            raise ValueError('aligned integer joint predictions within class bounds required')
    if membership.dtype!=np.bool_ or membership.shape!=(14784,):
        raise ValueError('explicit boolean joint-class membership map required')
    use_unseen=(confidence<threshold) & ~membership[unseen]
    return np.where(use_unseen,unseen,seen).astype(np.int64)
