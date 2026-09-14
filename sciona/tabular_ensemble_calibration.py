"""Separate-population regularized sigmoid calibration for tabular stacking."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sciona.tabular_ensemble_training import binary_labels,group_ids
from sciona.tabular_ensemble_features import validate_table


def _logits(probabilities):
    values=np.asarray(probabilities,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Expected finite binary probabilities')
    clipped=np.clip(values,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return (np.log(clipped)-np.log1p(-clipped)).reshape(-1,1)


@dataclass(frozen=True)
class CalibratedStack:
    stacked: object
    sigmoid: object
    calibration_groups: frozenset


def calibrate_stack(stacked,numeric,categorical,labels,groups):
    """Fit a C=1 logistic sigmoid on held-out stack-score logits only.

The base ensemble, feature transforms and stacker remain fixed. This is a
calibration procedure, not a guarantee of empirical calibration or improved
accuracy. Group identities are supplied by the caller and must be meaningful.
"""
    validate_table(numeric,categorical)
    y=binary_labels(labels,len(numeric));selected=group_ids(groups,len(numeric))
    if selected & stacked.training_groups:raise ValueError('Calibration groups overlap training')
    scores=stacked.predict(numeric,categorical)
    sigmoid=LogisticRegression(C=1.,max_iter=1000,random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        sigmoid.fit(_logits(scores),y)
    return CalibratedStack(stacked,sigmoid,selected)


def predict_calibrated(calibrated,numeric,categorical,groups):
    validate_table(numeric,categorical)
    selected=group_ids(groups,len(numeric))
    if selected & (calibrated.stacked.training_groups | calibrated.calibration_groups):
        raise ValueError('Prediction groups overlap fitted populations')
    probabilities=calibrated.sigmoid.predict_proba(_logits(calibrated.stacked.predict(numeric,categorical)))[:,1]
    if not np.isfinite(probabilities).all():raise ValueError('Nonfinite calibrated probabilities')
    return dict(probabilities=probabilities.tolist(),classes=(probabilities>=.5).astype(int).tolist())
