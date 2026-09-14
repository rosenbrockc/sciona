"""Complete generic text-pair ensemble and held-out F1 threshold lifecycle."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sciona.text_pair_features import PairFeatures,normalize_pairs


def _keys(pairs):return frozenset(tuple(sorted(p)) for p in normalize_pairs(pairs))


def _labels(labels,rows):
    y=np.asarray(labels)
    if y.shape!=(rows,) or y.dtype.kind not in 'biu' or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Expected aligned binary labels with both classes')
    return y.astype(np.int64)


@dataclass(frozen=True)
class FittedPairs:
    features: object
    linear: object
    forest: object
    training_keys: frozenset


@dataclass(frozen=True)
class CalibratedPairs:
    fitted: FittedPairs
    threshold: float
    calibration_f1: float
    calibration_keys: frozenset


def fit_ensemble(pairs,labels,*,components=2,seed=42,trees=64,max_depth=6,min_leaf=1):
    normalized=normalize_pairs(pairs);y=_labels(labels,len(normalized))
    keys=_keys(pairs)
    if len(keys)!=len(pairs):raise ValueError('Duplicate normalized unordered training pair')
    for value in (trees,max_depth,min_leaf):
        if type(value) is not int or value<1:raise ValueError('Invalid forest control')
    features=PairFeatures().fit(pairs,components=components,seed=seed)
    x=features.transform_training()
    linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
    forest=RandomForestClassifier(n_estimators=trees,max_depth=max_depth,min_samples_leaf=min_leaf,random_state=seed,n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        linear.fit(x,y)
    forest.fit(x,y)
    return FittedPairs(features,linear,forest,keys)


def ensemble_scores(fitted,pairs):
    x=fitted.features.transform(pairs)
    # Both fitted classifiers have explicit binary class order [0, 1].
    scores=(fitted.linear.predict_proba(x)[:,1]+fitted.forest.predict_proba(x)[:,1])/2
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Invalid ensemble scores')
    return scores


def best_f1_threshold(scores,labels):
    values=np.asarray(scores)
    if values.ndim!=1 or not len(values) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('Expected finite scores in [0,1]')
    y=_labels(labels,len(values));best=(-1.,-1.)
    for threshold in np.unique(values):
        predicted=values>=threshold
        true_positive=np.count_nonzero(predicted & (y==1))
        false_positive=np.count_nonzero(predicted & (y==0))
        false_negative=np.count_nonzero(~predicted & (y==1))
        score=2*true_positive/(2*true_positive+false_positive+false_negative)
        best=max(best,(float(score),float(threshold)))
    return best[1],best[0]


def calibrate_threshold(fitted,pairs,labels):
    keys=_keys(pairs)
    if len(keys)!=len(pairs) or keys & fitted.training_keys:
        raise ValueError('Calibration pairs must be distinct from training')
    y=_labels(labels,len(pairs))
    threshold,score=best_f1_threshold(ensemble_scores(fitted,pairs),y)
    return CalibratedPairs(fitted,threshold,score,keys)


def predict_pairs(calibrated,pairs):
    keys=_keys(pairs)
    if keys & (calibrated.fitted.training_keys | calibrated.calibration_keys):
        raise ValueError('Prediction pairs must be disjoint from fitted populations')
    scores=ensemble_scores(calibrated.fitted,pairs)
    return dict(scores=scores.tolist(),matches=(scores>=calibrated.threshold).astype(int).tolist(),
        threshold=calibrated.threshold,calibration_f1=calibrated.calibration_f1)
