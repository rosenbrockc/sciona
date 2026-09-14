"""Explicit causal isolation-forest branch of the generic NAB intake.

This is a new generic pipeline, not Numenta HTM or a historical winning detector.
Each model fits past windows only; rolling score statistics exclude the current
score. Scoring labels are never passed into the detector.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

_FIELDS={'window_size','fit_history','min_fit_windows','threshold_history','min_threshold_scores','n_estimators','max_samples','seed','std_ddof'}


def detect(values,config):
    x=np.asarray(values)
    if x.ndim!=1 or x.size<1 or x.dtype.kind not in 'fi' or not np.isfinite(x).all():raise ValueError('Finite univariate observations required')
    if not isinstance(config,dict) or set(config)!=_FIELDS or any(type(v) is not int for v in config.values()):raise ValueError('Explicit integer detector controls required')
    for name in _FIELDS-{'seed','std_ddof'}:
        if config[name]<1:raise ValueError('Positive detector controls required')
    if not 0<=config['seed']<2**32 or config['std_ddof'] not in (0,1):raise ValueError('Invalid seed or variance convention')
    if not 2<=config['min_fit_windows']<=config['fit_history'] or not 2<=config['min_threshold_scores']<=config['threshold_history']:
        raise ValueError('At least two history samples required')
    if config['max_samples']<2:raise ValueError('At least two forest samples required')
    if np.max(abs(x))>np.finfo(np.float32).max:raise ValueError('Observations exceed forest numeric range')
    window=config['window_size'];scores=[None]*len(x);thresholds=[None]*len(x);flags=[False]*len(x);prior_scores=[];fits=0
    for i in range(window-1,len(x)):
        first=max(window-1,i-config['fit_history'])
        if i-first<config['min_fit_windows']:continue
        training=np.stack([x[j-window+1:j+1] for j in range(first,i)])
        model=IsolationForest(n_estimators=config['n_estimators'],max_samples=min(config['max_samples'],len(training)),
            contamination='auto',max_features=1.,bootstrap=False,n_jobs=1,random_state=(config['seed']+i)%(2**32))
        model.fit(training);fits+=1
        score=float(-model.score_samples(x[i-window+1:i+1][None,:])[0])
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid forest anomaly score')
        history=np.asarray(prior_scores[-config['threshold_history']:])
        if len(history)>=config['min_threshold_scores']:
            threshold=float(history.mean()+3*history.std(ddof=config['std_ddof']))
            thresholds[i]=threshold;flags[i]=score>threshold
        scores[i]=score;prior_scores.append(score)
    return dict(anomaly_scores=scores,thresholds=thresholds,detections=flags,models_fitted=fits)
