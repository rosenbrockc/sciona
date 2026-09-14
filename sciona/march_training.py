"""Chronological fitting, held-out sigmoid calibration and symmetric matchups."""
import warnings
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sciona.march_features import season_features,matchup_features


def fit_predict(games,training,calibration,prediction,*,feature_controls,regularization,max_iterations,tolerance,seed):
    if type(seed) is not int or not 0<=seed<2**32 or type(max_iterations) is not int or max_iterations<1:raise ValueError('Explicit optimizer controls required')
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in (regularization,tolerance)):raise ValueError('Positive finite fitting controls required')
    seasons=season_features(games,**feature_controls)
    groups=[];identities=set()
    for rows,labeled in [(training,True),(calibration,True),(prediction,False)]:
        if not isinstance(rows,list) or not rows:raise ValueError('Nonempty explicit chronological populations required')
        features=[];reverse=[];targets=[];years=set()
        for row in rows:
            fields={'season','order','a','b'}|({'outcome'} if labeled else set())
            if not isinstance(row,dict) or set(row)!=fields or any(type(v) is not int or v<0 for v in row.values()):raise ValueError('Exact integer matchup fields required')
            year=row['season'];years.add(year)
            if year not in seasons or row['order']<=seasons[year]['last_order']:raise ValueError('Regular-season features must precede every target matchup')
            if labeled:
                key=year,row['order']
                if key in identities:raise ValueError('Duplicate labeled game')
                identities.add(key)
                if row['outcome'] not in (0,1):raise ValueError('Binary outcome required')
                targets.append(row['outcome'])
            features.append(matchup_features(seasons,year,row['a'],row['b']))
            reverse.append(matchup_features(seasons,year,row['b'],row['a']))
        groups.append((np.asarray(features),np.asarray(reverse),np.asarray(targets),years))
    train,cal,pred=groups
    if not max(train[3])<min(cal[3]) or not max(cal[3])<min(pred[3]):raise ValueError('Training, calibration and prediction seasons must be strictly separated and ordered')
    if len(training)<2 or len(calibration)<2:raise ValueError('At least two fitting and calibration matchups required')
    def augment(group):return np.concatenate(group[:2]),np.concatenate([group[2],1-group[2]])
    tx,ty=augment(train);cx,cy=augment(cal)
    base=make_pipeline(StandardScaler(),LogisticRegression(C=regularization,max_iter=max_iterations,tol=tolerance,solver='lbfgs',random_state=seed))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        base.fit(tx,ty)
        # One explicit held-out population: FrozenEstimator never refits the base.
        calibration_indices=np.arange(len(cx))
        calibrated=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid',ensemble=False,n_jobs=1,
            cv=[(calibration_indices,calibration_indices)])
        calibrated.fit(cx,cy)
    forward=calibrated.predict_proba(pred[0])[:,1];reverse=calibrated.predict_proba(pred[1])[:,1]
    probabilities=.5*(forward+1-reverse)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():raise ValueError('Invalid predicted probability')
    return dict(probabilities=probabilities,training_rows=len(tx),calibration_rows=len(cx),predicted_matchups=len(probabilities))
