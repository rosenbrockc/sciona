"""Independent one-vs-rest XGBoost realization for Otto source model 14.

Zero counts are computed before raw zeros become missing. Count values remain
observed, including zero. Historical treatment of that appended zero is unknown.
"""
import numpy as np
import xgboost as xgb
from sciona.otto_preprocessing import representation


def features(values):
    raw=representation(values,kind='raw')
    zeros=(raw==0).sum(axis=1,keepdims=True).astype(float)
    raw[raw==0]=np.nan
    return np.concatenate((raw,zeros),axis=1)


def fit_predict(reference,labels,query,*,seed,controls):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned seed required')
    if type(controls) is not dict or set(controls)!={'rounds','max_depth','eta','subsample','colsample_bytree'}:
        raise ValueError('Explicit boosting controls required')
    for key in ('rounds','max_depth'):
        if type(controls[key]) is not int or controls[key]<1:raise ValueError('Positive integer boosting controls required')
    for key in ('eta','subsample','colsample_bytree'):
        value=controls[key]
        if type(value) not in (int,float) or not 0<value<=1:raise ValueError('Boosting fractions must be in (0,1]')
    x=features(reference);q=features(query);y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class reference labels required')
    params={k:v for k,v in controls.items() if k!='rounds'}
    params.update(objective='binary:logistic',eval_metric='logloss',tree_method='hist',nthread=1,seed=seed,verbosity=0)
    query_matrix=xgb.DMatrix(q,missing=np.nan,nthread=1)
    result=[]
    for label in range(9):
        training_matrix=xgb.DMatrix(x,label=(y==label).astype(int),missing=np.nan,nthread=1)
        model=xgb.train(params,training_matrix,num_boost_round=controls['rounds'])
        result.append(model.predict(query_matrix))
    scores=np.column_stack(result).astype(np.float64)
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Invalid binary class scores')
    return scores
