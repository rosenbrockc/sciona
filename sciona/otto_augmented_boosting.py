"""Single multiclass augmented-feature booster for Otto model 15.

Independent explicit controls; historical parameters and seed schedule unknown.
"""
import numpy as np
import xgboost as xgb
from sciona.otto_preprocessing import representation
from sciona.otto_augmented_features import fit_augmentation


def _fit_encoded(reference,labels,query,*,seed,controls):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned seed required')
    if type(controls) is not dict or set(controls)!={'rounds','max_depth','eta','subsample','colsample_bytree'}:
        raise ValueError('Explicit boosting controls required')
    for key in ('rounds','max_depth'):
        if type(controls[key]) is not int or controls[key]<1:raise ValueError('Positive integer boosting controls required')
    for key in ('eta','subsample','colsample_bytree'):
        value=controls[key]
        if type(value) not in (int,float) or not 0<value<=1:raise ValueError('Boosting fractions must be in (0,1]')
    x=representation(reference,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class reference labels required')
    params={k:v for k,v in controls.items() if k!='rounds'}
    params.update(objective='multi:softprob',num_class=9,eval_metric='mlogloss',tree_method='hist',nthread=1,seed=seed,verbosity=0)
    query_matrix=xgb.DMatrix(q,missing=np.nan,nthread=1)
    training_matrix=xgb.DMatrix(x,label=y,missing=np.nan,nthread=1)
    model=xgb.train(params,training_matrix,num_boost_round=controls['rounds'])
    scores=model.predict(query_matrix)
    if scores.shape!=(len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(axis=1),1.):
        raise ValueError('Invalid multiclass probabilities')
    return scores.astype(np.float64)


def fit_predict(reference,labels,query,*,seed,controls):
    if type(controls) is not dict or set(controls)!={'augmentation','boosting'}:
        raise ValueError('Explicit augmentation and boosting controls required')
    options=controls['augmentation']
    if type(options) is not dict or set(options)!={'cluster_counts','ddof','n_init','max_iter'}:
        raise ValueError('Invalid augmentation controls')
    augmentation=fit_augmentation(reference,seed=seed,**options)
    return _fit_encoded(augmentation.transform(reference),labels,augmentation.transform(query),seed=seed,controls=controls['boosting'])
