"""250 seeded multiclass XGBoost meta fits for Otto.

Independent explicit controls; historical parameters and seed schedule unknown.
"""
import numpy as np
import xgboost as xgb


def fit_bag(reference,labels,query,*,seed,controls):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned seed required')
    if type(controls) is not dict or set(controls)!={'rounds','max_depth','eta','subsample','colsample_bytree'}:
        raise ValueError('Explicit boosting controls required')
    for key in ('rounds','max_depth'):
        if type(controls[key]) is not int or controls[key]<1:raise ValueError('Positive integer boosting controls required')
    for key in ('eta','subsample','colsample_bytree'):
        value=controls[key]
        if type(value) not in (int,float) or not 0<value<=1:raise ValueError('Boosting fractions must be in (0,1]')
    x=np.asarray(reference,dtype=float);q=np.asarray(query,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or x.shape[1]<1 or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Finite nonempty meta matrices required')
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class reference labels required')
    params={k:v for k,v in controls.items() if k!='rounds'}
    params.update(objective='multi:softprob',num_class=9,eval_metric='mlogloss',tree_method='hist',nthread=1,seed=seed,verbosity=0)
    query_matrix=xgb.DMatrix(q,missing=np.nan,nthread=1)
    training_matrix=xgb.DMatrix(x,label=y,missing=np.nan,nthread=1)
    result=[]
    for run in range(250):
        params['seed']=(seed+run)%(2**32)
        model=xgb.train(params,training_matrix,num_boost_round=controls['rounds'])
        scores=model.predict(query_matrix)
        if scores.shape!=(len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(axis=1),1.):
            raise ValueError('Invalid multiclass probabilities')
        scores=scores.astype(np.float64)
        # Remove float32 normalization drift before the strict final blend.
        scores/=scores.sum(axis=1,keepdims=True)
        result.append(scores)
    return np.stack(result)
