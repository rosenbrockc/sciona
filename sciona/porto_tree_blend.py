"""Raw-prepared LightGBM branch and equal-probability Porto ensemble mean."""
import numpy as np
import lightgbm as lgb
from sciona.porto_transforms import _matrix


def fit_tree(reference,labels,query,*,seed,controls):
    fields={'rounds','num_leaves','learning_rate','min_data_in_leaf','feature_fraction','bagging_fraction','bagging_freq','lambda_l2'}
    if type(controls) is not dict or set(controls)!=fields:raise ValueError('Explicit complete tree controls required')
    for key,minimum in [('rounds',1),('num_leaves',2),('min_data_in_leaf',1),('bagging_freq',0)]:
        if type(controls[key]) is not int or controls[key]<minimum:raise ValueError('Invalid integer tree control')
    for key in ('learning_rate','feature_fraction','bagging_fraction'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<controls[key]<=1:raise ValueError('Invalid tree rate/fraction')
    if type(controls['lambda_l2']) not in (int,float) or not np.isfinite(controls['lambda_l2']) or controls['lambda_l2']<0:raise ValueError('Invalid tree regularization')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Valid integer seed required')
    x=_matrix(reference);q=_matrix(query);y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1}:raise ValueError('Aligned binary population required')
    params={k:v for k,v in controls.items() if k!='rounds'}
    params.update(objective='binary',metric='binary_logloss',verbosity=-1,num_threads=1,
        deterministic=True,force_col_wise=True,seed=seed,feature_fraction_seed=seed,bagging_seed=seed)
    model=lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=controls['rounds'])
    scores=model.predict(q)
    if scores.shape!=(len(q),) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Invalid tree probabilities')
    return dict(probabilities=scores,boosting_iterations=model.current_iteration())


def blend(neural_probabilities,tree_probabilities):
    neural=np.asarray(neural_probabilities);tree=np.asarray(tree_probabilities)
    if neural.ndim!=2 or neural.shape[0]!=5 or not neural.shape[1] or tree.shape!=(neural.shape[1],):raise ValueError('Exactly five aligned neural models and one tree required')
    for values in (neural,tree):
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Finite probabilities in [0,1] required')
    return (neural.sum(axis=0,dtype=np.float64)+tree)/6
