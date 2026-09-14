"""Independent augmented LightGBM branch for the pending Santander realization.

Global supervised encoding is intentional; validation scores are not unbiased
CV estimates. Tree controls and categorical treatment must be explicit because
the primary winner explanation does not provide a complete tree configuration.
"""
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sciona.santander_encoding import encode_populations
from sciona.santander_augmentation import shuffle_feature_groups


def augment_reference(categories,raw,substituted,labels,*,seed):
    """Keep originals and append 16 positive / four negative shuffled copies.

Shuffle a raw/category/substituted triplet together within class for each
feature. Keeping originals is an explicit independent interpretation.
"""
    rng=np.random.default_rng(seed)
    initial=shuffle_feature_groups(categories,raw,substituted,labels,rng=rng,training=False)
    if set(initial[3].tolist())!={0,1}:raise ValueError('Both classes required for tree augmentation')
    parts=[initial]
    for label,copies in ((1,16),(0,4)):
        rows=initial[3]==label
        for _ in range(copies):
            parts.append(shuffle_feature_groups(*(a[rows] for a in initial[:3]),initial[3][rows],rng=rng,training=True))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def train_tree_cv(training,labels,test,*,controls,folds=10,seeds=(42,),fold_seed=42):
    fields={'num_leaves','learning_rate','feature_fraction','bagging_fraction','bagging_freq',
            'min_data_in_leaf','max_rounds','stopping_rounds','categorical'}
    if type(controls) is not dict or set(controls)!=fields:raise ValueError('Explicit complete tree controls required')
    for key,minimum in [('num_leaves',2),('bagging_freq',0),('min_data_in_leaf',1),('max_rounds',1),('stopping_rounds',1)]:
        if type(controls[key]) is not int or controls[key]<minimum:raise ValueError('Invalid tree integer control')
    for key in ('learning_rate','feature_fraction','bagging_fraction'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<controls[key]<=1:raise ValueError('Invalid tree rate/fraction')
    if type(controls['categorical']) is not bool:raise ValueError('Explicit categorical treatment required')
    if type(folds) is not int or folds<2 or type(fold_seed) is not int or not 0<=fold_seed<2**31:
        raise ValueError('Valid fold controls required')
    if type(seeds) not in (list,tuple) or not seeds or any(type(s) is not int or not 0<=s<2**31 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Distinct bounded integer seeds required')
    encoded=encode_populations(training,labels,test);x=np.asarray(training);q=np.asarray(test);y=np.asarray(labels)
    if np.bincount(y.astype(int),minlength=2).min()<folds:raise ValueError('Both classes must populate every fold')
    arrays=(encoded['training_categories'],x,encoded['training_substituted'])
    query=np.column_stack((encoded['test_categories'],q,encoded['test_substituted']))
    predictions=[];records=[]
    for fold,(fit,valid) in enumerate(StratifiedKFold(folds,shuffle=True,random_state=fold_seed).split(x,y)):
        validation=np.column_stack([a[valid] for a in arrays])
        for seed in seeds:
            augmented=augment_reference(*(a[fit] for a in arrays),y[fit],seed=seed)
            params=dict(objective='binary',metric='auc',num_threads=1,verbosity=-1,
                deterministic=True,force_col_wise=True,seed=seed,feature_fraction_seed=seed,bagging_seed=seed,
                num_leaves=controls['num_leaves'],learning_rate=controls['learning_rate'],
                feature_fraction=controls['feature_fraction'],bagging_fraction=controls['bagging_fraction'],
                bagging_freq=controls['bagging_freq'],min_data_in_leaf=controls['min_data_in_leaf'])
            categorical=list(range(x.shape[1])) if controls['categorical'] else []
            reference=lgb.Dataset(np.column_stack(augmented[:3]),label=augmented[3],categorical_feature=categorical)
            held=lgb.Dataset(validation,label=y[valid],reference=reference)
            history={}
            model=lgb.train(params,reference,num_boost_round=controls['max_rounds'],valid_sets=[held],valid_names=['held'],
                callbacks=[lgb.early_stopping(controls['stopping_rounds'],verbose=False),lgb.record_evaluation(history)])
            best=model.best_iteration
            restored=float(roc_auc_score(y[valid],model.predict(validation,num_iteration=best)))
            aucs=history['held']['auc']
            if best<1 or not np.isclose(restored,max(aucs),rtol=0,atol=1e-12):raise ValueError('Selected tree checkpoint differs')
            scores=model.predict(query,num_iteration=best)
            if scores.shape!=(len(q),) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Invalid tree predictions')
            predictions.append(scores)
            records.append(dict(fold=fold,seed=seed,fit_rows=len(fit),augmented_rows=len(augmented[3]),
                validation_rows=len(valid),best_iteration=best,validation_auc=aucs,restored_auc=restored))
    return dict(mean_ranks=np.mean([rankdata(p,method='average') for p in predictions],axis=0),
        model_probabilities=np.stack(predictions),models=records,
        validation_scope='Globally encoded; not fold-isolated validation')
