"""Independent CPU reconstruction of the full VSB repeated-fold ensemble."""
import numpy as np
import lightgbm as lgb
from sciona.vsb_threshold import binary_labels,repeated_folds,threshold


def fit(training,signal_labels,query,*,repetitions=25,split_seed=123948,tree_seed=23974,rounds=10000,patience=100):
    x=np.asarray(training);q=np.asarray(query);labels=np.asarray(signal_labels)
    for a in (x,q):
        if a.ndim!=2 or not len(a) or a.shape[1]!=9 or a.dtype.kind not in 'iuf' or np.isinf(a).any():raise ValueError('Nine real features with optional NaN required')
    if labels.shape!=(len(x),3):raise ValueError('Three aligned signal labels per measurement required')
    binary_labels(labels.reshape(-1));y=labels.any(axis=1).astype(int)
    for value in (rounds,patience):
        if type(value) is not int or value<1:raise ValueError('Positive training limits required')
    if type(tree_seed) is not int or not 0<=tree_seed<2**31:raise ValueError('Bounded tree seed required')
    plans=repeated_folds(y,seed=split_seed,repetitions=repetitions)
    # Full notebook uses this explicit permutation of the nine aggregate features.
    order=[2,7,3,4,8,0,6,5,1];x=x[:,order].astype(float);q=q[:,order].astype(float)
    params=dict(objective='binary',boosting='gbdt',learning_rate=.01,num_leaves=80,num_threads=1,
        metric='binary_logloss',feature_fraction=.8,bagging_freq=1,bagging_fraction=.8,seed=tree_seed,
        verbosity=-1,deterministic=True,force_col_wise=True)
    train_scores=np.zeros(len(x));validation_scores=np.zeros(len(x));test_scores=np.zeros(len(x));query_scores=np.zeros(len(q));iterations=[]
    for train,val,test in plans:
        reference=lgb.Dataset(x[train],label=y[train])
        nominal_test=lgb.Dataset(x[test],label=y[test],reference=reference)
        validation=lgb.Dataset(x[val],label=y[val],reference=reference)
        model=lgb.train(params,reference,num_boost_round=rounds,
            valid_sets=[reference,nominal_test,validation],valid_names=['train','test','validation'],
            callbacks=[lgb.early_stopping(patience,first_metric_only=False,verbose=False)])
        train_scores[train]+=model.predict(x[train]);validation_scores[val]+=model.predict(x[val]);test_scores[test]+=model.predict(x[test])
        query_scores+=model.predict(q)/len(plans);iterations.append(model.best_iteration)
    train_scores/=3*repetitions;validation_scores/=repetitions;test_scores/=repetitions
    selected=threshold(labels.reshape(-1),np.repeat(train_scores,3))
    if not np.isfinite(query_scores).all():raise ValueError('Nonfinite model output')
    return dict(probabilities=query_scores,signal_decisions=np.repeat((query_scores>selected['threshold']).astype(int)[:,None],3,axis=1),
        threshold=selected['threshold'],threshold_fit_mcc=selected['mcc'],models=len(plans),best_iterations=iterations,
        training_probabilities=train_scores,validation_probabilities=validation_scores,test_probabilities=test_scores,
        scope='Independent CPU reconstruction; both holdouts influence early stopping and final threshold uses in-training signal predictions.')
