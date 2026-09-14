"""Independent grouped quantile search using pinned numerical parameter ranges."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut,RandomizedSearchCV
from sklearn.metrics import make_scorer
from sciona.m5u_loss import pinball


def search(frame,targets,groups,quantile,*,fast=False,iterations=4,seed=0):
    targets,groups=np.asarray(targets),np.asarray(groups)
    if (not isinstance(frame,pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or targets.shape!=(len(frame),) or groups.shape!=targets.shape or groups.dtype.kind not in 'iu'
            or len(np.unique(groups))<2):raise ValueError('Aligned features, targets and at least two groups required')
    pinball(targets,targets,quantile)
    if type(fast) is not bool or type(iterations) is not int or iterations<1 or type(seed) is not int or not 0<=seed<2**31:
        raise ValueError('Invalid search controls')
    for name in frame:
        column=frame[name]
        if isinstance(column.dtype,pd.CategoricalDtype):continue
        if column.dtype.kind not in 'iuf' or np.isinf(column.to_numpy()).any():raise ValueError('Unsupported feature values')
    distributions=json.loads(Path(__file__).with_name('m5u_search_parameters.json').read_text())['fast' if fast else 'full']
    folds=list(LeaveOneGroupOut().split(frame,targets,groups))
    for training,validation in folds:
        assert not set(groups[training]).intersection(groups[validation])
    learner=lgb.LGBMRegressor(objective='quantile',alpha=float(quantile),verbosity=-1,
        hist_pool_size=1000,importance_type='gain',seed=seed,n_jobs=1,deterministic=True,force_col_wise=True)
    searcher=RandomizedSearchCV(learner,distributions,n_iter=iterations,cv=folds,n_jobs=1,
        scoring=make_scorer(pinball,greater_is_better=False,quantile=quantile),random_state=seed,error_score='raise',refit=True)
    searcher.fit(frame,targets)
    result=searcher.cv_results_
    if not np.isfinite(result['mean_test_score']).all():raise ValueError('Search scores must be finite')
    return searcher.best_estimator_,dict(candidates=len(result['params']),folds=len(folds),
        candidate_fits=len(result['params'])*len(folds),refits=1,
        candidate_losses=(-result['mean_test_score']).tolist(),best_index=int(searcher.best_index_),
        best_parameters=searcher.best_params_,best_loss=float(-searcher.best_score_),
        group_isolation=True,scope='Grouped internal model selection with explicit independent CPU/RNG choices; no untouched-test claim.')
