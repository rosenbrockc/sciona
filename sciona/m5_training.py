"""Independent M5 single-family learner with source-configured boosting controls.

Pool selection and feature selection are performed by the caller. Diagnostics
do not control training duration. Recursive diagnostics overlap training data;
missing nonrecursive diagnostic targets use LightGBM's zero-label convention.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb


def parameters(recursive, pooling):
    if not isinstance(recursive,bool) or pooling not in ('outlet','outlet_category','outlet_department'):
        raise ValueError('Unknown model family')
    broad=recursive and pooling=='outlet'
    return dict(boosting_type='gbdt',objective='tweedie',tweedie_variance_power=1.1,
        metric='rmse',subsample=.5,subsample_freq=1,learning_rate=.015,
        num_leaves=2047 if broad else 255,min_data_in_leaf=4095 if broad else 255,
        feature_fraction=.5,max_bin=100,boost_from_average=False,verbosity=-1,
        seed=42 if recursive else 1995,num_threads=1,deterministic=True,force_col_wise=True)


def fit(frame, targets, days, cutoff, *, recursive, pooling, first_day):
    """Train one selected pool at 3000 configured rounds without early stopping."""
    params=parameters(recursive,pooling)
    targets,days=np.asarray(targets),np.asarray(days)
    if (not isinstance(frame,pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or targets.shape!=(len(frame),) or targets.dtype.kind not in 'iuf'
            or days.shape!=targets.shape or days.dtype.kind not in 'iu'):
        raise ValueError('Expected aligned feature frame, numeric targets and integer days')
    for value in (cutoff,first_day):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError('Day controls must be integers')
    if first_day>cutoff or np.isinf(targets).any() or (targets<0).any():
        raise ValueError('Invalid training range or targets')
    for column in frame:
        series=frame[column]
        if isinstance(series.dtype,pd.CategoricalDtype):
            continue
        if series.dtype.kind not in 'iuf' or np.isinf(series.to_numpy()).any():
            raise ValueError('Features must be numeric or categorical without infinity')
    training=(days>=first_day)&(days<=cutoff)
    diagnostic=(training&(days>cutoff-28)) if recursive else ((days>cutoff)&(days<=cutoff+28))
    if not training.any() or not diagnostic.any() or np.isnan(targets[training]).any():
        raise ValueError('Complete training targets and nonempty diagnostics required')
    if not (targets[training]>0).any():
        raise ValueError('Tweedie training requires a positive target')
    train=lgb.Dataset(frame.loc[training],label=targets[training],free_raw_data=False)
    validation=lgb.Dataset(frame.loc[diagnostic],label=np.nan_to_num(targets[diagnostic],nan=0.),reference=train)
    evaluation={}
    model=lgb.train(params,train,num_boost_round=3000,valid_sets=[validation],valid_names=['diagnostic'],
                    callbacks=[lgb.record_evaluation(evaluation)])
    assert len(evaluation['diagnostic']['rmse'])==3000
    return model,dict(configured_rounds=3000,evaluated_rounds=3000,retained_iterations=model.current_iteration(),
        training_rows=int(training.sum()),diagnostic_rows=int(diagnostic.sum()),
        overlapping_rows=int((training&diagnostic).sum()),
        missing_diagnostic_targets=int(np.isnan(targets[diagnostic]).sum()),
        recursive=recursive,pooling=pooling,
        scope='Single deterministic CPU pool; diagnostic RMSE is not an unbiased accuracy estimate.')
