"""Explicit rolling one-step validation, TPE selection and final LightGBM refit."""
import numpy as np
import lightgbm as lgb
import optuna
from sciona.future_sales_features import Panel

_FIELDS={'train_start','validation_start','trials','startup_trials','max_rounds','stopping_rounds','seed','search_space'}


def train_forecast(panel,config):
    if not isinstance(panel,Panel) or not isinstance(config,dict) or set(config)!=_FIELDS:raise ValueError('Prepared panel and exact training controls required')
    for name in _FIELDS-{'search_space'}:
        if type(config[name]) is not int or config[name]<0:raise ValueError('Nonnegative integer training controls required')
    if not 0<=config['seed']<2**32 or not 1<=config['startup_trials']<config['trials'] or not 1<=config['stopping_rounds']<config['max_rounds']:
        raise ValueError('Require Bayesian trials beyond startup and usable early stopping')
    forecast=int(panel.periods.max());start=config['train_start'];validation=config['validation_start']
    if not 0<=start<validation<forecast:raise ValueError('Ordered nonempty fitting, validation and forecast periods required')
    space=config['search_space']
    if not isinstance(space,dict) or set(space)!={'num_leaves','min_data_in_leaf','learning_rate'}:raise ValueError('Exact bounded search space required')
    for name,bounds in space.items():
        if not isinstance(bounds,list) or len(bounds)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in bounds):raise ValueError('Finite search bounds required')
        lo,hi=bounds
        if not 0<lo<hi:raise ValueError('Nondegenerate positive search bounds required')
        if name!='learning_rate' and (any(type(v) is not int for v in bounds) or lo<(2 if name=='num_leaves' else 1)):raise ValueError('Valid integer tree bounds required')
        if name=='learning_rate' and hi>1:raise ValueError('Learning rate exceeds supported range')
    fitting=(panel.periods>=start)&(panel.periods<validation)
    validating=(panel.periods>=validation)&(panel.periods<forecast)
    final=(panel.periods>=start)&(panel.periods<forecast);predicting=panel.periods==forecast
    if min(fitting.sum(),validating.sum())<2:raise ValueError('At least two fitting and validation rows required')
    x,y=panel.features,panel.targets
    if not np.isfinite(y[final]).all() or not np.isnan(y[predicting]).all() or np.isinf(x).any():raise ValueError('Invalid observed/forecast panel targets')
    def parameters(extra):
        return dict(objective='regression',metric='None',verbosity=-1,deterministic=True,force_col_wise=True,num_threads=1,
            seed=config['seed'],feature_fraction_seed=config['seed'],bagging_seed=config['seed'],data_random_seed=config['seed'],
            feature_pre_filter=False,**extra)
    def dataset(mask):return lgb.Dataset(x[mask],label=y[mask],categorical_feature=[0,1,2],free_raw_data=False)
    def metric(predictions,data):
        value=float(np.sqrt(np.mean((np.clip(predictions,0,20)-data.get_label())**2)))
        if not np.isfinite(value):raise ValueError('Nonfinite validation score')
        return 'clipped_rmse',value,False
    history=[]
    def objective(trial):
        chosen={n:trial.suggest_int(n,*space[n]) for n in ('num_leaves','min_data_in_leaf')}
        chosen['learning_rate']=trial.suggest_float('learning_rate',*space['learning_rate'],log=True)
        model=lgb.train(parameters(chosen),dataset(fitting),num_boost_round=config['max_rounds'],valid_sets=[dataset(validating)],feval=metric,
            callbacks=[lgb.early_stopping(config['stopping_rounds'],first_metric_only=True,verbose=False)])
        best=int(model.best_iteration)
        if best<1:raise ValueError('No valid selected boosting iteration')
        prediction=model.predict(x[validating],num_iteration=best)
        score=metric(prediction,lgb.Dataset(x[validating],label=y[validating]))[1]
        trial.set_user_attr('best_iteration',best)
        history.append(dict(trial=trial.number,best_iteration=best,validation_rmse=score))
        return score
    # Silence potentially private objective metrics while preserving caller logging.
    verbosity=optuna.logging.get_verbosity()
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler=optuna.samplers.TPESampler(seed=config['seed'],n_startup_trials=config['startup_trials'],multivariate=False,constant_liar=False)
        study=optuna.create_study(direction='minimize',sampler=sampler)
        study.optimize(objective,n_trials=config['trials'],n_jobs=1,show_progress_bar=False)
    finally:optuna.logging.set_verbosity(verbosity)
    winner=study.best_trial;rounds=winner.user_attrs['best_iteration']
    model=lgb.train(parameters(winner.params),dataset(final),num_boost_round=rounds)
    predictions=np.clip(model.predict(x[predicting],num_iteration=rounds),0.,20.)
    if not np.isfinite(predictions).all():raise ValueError('Invalid forecast')
    return dict(predictions=predictions,trials=history,selected_trial=winner.number,selected_rounds=rounds,
        training_rows=int(fitting.sum()),validation_rows=int(validating.sum()),refit_rows=int(final.sum()),forecast_rows=int(predicting.sum()))
