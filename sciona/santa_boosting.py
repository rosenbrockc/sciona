"""Two RMSE LightGBM threshold models with held-out early stopping."""
from dataclasses import dataclass
import copy
import numpy as np
import lightgbm as lgb
from sciona.santa_agent_features import BanditHistory, choose_action
from sciona.santa_replay_training import prepare_replay_training


@dataclass(repr=False)
class ThresholdModels:
    normal: object
    transformed: object
    evaluation: dict
    rows: dict

    def predict(self, history, *, seed):
        if type(history) is not BanditHistory:
            raise ValueError('Bandit history required')
        normal, transformed = history.features()
        a = self.normal.predict(normal, num_iteration=self.normal.best_iteration, num_threads=1)
        b = self.transformed.predict(transformed, num_iteration=self.transformed.best_iteration, num_threads=1)
        return choose_action(a, b, history, seed=seed)


def fit_threshold_models(training, validation, *, seed=42, max_rows=100000,
                         num_boost_round=1024, stopping_rounds=50, num_leaves=4095):
    for value, lower in [(num_boost_round,1),(stopping_rounds,1),(num_leaves,2)]:
        if type(value) is not int or value < lower:
            raise ValueError('Positive integer boosting controls required')
    if num_leaves > 131072:
        raise ValueError('Too many leaves')
    populations = prepare_replay_training(training, validation, seed=seed, max_rows=max_rows)
    params = dict(boosting_type='gbdt', objective='regression', metric='rmse',
                  num_leaves=num_leaves, learning_rate=.05, feature_fraction=.9,
                  bagging_fraction=.5, bagging_freq=5, seed=seed, verbosity=-1,
                  num_threads=1, deterministic=True, force_col_wise=True)
    models, evaluation = {}, {}
    for name, target in [('normal','raw_target'),('transformed','transformed_target')]:
        train = populations['training']; valid = populations['validation']
        training_set = lgb.Dataset(train[name], label=train[target])
        validation_set = lgb.Dataset(valid[name], label=valid[target], reference=training_set)
        history = {}
        model = lgb.train(params, training_set, num_boost_round=num_boost_round,
                          valid_sets=[validation_set], valid_names=['heldout'],
                          callbacks=[lgb.early_stopping(stopping_rounds, verbose=False),
                                     lgb.record_evaluation(history)])
        predictions = model.predict(valid[name], num_iteration=model.best_iteration, num_threads=1)
        if not np.isfinite(predictions).all() or (name == 'transformed' and (predictions <= 0).any()):
            raise ValueError('Invalid fitted threshold predictions')
        evaluation[name] = dict(best_iteration=model.best_iteration,
                               heldout_rmse=float(np.sqrt(np.mean((predictions-valid[target])**2))),
                               iteration_rmse=copy.deepcopy(history['heldout']['rmse']))
        models[name] = model
    return ThresholdModels(models['normal'], models['transformed'], evaluation,
                           {name: p['rows'] for name,p in populations.items()})
