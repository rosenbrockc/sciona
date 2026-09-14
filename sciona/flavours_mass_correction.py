"""Independent repeated cross-fit mass regression for the corrected Flavours CDG.

Method reference: Go Polar Bears author notebook, cell 20, pinned in source
triage. Modern CPU XGBoost squared-error training is an explicit reconstruction;
historical XGBoost 0.4 numerical equivalence is not claimed.
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold


FOLD_SEEDS = (1, 2, 3, 4, 5, 6, 555)
BOOSTING_ROUNDS = 2800


def correct_mass(features, mass, query_features):
    """Fit all 35 regressors and return OOF and ensemble mass estimates.

    Inputs are caller-prepared numerical matrices. Infinite values are rejected;
    NaN features use XGBoost's missing-value handling. No targets from query
    populations are accepted. Five or more training rows are required.
    The OOF matrix is *not* an independently nested downstream validation set:
    other rows' corrected features may depend on a downstream held-out target.
    """
    train = np.asarray(features, dtype=np.float64)
    target = np.asarray(mass, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if train.ndim != 2 or train.shape[0] < 5 or train.shape[1] == 0:
        raise ValueError('Training features require at least five rows and one column')
    if query.ndim != 2 or query.shape[1] != train.shape[1] or len(query) == 0:
        raise ValueError('Nonempty query features must have the training width')
    if target.shape != (len(train),) or not np.isfinite(target).all():
        raise ValueError('Finite mass targets must align exactly with training rows')
    if np.isinf(train).any() or np.isinf(query).any():
        raise ValueError('Infinite features are not supported')
    params = dict(objective='reg:squarederror', eta=0.05, max_depth=6,
                  min_child_weight=10, subsample=0.8, colsample_bytree=0.9,
                  seed=1, nthread=1, tree_method='exact', base_score=0.5)
    oof = np.zeros(len(train), dtype=np.float64)
    coverage = np.zeros(len(train), dtype=np.int64)
    prediction = np.zeros(len(query), dtype=np.float64)
    query_matrix = xgb.DMatrix(query, missing=np.nan, nthread=1)
    fit_rounds = []
    for seed in FOLD_SEEDS:
        for fit, held in KFold(5, shuffle=True, random_state=seed).split(train):
            model = xgb.train(params, xgb.DMatrix(train[fit], label=target[fit], missing=np.nan, nthread=1),
                              num_boost_round=BOOSTING_ROUNDS)
            held_prediction = model.predict(xgb.DMatrix(train[held], missing=np.nan, nthread=1))
            new_prediction = model.predict(query_matrix)
            if not (np.isfinite(held_prediction).all() and np.isfinite(new_prediction).all()):
                raise ValueError('Mass regressor produced nonfinite predictions')
            oof[held] += held_prediction
            coverage[held] += 1
            prediction += new_prediction
            fit_rounds.append(model.num_boosted_rounds())
    if not np.all(coverage == len(FOLD_SEEDS)):
        raise RuntimeError('Repeated folds did not cover every row exactly seven times')
    return dict(oof_mass=oof / coverage, query_mass=prediction / len(fit_rounds),
                fit_rounds=tuple(fit_rounds), oof_coverage=coverage,
                runtime_version=xgb.__version__)
