"""Independent implementation of the five source-listed XGBoost branches.

The author notebook performs five-fold OOF training followed by a full-data
refit per classifier, not query-time averaging of the five fold models.
Input representations must be prepared by the encompassing pipeline.
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold


# branch: representation, rounds, eta, depth, child weight, column fraction, seed
BRANCHES = {
    'xgb1': ('proxy', 5, 0.39, 100, 0.1, 0.4, 4),
    'xgb2': ('restricted', 35, 0.2, 15, 15, 0.9, 5),
    'xgb3': ('base', 700, 0.05, 6, 10, 1.0, 1),
    'xgb4': ('proxy', 3, 0.5, 155, 0.1, 0.4, 4),
    'xgb5': ('corrected', 1500, 0.05, 6, 10, 1.0, 1),
}


def train_classifiers(training_views, labels, query_views):
    """Return five OOF vectors, five full-refit query vectors and fit evidence.

    Modern exact-tree CPU XGBoost and base score 0.5 are explicit reference
    choices. Every classifier gets all its published rounds. Feature dependence
    introduced upstream by mass cross-fitting remains a downstream OOF caveat.
    """
    keys = {'base', 'restricted', 'proxy', 'corrected'}
    if set(training_views) != keys or set(query_views) != keys:
        raise ValueError('Exactly four required feature representations must be supplied')
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim != 1 or len(y) < 5 or not np.isfinite(y).all() or set(y) != {0, 1}:
        raise ValueError('Labels require at least five rows and both binary classes')
    train = {k: np.asarray(v, dtype=np.float64) for k, v in training_views.items()}
    query = {k: np.asarray(v, dtype=np.float64) for k, v in query_views.items()}
    query_rows = None
    for key in keys:
        a, b = train[key], query[key]
        if a.ndim != 2 or a.shape[0] != len(y) or a.shape[1] == 0:
            raise ValueError('Training views must be nonempty and aligned with labels')
        if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] != a.shape[1]:
            raise ValueError('Query views must match their training widths')
        if query_rows is not None and b.shape[0] != query_rows:
            raise ValueError('Query view rows must align')
        query_rows = b.shape[0]
        if np.isinf(a).any() or np.isinf(b).any():
            raise ValueError('Infinite feature values are unsupported')
    folds = tuple(KFold(5, shuffle=True, random_state=555).split(y))
    if any(len(np.unique(y[fit])) != 2 for fit, _ in folds):
        raise ValueError('Both classes are required in every training fold')
    out = {'oof': {}, 'query': {}, 'fits': [], 'checkpoint_replays_exact': True,
           'runtime_version': xgb.__version__}
    for branch, (view, rounds, eta, depth, child, columns, seed) in BRANCHES.items():
        params = dict(objective='binary:logistic', eta=eta, max_depth=depth,
                      min_child_weight=child, subsample=1.0, colsample_bytree=columns,
                      seed=seed, nthread=1, tree_method='exact', base_score=0.5)
        oof = np.empty(len(y), dtype=np.float64)
        for fold, (fit, held) in enumerate(folds):
            model = xgb.train(params, xgb.DMatrix(train[view][fit], label=y[fit], nthread=1),
                              num_boost_round=rounds)
            oof[held] = model.predict(xgb.DMatrix(train[view][held], nthread=1))
            out['fits'].append(dict(branch=branch, fold=fold,
                                    rounds=model.num_boosted_rounds()))
        model = xgb.train(params, xgb.DMatrix(train[view], label=y, nthread=1), num_boost_round=rounds)
        matrix = xgb.DMatrix(query[view], nthread=1)
        prediction = model.predict(matrix)
        # Deserialization also uses XGBoost's scoped runtime thread setting.
        # A Booster parameter alone does not cover its native load path.
        with xgb.config_context(nthread=1):
            restored = xgb.Booster(params={'nthread': 1})
            restored.load_model(model.save_raw())
            if not np.array_equal(prediction, restored.predict(matrix)):
                raise RuntimeError('Full-refit checkpoint prediction replay differed')
        for values in (oof, prediction):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError('Classifier produced invalid probabilities')
        out['oof'][branch], out['query'][branch] = oof, prediction
        out['fits'].append(dict(branch=branch, fold='full', rounds=model.num_boosted_rounds()))
    return out
