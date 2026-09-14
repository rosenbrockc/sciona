"""Independent grouped row-level DART cross-fitting for Amex prediction slots."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sciona.amex_numeric import _matrix


def fit(training,labels,query,*,seed=42,rounds=4500):
    if type(training) is not list or not training or type(query) is not list or not query:raise ValueError('Training and query sequence populations required')
    xs=[_matrix(x) for x in training];qs=[_matrix(x) for x in query];width=xs[0].shape[1]
    if any(x.shape[1]!=width or len(x)>13 for x in xs+qs):raise ValueError('Aligned widths and at most thirteen rows per customer required')
    if type(labels) is not list or len(labels)!=len(xs) or any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('One integer binary label per customer required')
    y=np.asarray(labels);counts=np.bincount(y,minlength=2)
    if min(counts)<5:raise ValueError('Five customers in each class required')
    if type(seed) is not int or not 0<=seed<2**31 or type(rounds) is not int or rounds<1:raise ValueError('Bounded seed and positive rounds required')
    lengths=[len(x) for x in xs];query_lengths=[len(x) for x in qs]
    groups=np.repeat(np.arange(len(xs)),lengths);x=np.vstack(xs);q=np.vstack(qs);row_labels=np.repeat(y,lengths)
    oof=np.full(len(x),np.nan);prediction=np.zeros(len(q));coverage=np.zeros(len(x),int);iterations=[]
    params=dict(objective='binary',metric='binary_logloss',boosting='dart',max_depth=-1,num_leaves=64,learning_rate=.035,
        bagging_freq=5,bagging_fraction=.7,feature_fraction=.7,min_data_in_leaf=256,max_bin=63,min_data_in_bin=256,
        tree_learner='serial',boost_from_average=False,lambda_l1=.1,lambda_l2=30,num_threads=1,verbosity=-1,seed=seed,
        deterministic=True,force_col_wise=True)
    for train_customers,valid_customers in StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y):
        tr=np.flatnonzero(np.isin(groups,train_customers));va=np.flatnonzero(np.isin(groups,valid_customers))
        reference=lgb.Dataset(x[tr],label=row_labels[tr]);validation=lgb.Dataset(x[va],label=row_labels[va],reference=reference)
        model=lgb.train(params,reference,num_boost_round=rounds,valid_sets=[reference,validation])
        oof[va]=model.predict(x[va],num_iteration=-1);coverage[va]+=1
        # Source query inference reloads saved boosters. Exercise that boundary in memory.
        restored=lgb.Booster(model_str=model.model_to_string(num_iteration=-1))
        prediction+=restored.predict(q,num_iteration=-1)/5
        iterations.append(model.current_iteration())
    if not np.all(coverage==1) or not np.isfinite(oof).all() or not np.isfinite(prediction).all():raise ValueError('Incomplete finite cross-fit predictions')
    return dict(training_predictions=np.split(oof,np.cumsum(lengths)[:-1]),query_predictions=np.split(prediction,np.cumsum(query_lengths)[:-1]),
        models=5,iterations=iterations,configured_rounds=rounds,
        scope='Customer-stratified row DART; no active early stopping. Independent current single-thread CPU runtime; historical parity unqualified.')
