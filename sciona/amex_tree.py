"""Independent two-variant downstream Amex DART ensemble."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sciona.amex_numeric import _matrix


def fit(manual,slots,labels,query_manual,query_slots,*,seed=42,rounds=4500):
    x=_matrix(manual);s=_matrix(slots);q=_matrix(query_manual);t=_matrix(query_slots)
    if s.shape!=(len(x),13) or t.shape!=(len(q),13) or x.shape[1]!=q.shape[1]:raise ValueError('Aligned manual features and thirteen prediction slots required')
    for p in (s,t):
        valid=p[~np.isnan(p)]
        if np.any((valid<0)|(valid>1)):raise ValueError('Finite probability slots or missing values required')
    if type(labels) is not list or len(labels)!=len(x) or any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('Aligned integer binary labels required')
    y=np.asarray(labels)
    if min(np.bincount(y,minlength=2))<5:raise ValueError('Five customers per class required')
    if type(seed) is not int or not 0<=seed<2**31 or type(rounds) is not int or rounds<1:raise ValueError('Bounded seed and positive rounds required')
    plans=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(x,y))
    params=dict(objective='binary',metric='binary_logloss',boosting='dart',max_depth=-1,num_leaves=64,learning_rate=.035,
        bagging_freq=5,bagging_fraction=.75,feature_fraction=.05,min_data_in_leaf=256,max_bin=63,min_data_in_bin=256,
        tree_learner='serial',boost_from_average=False,lambda_l1=.1,lambda_l2=30,num_threads=1,verbosity=-1,seed=seed,
        deterministic=True,force_col_wise=True)
    result={}
    for name,reference,query in [('manual',x,q),('manual_with_slots',np.concatenate((x,s),axis=1),np.concatenate((q,t),axis=1))]:
        oof=np.full(len(x),np.nan);pred=np.zeros(len(q));coverage=np.zeros(len(x),int);iterations=[]
        for tr,va in plans:
            dataset=lgb.Dataset(reference[tr],label=y[tr]);valid=lgb.Dataset(reference[va],label=y[va],reference=dataset)
            model=lgb.train(params,dataset,num_boost_round=rounds,valid_sets=[dataset,valid])
            oof[va]=model.predict(reference[va],num_iteration=-1);coverage[va]+=1
            restored=lgb.Booster(model_str=model.model_to_string(num_iteration=-1))
            pred+=restored.predict(query,num_iteration=-1)/5;iterations.append(model.current_iteration())
        if not np.all(coverage==1) or not np.isfinite(oof).all() or not np.isfinite(pred).all():raise ValueError('Incomplete finite variant predictions')
        result[name]=dict(training_predictions=oof,query_predictions=pred,models=5,iterations=iterations)
    return dict(variants=result,models=10,configured_rounds=rounds,
        scope='Independent current CPU DART; customer-stratified folds and all-round prediction. No active early stopping or historical accuracy qualification.')
