"""250 seeded AdaBoost/ExtraTrees meta fits with explicit modern SAMME.

Historical SAMME versus SAMME.R and tree controls remain unresolved. Inputs
must be prepared meta features; this component does not certify their OOF origin.
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier


def fit_bag(reference,labels,query,*,seed,controls):
    x=np.asarray(reference,dtype=float);q=np.asarray(query,dtype=float);y=np.asarray(labels)
    if x.ndim!=2 or q.ndim!=2 or not len(x) or not len(q) or x.shape[1]<1 or x.shape[1]!=q.shape[1] or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Aligned finite meta features required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):raise ValueError('Nine aligned integer classes required')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned seed required')
    keys={'algorithm','boost_rounds','learning_rate','trees','max_depth','min_samples_leaf','max_features'}
    if type(controls) is not dict or set(controls)!=keys or controls['algorithm']!='SAMME':raise ValueError('Explicit supported SAMME and tree controls required')
    for name in ('boost_rounds','trees','max_depth','min_samples_leaf'):
        if type(controls[name]) is not int or controls[name]<1:raise ValueError('Positive integer fitting controls required')
    if type(controls['learning_rate']) not in (int,float) or not np.isfinite(controls['learning_rate']) or controls['learning_rate']<=0:raise ValueError('Positive learning rate required')
    if type(controls['max_features']) is not int or not 1<=controls['max_features']<=x.shape[1]:raise ValueError('Valid candidate feature count required')
    predictions=[]
    for run in range(250):
        run_seed=(seed+run)%2**32
        trees=ExtraTreesClassifier(n_estimators=controls['trees'],max_depth=controls['max_depth'],min_samples_leaf=controls['min_samples_leaf'],max_features=controls['max_features'],n_jobs=1,random_state=run_seed)
        model=AdaBoostClassifier(estimator=trees,n_estimators=controls['boost_rounds'],learning_rate=controls['learning_rate'],random_state=run_seed)
        model.fit(x,y)
        if not np.array_equal(model.classes_,np.arange(9)):raise ValueError('Unexpected meta class order')
        scores=model.predict_proba(q)
        if scores.shape!=(len(q),9) or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(axis=1),1.):raise ValueError('Invalid meta probabilities')
        predictions.append(scores)
    return np.stack(predictions)
