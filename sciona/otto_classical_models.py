"""Three documented log1p classifier families with explicit fitting controls."""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.exceptions import ConvergenceWarning
from sciona.otto_preprocessing import representation


def _model(family, controls, seed):
    if type(controls) is not dict:raise ValueError('Explicit model controls required')
    if family=='logistic' and set(controls)=={'C','max_iter'}:
        return LogisticRegression(C=controls['C'],max_iter=controls['max_iter'],solver='lbfgs',random_state=seed)
    if family=='extra_trees' and set(controls)=={'n_estimators','max_features','min_samples_leaf'}:
        return ExtraTreesClassifier(**controls,random_state=seed,n_jobs=1)
    if family=='multinomial_nb' and set(controls)=={'alpha'}:
        return MultinomialNB(**controls)
    raise ValueError('Unsupported family or incomplete model controls')


def crossfit(training,labels,folds,training_ids,query,query_ids,*,family,controls,seed):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit unsigned seed required')
    x=representation(training,kind='log1p');q=representation(query,kind='log1p')
    y=np.asarray(labels);f=np.asarray(folds)
    if q.shape[1]!=x.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):
        raise ValueError('Aligned nine-class populations required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Exactly five aligned folds required')
    seen=set()
    for ids,count in [(training_ids,len(x)),(query_ids,len(q))]:
        if type(ids) is not list or len(ids)!=count or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque identities required')
        for identity in ids:
            if identity in seen:raise ValueError('Duplicate or overlapping identities')
            seen.add(identity)
    for fold in range(5):
        if set(y[f!=fold].tolist())!=set(range(9)):raise ValueError('Fitting fold missing class')
    _model(family,controls,seed)
    def fit_predict(reference,target,evaluation):
        model=_model(family,controls,seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(reference,target)
        if not np.array_equal(model.classes_,np.arange(9)):raise ValueError('Class order differs')
        result=model.predict_proba(evaluation)
        if not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.sum(axis=1),1.):
            raise ValueError('Invalid classifier probabilities')
        return result
    oof=np.empty((len(x),9));coverage=np.zeros(len(x),dtype=int)
    for fold in range(5):
        held=f==fold
        oof[held]=fit_predict(x[~held],y[~held],x[held]);coverage[held]+=1
    if not np.all(coverage==1):raise ValueError('Incomplete OOF coverage')
    return dict(oof=oof,query=fit_predict(x,y,q),fitting_runs=6,family=family)
