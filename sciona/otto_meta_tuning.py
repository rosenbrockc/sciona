"""Four-fold selection on supplied meta features, retaining full source bag sizes.

Scores select controls; they are not an independent final test estimate.
First-level OOF provenance and embedding population scope remain caller duties.
"""
import numpy as np


def _learner(family):
    if family=='xgboost':
        from sciona.otto_meta_xgboost import fit_bag
        return fit_bag,250
    if family=='adaboost_extratrees':
        from sciona.otto_meta_adaboost import fit_bag
        return fit_bag,250
    if family=='lasagne_neural':
        from sciona.otto_meta_neural import fit_bag
        return fit_bag,600
    raise ValueError('Explicit supported meta family required')


def tune(training,labels,folds,identities,*,family,seed,candidates):
    x=np.asarray(training,dtype=float);y=np.asarray(labels);f=np.asarray(folds)
    if x.ndim!=2 or not len(x) or x.shape[1]<1 or not np.isfinite(x).all():raise ValueError('Finite nonempty meta matrix required')
    if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):raise ValueError('Nine aligned integer classes required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(4)):raise ValueError('Exactly four aligned folds required')
    if type(identities) is not list or len(identities)!=len(x) or any(type(i) is not str or not i for i in identities) or len(set(identities))!=len(identities):raise ValueError('Unique aligned opaque identities required')
    if type(candidates) is not list or not candidates or any(type(c) is not dict for c in candidates):raise ValueError('Explicit candidate controls required')
    for fold in range(4):
        if set(y[f!=fold].tolist())!=set(range(9)):raise ValueError('Each fitting population requires all classes')
    learner,runs=_learner(family);predictions=[];metrics=[]
    for controls in candidates:
        oof=np.empty((len(x),9));coverage=np.zeros(len(x),dtype=int)
        for fold in range(4):
            held=f==fold
            bag=learner(x[~held],y[~held],x[held],seed=seed,controls=controls)
            if bag.shape!=(runs,int(held.sum()),9) or not np.isfinite(bag).all() or (bag<0).any() or (bag>1).any() or not np.allclose(bag.sum(axis=2),1.,atol=1e-8):raise ValueError('Invalid full-size meta bag')
            oof[held]=bag.mean(axis=0);coverage[held]+=1
        if not np.all(coverage==1):raise ValueError('Incomplete OOF coverage')
        # Explicit double-precision floor for zero-probability class outcomes.
        loss=float(-np.log(np.maximum(oof[np.arange(len(x)),y],np.finfo(float).eps)).mean())
        predictions.append(oof);metrics.append(loss)
    selected=int(np.argmin(metrics)) # First supplied candidate wins exact ties.
    return dict(selected_index=selected,log_losses=metrics,candidate_oof=np.stack(predictions),folds=4,models_per_candidate=4*runs)
