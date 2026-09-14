"""Four-fold meta selection, full-reference bags and documented final blend."""
import numpy as np
from sciona.otto_meta_tuning import tune,_learner
from sciona.otto_final_blend import blend

FAMILIES=('xgboost','lasagne_neural','adaboost_extratrees')


def fit(layout,labels,folds,identities,*,seed,candidates,progress=None):
    if type(candidates) is not dict or set(candidates)!=set(FAMILIES):raise ValueError('Candidates for all three meta families required')
    pairs={}
    for family in FAMILIES:
        prefix='neural' if family=='lasagne_neural' else 'tree'
        x=np.asarray(layout[prefix+'_training'],dtype=float);q=np.asarray(layout[prefix+'_query'],dtype=float)
        if x.ndim!=2 or q.ndim!=2 or not len(q) or x.shape[1]!=q.shape[1] or not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Aligned finite meta train/query matrices required')
        pairs[family]=(x,q)
    if len({(len(x),len(q)) for x,q in pairs.values()})!=1:raise ValueError('Meta families must share row counts')
    bags={};selection={}
    for family in FAMILIES:
        x,q=pairs[family]
        result=tune(x,labels,folds,identities,family=family,seed=seed,candidates=candidates[family])
        selected=result['selected_index']
        learner,runs=_learner(family)
        bags[family]=learner(x,labels,q,seed=seed,controls=candidates[family][selected])
        selection[family]=dict(selected_index=selected,log_losses=result['log_losses'],tuning_model_fits=result['models_per_candidate']*len(candidates[family]),refit_models=runs)
        if progress is not None:progress(family)
    return dict(blend=blend(bags['xgboost'],bags['lasagne_neural'],bags['adaboost_extratrees']),selection=selection)
