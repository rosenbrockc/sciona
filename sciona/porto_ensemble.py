"""Complete independent five-neural/one-tree Porto ensemble orchestration.

Each neural entry supplies explicit DAE/classifier controls and a seed. These
are not inferred historical configurations. Unsupervised preparation/DAEs see
query features; labels enter supervised learners only. No raw/DAE concatenation
or rank transformation is introduced into the final arithmetic mean.
"""
import json
import numpy as np
from sciona.porto_transforms import _matrix
from sciona.porto_preparation import prepare_populations
from sciona.porto_neural_branch import fit_branch
from sciona.porto_tree_blend import fit_tree,blend


def fit(training,labels,query,*,preparation,neural_models,tree,progress=None):
    if type(preparation) is not dict or set(preparation)!={'dropped_columns','categorical_columns','binary_columns'}:
        raise ValueError('Complete column-role configuration required')
    if type(neural_models) is not list or len(neural_models)!=5:
        raise ValueError('Exactly five explicit neural configurations required')
    for model in neural_models:
        if type(model) is not dict or set(model)!={'seed','dae_controls','neural_controls'} or any(type(model[k]) is not dict for k in ('dae_controls','neural_controls')):
            raise ValueError('Explicit seed, DAE and classifier controls required')
        if type(model['seed']) is not int or not 0<=model['seed']<2**31:raise ValueError('Valid neural seeds required')
    if type(tree) is not dict or set(tree)!={'seed','controls'} or type(tree['controls']) is not dict or type(tree['seed']) is not int or not 0<=tree['seed']<2**31:
        raise ValueError('Explicit tree seed and controls required')
    try:identities=[json.dumps(m,sort_keys=True,allow_nan=False) for m in neural_models]
    except (ValueError,TypeError):raise ValueError('Finite JSON model controls required') from None
    if len(set(identities))!=5:raise ValueError('Duplicate neural configurations are not distinct ensemble entries')
    x=_matrix(training);q=_matrix(query);y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1}:
        raise ValueError('Aligned binary populations required')
    prepared=prepare_populations(x,q,**preparation)
    neural=[];records=[]
    for index,controls in enumerate(neural_models):
        result=fit_branch(prepared['dae_training'],y,prepared['dae_query'],**controls)
        neural.append(result['probabilities'])
        records.append({k:v for k,v in result.items() if k!='probabilities'})
        if progress is not None:progress('neural',index)
    tree_result=fit_tree(prepared['tree_training'],y,prepared['tree_query'],**tree)
    if progress is not None:progress('tree',0)
    scores=blend(np.stack(neural),tree_result['probabilities'])
    return dict(probabilities=scores,neural_probabilities=np.stack(neural),tree_probabilities=tree_result['probabilities'],
        neural_records=records,tree_iterations=tree_result['boosting_iterations'],
        models=6,dae_models=5,prepared_width=prepared['prepared_width'],
        scope='Independent explicit six-model configuration; transductive feature preparation and DAE training, historical ensemble settings unqualified.')
