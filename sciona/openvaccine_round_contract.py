"""Preflight all member recipes before model loading or generation writes."""
import numpy as np
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_models import validate_inputs

_FIELDS={'supervised_batches','validation_batch','eligible','maximum_uncertainty',
         'perturbations','sample_weights','reverse_flags'}


def _weights(value,batch):
    value=np.asarray(value)
    if (value.dtype!=np.float32 or value.shape!=(batch,) or not np.isfinite(value).all()
            or (value<0).any() or not np.isfinite(value.sum()) or value.sum()<=0):
        raise ValueError('Invalid member sample weights')


def _batch(batch,*,validation):
    if not isinstance(batch,(list,tuple)) or len(batch)!=4:
        raise ValueError('Four-part model batch required')
    n,a,t,w=batch
    n,a=validate_inputs(n,a);t=np.asarray(t)
    if t.dtype!=np.float32 or t.shape!=(len(n),n.shape[1]-2,5) or np.isinf(t).any():
        raise ValueError('Invalid member targets')
    _weights(w,len(n))
    observed=np.isfinite(t)
    if not (observed & (np.asarray(w)>0)[:,None,None]).any():
        raise ValueError('Batch requires observed targets with positive weight')


def validate_recipes(recipes,nodes,adjacency,*,std_ddof):
    nodes,adjacency=validate_inputs(nodes,adjacency)
    if type(std_ddof) is not int or std_ddof not in (0,1):
        raise ValueError('Explicit integer standard deviation ddof required')
    if not isinstance(recipes,dict) or set(recipes)!=EXPECTED_MEMBERS:
        raise ValueError('Complete twenty-member recipes required')
    target_shape=(len(nodes),nodes.shape[1]-2,5)
    for recipe in recipes.values():
        if not isinstance(recipe,dict) or set(recipe)!=_FIELDS:
            raise ValueError('Exact member recipe fields required')
        batches=recipe['supervised_batches']
        if not isinstance(batches,(list,tuple)) or not batches:
            raise ValueError('Nonempty materialized supervised batches required')
        for batch in batches:_batch(batch,validation=False)
        _batch(recipe['validation_batch'],validation=True)
        eligible=np.asarray(recipe['eligible']);draws=np.asarray(recipe['perturbations'])
        if eligible.dtype!=np.bool_ or eligible.shape!=target_shape or not eligible.any():
            raise ValueError('Invalid pseudo-label eligibility')
        if draws.dtype!=np.float32 or draws.shape!=target_shape or not np.isfinite(draws).all():
            raise ValueError('Invalid pseudo-label perturbations')
        limit=recipe['maximum_uncertainty']
        if isinstance(limit,(bool,str)) or not np.isscalar(limit) or not np.isfinite(limit) or limit<0:
            raise ValueError('Invalid pseudo-label uncertainty threshold')
        _weights(recipe['sample_weights'],len(nodes))
        if not (eligible & (np.asarray(recipe['sample_weights'])>0)[:,None,None]).any():
            raise ValueError('Pseudo-label eligibility needs positive weight support')
        flags=np.asarray(recipe['reverse_flags'])
        if flags.dtype!=np.bool_ or flags.shape!=(len(nodes),):
            raise ValueError('Invalid pseudo-label reversal flags')
