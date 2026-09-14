"""Raw population and all-round preflight before external execution."""
import numpy as np
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_targets import validation_rollback_required

_FIELDS={'supervised_reverse_flags','eligible','maximum_uncertainty','perturbations','sample_weights','reverse_flags'}


def _length(sequences):
    if not isinstance(sequences,(list,tuple)) or not sequences:
        raise ValueError('Nonempty explicit sequence population required')
    if any(not isinstance(s,str) or len(s)<2 or set(s)-set('ACGU') for s in sequences):
        raise ValueError('Unsupported sequence population')
    lengths={len(s) for s in sequences}
    if len(lengths)!=1:raise ValueError('Each population requires equal sequence lengths')
    return next(iter(lengths))


def validate_lifecycle(sequences,targets,pseudo_sequences,prediction_sequences,plans,membership,*,rollback_policy,absolute_tolerance,std_ddof):
    length=_length(sequences);pseudo_length=_length(pseudo_sequences);_length(prediction_sequences)
    if targets.shape!=(len(sequences),length,5):raise ValueError('Labeled target population shape mismatch')
    if rollback_policy not in ('weights_only','model_and_optimizer'):raise ValueError('Unsupported rollback policy')
    validation_rollback_required(0.,0.,absolute_tolerance=absolute_tolerance)
    if type(std_ddof) is not int or std_ddof not in (0,1):raise ValueError('Explicit integer standard deviation ddof required')
    shape=(len(pseudo_sequences),pseudo_length,5)
    for plan in plans:
        if not isinstance(plan,dict) or set(plan)!=EXPECTED_MEMBERS:raise ValueError('All twenty member plans required in every round')
        for key,p in plan.items():
            if not isinstance(p,dict) or set(p)!=_FIELDS:raise ValueError('Exact round-plan fields required')
            for name,size in [('supervised_reverse_flags',len(membership[key].train)),('reverse_flags',len(pseudo_sequences))]:
                flags=np.asarray(p[name])
                if flags.dtype!=np.bool_ or flags.shape!=(size,):raise ValueError('Invalid round reversal decisions')
            eligible=np.asarray(p['eligible']);draws=np.asarray(p['perturbations']);weights=np.asarray(p['sample_weights'])
            if eligible.dtype!=np.bool_ or eligible.shape!=shape:raise ValueError('Invalid round eligibility')
            if draws.dtype!=np.float32 or draws.shape!=shape or not np.isfinite(draws).all():raise ValueError('Invalid round perturbations')
            if (weights.dtype!=np.float32 or weights.shape!=(shape[0],) or not np.isfinite(weights).all()
                    or (weights<0).any() or not np.isfinite(weights.sum()) or weights.sum()<=0):raise ValueError('Invalid round weights')
            if not (eligible & (weights>0)[:,None,None]).any():raise ValueError('No eligible positive-weight pseudo-labels')
            threshold=p['maximum_uncertainty']
            if isinstance(threshold,(bool,str)) or not np.isscalar(threshold) or not np.isfinite(threshold) or threshold<0:
                raise ValueError('Invalid round uncertainty threshold')
