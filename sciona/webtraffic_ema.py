"""TensorFlow1.10 EMA variable arithmetic over explicit named snapshots.

The caller supplies the observed shared step and parameter values; this module
makes no scheduling choice between Adam, global-step increment, and EMA reads.
"""
import torch


def initial_ema(parameters):
    if not parameters or any(p.dtype!=torch.float32 or not torch.isfinite(p).all() for p in parameters.values()):
        raise ValueError('Nonempty finite float32 parameter mapping required')
    return {name:p.detach().clone() for name,p in parameters.items()}


@torch.no_grad()
def update_ema(shadows, snapshot, observed_global_step):
    if type(observed_global_step) is not int or observed_global_step<0:
        raise ValueError('Observed nonnegative shared step required')
    if not shadows or shadows.keys()!=snapshot.keys():
        raise ValueError('Snapshot must match all shadow parameter names')
    reference=next(iter(shadows.values()))
    count=reference.new_tensor(observed_global_step,dtype=torch.float32)
    decay=torch.minimum(count.new_tensor(.99),(1+count)/(10+count))
    result={}
    for name,shadow in shadows.items():
        value=snapshot[name]
        if (shadow.dtype!=torch.float32 or value.dtype!=torch.float32 or shadow.shape!=value.shape
            or shadow.device!=value.device or shadow.device!=reference.device
            or not torch.isfinite(shadow).all() or not torch.isfinite(value).all()):
            raise ValueError('Finite matching float32 shadow/snapshot tensors required')
        result[name]=shadow-(1-decay)*(shadow-value)
    return result,decay


def ema_parameters(shadows, expected_names):
    """Materialize independent inference values; reject missing or extra parameters."""
    if set(shadows)!=set(expected_names):
        raise ValueError('EMA parameter population does not match inference model')
    return initial_ema(shadows)
