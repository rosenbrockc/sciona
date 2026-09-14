"""Dense float32 Adam mathematics matching TensorFlow 1.10 defaults.

Implements the documented epsilon-hat formula with explicit restorable state.
No TensorFlow kernel bitwise parity, EMA scheduling or global-step increment claim.
"""
import torch


def initial_state(parameters):
    if not parameters:
        raise ValueError('At least one parameter required')
    reference=parameters[0]
    if any(p.dtype!=torch.float32 or p.device!=reference.device for p in parameters):
        raise ValueError('Same-device float32 parameters required')
    return dict(step=0,beta1_power=reference.new_tensor(.9),beta2_power=reference.new_tensor(.999),
                m=[torch.zeros_like(p) for p in parameters],v=[torch.zeros_like(p) for p in parameters])


@torch.no_grad()
def adam_step(parameters, gradients, state):
    """Return new parameters/state after global-norm clipping at10; inputs unchanged."""
    if not parameters or len(parameters)!=len(gradients):
        raise ValueError('Matching nonempty parameter/gradient lists required')
    if set(state)!={'step','beta1_power','beta2_power','m','v'}:
        raise ValueError('Complete Adam state required')
    if len(state['m'])!=len(parameters) or len(state['v'])!=len(parameters):
        raise ValueError('Adam slot population mismatch')
    device=parameters[0].device
    for p,g,m,v in zip(parameters,gradients,state['m'],state['v']):
        if any(x.dtype!=torch.float32 or x.device!=device or x.shape!=p.shape or not torch.isfinite(x).all()
               for x in [p,g,m,v]) or (v<0).any():
            raise ValueError('Finite float32 parameters/gradients and valid matching slots required')
    if type(state['step']) is not int or state['step']<0:
        raise ValueError('Nonnegative optimizer step required')
    for key in ['beta1_power','beta2_power']:
        power=state[key]
        if power.shape!=() or power.dtype!=torch.float32 or power.device!=device or not 0<=power<1:
            raise ValueError('Valid scalar beta power required')
    norm=torch.sqrt(torch.stack([g.square().sum() for g in gradients]).sum())
    if not torch.isfinite(norm):raise ValueError('Finite global gradient norm required')
    scale=torch.minimum(norm.new_tensor(1.),norm.new_tensor(10.)/norm)
    clipped=[g*scale for g in gradients]
    b1=norm.new_tensor(.9);b2=norm.new_tensor(.999)
    rate=norm.new_tensor(.001)*torch.sqrt(1-state['beta2_power'])/(1-state['beta1_power'])
    means=[b1*m+(1-b1)*g for m,g in zip(state['m'],clipped)]
    variances=[b2*v+(1-b2)*g.square() for v,g in zip(state['v'],clipped)]
    updated=[p-rate*m/(torch.sqrt(v)+1e-8) for p,m,v in zip(parameters,means,variances)]
    next_state=dict(step=state['step']+1,beta1_power=state['beta1_power']*b1,
                    beta2_power=state['beta2_power']*b2,m=means,v=variances)
    return updated,next_state,norm
