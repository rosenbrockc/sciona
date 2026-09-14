"""Explicit-parameter s32 gradient/update path; EMA and scheduling remain external.

Preserves separate cuDNN input/recurrent weights and biases as optimizer variables.
"""
import torch
from .webtraffic_training_forward import s32_forward
from .webtraffic_adam import initial_state,adam_step

PARAMETER_NAMES=('encoder_wi','encoder_wh','encoder_bi','encoder_bh',
                 'decoder_gate_kernel','decoder_gate_bias','decoder_candidate_kernel',
                 'decoder_candidate_bias','projection_kernel','projection_bias')


def parameter_views(parameters):
    if set(parameters)!=set(PARAMETER_NAMES):
        raise ValueError('Complete named model parameters required')
    p=parameters;depth=267
    encoder=dict(gate_kernel=torch.cat([p['encoder_wi'][:2*depth].T,p['encoder_wh'][:2*depth].T],dim=0),
                 gate_bias=p['encoder_bi'][:2*depth]+p['encoder_bh'][:2*depth],
                 candidate_input_kernel=p['encoder_wi'][2*depth:].T,
                 candidate_hidden_kernel=p['encoder_wh'][2*depth:].T,
                 candidate_input_bias=p['encoder_bi'][2*depth:],candidate_hidden_bias=p['encoder_bh'][2*depth:])
    decoder={key:p['decoder_'+key] for key in ['gate_kernel','gate_bias','candidate_kernel','candidate_bias']}
    return [encoder],[decoder],p['projection_kernel'],p['projection_bias']


def new_optimizer(parameters):
    parameter_views(parameters)
    return dict(parameter_names=PARAMETER_NAMES,adam=initial_state([parameters[k] for k in PARAMETER_NAMES]))


def forward(parameters,batch,training=True):
    enc,dec,projection,bias=parameter_views(parameters)
    return s32_forward(batch['x'],batch['y'],batch['previous'],batch['mean'],batch['std'],batch['truth'],
                       enc,dec,projection,bias,gate_mask=batch.get('gate_mask'),
                       decoder_masks=batch.get('decoder_masks'),training=training)


def train_step(parameters,optimizer,batch):
    """Functional update; named inputs and optimizer state are left unchanged."""
    if tuple(optimizer['parameter_names'])!=PARAMETER_NAMES:
        raise ValueError('Optimizer slots must match canonical parameter order')
    leaves={key:parameters[key].detach().clone().requires_grad_(True) for key in PARAMETER_NAMES}
    result=forward(leaves,batch)
    ordered=[leaves[key] for key in PARAMETER_NAMES]
    gradients=torch.autograd.grad(result['total_loss'],ordered)
    updated,state,norm=adam_step(ordered,gradients,optimizer['adam'])
    metrics={key:result[key].detach() for key in ['total_loss','smooth','encoder_penalty','decoder_penalty']}
    metrics['gradient_norm']=norm
    return dict(zip(PARAMETER_NAMES,updated)),dict(parameter_names=PARAMETER_NAMES,adam=state),metrics
