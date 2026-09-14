"""Three-model lifecycle over explicit batches, parameters and EMA observations.

Caller-declared EMA observations are required; original TF scheduling is not inferred.
"""
import copy
import numpy as np
import pandas as pd
import torch
from .webtraffic_train_step import new_optimizer,train_step,forward,PARAMETER_NAMES
from .webtraffic_ema import initial_ema,update_ema,ema_parameters
from .webtraffic_ensemble import average_checkpoint_predictions,finalize_predictions


def batch_inputs(record,gate_mask=None,decoder_masks=None):
    """Map the original eleven-tensor InputPipe tuple to the model's runtime inputs."""
    def tensor(index):return torch.as_tensor(np.array(record[index],copy=True),dtype=torch.float32)
    return dict(x=tensor(1),y=tensor(5),previous=tensor(2)[:,-1],mean=tensor(7),std=tensor(8),truth=tensor(4),
                gate_mask=gate_mask,decoder_masks=decoder_masks)


def initial_lifecycle(parameters):
    if len(parameters)!=3:raise ValueError('Final source route requires three models')
    models=[]
    for p in parameters:
        p={k:v.detach().clone() for k,v in p.items()}
        models.append(dict(parameters=p,optimizer=new_optimizer(p),ema=initial_ema(p)))
    return dict(shared_step=0,models=models)


def advance(state,batches,observations):
    """Advance three models once using declared EMA read phases; leave inputs intact."""
    if len(state['models'])!=3 or len(batches)!=3 or len(observations)!=3:
        raise ValueError('Three models, batches and EMA observations required')
    step=state['shared_step'];models=[];metrics=[]
    for model,batch,observation in zip(state['models'],batches,observations):
        if (set(observation)!={'parameter_phase','global_step'} or observation['parameter_phase'] not in {'before','after'}
            or type(observation['global_step']) is not int or observation['global_step'] not in {step,step+1}):
            raise ValueError('Explicit before/after parameter and current/next step observation required')
        p,opt,metric=train_step(model['parameters'],model['optimizer'],batch)
        snapshot=model['parameters'] if observation['parameter_phase']=='before' else p
        ema,_=update_ema(model['ema'],snapshot,observation['global_step'])
        models.append(dict(parameters=p,optimizer=opt,ema=ema));metrics.append(metric)
    return dict(shared_step=step+1,models=models),metrics


def save_checkpoint(checkpoints,state):
    """Retain ten independent runtime snapshots, including Adam and EMA state."""
    return [*checkpoints,copy.deepcopy(state)][-10:]


@torch.no_grad()
def predict_checkpoints(checkpoints,prediction_batches,runtime_pages):
    if not checkpoints:raise ValueError('At least one checkpoint required')
    models=[]
    for model_index in range(3):
        log_predictions=[]
        for checkpoint in checkpoints:
            params=ema_parameters(checkpoint['models'][model_index]['ema'],PARAMETER_NAMES)
            records=[]
            for record in prediction_batches:
                values=forward(params,batch_inputs(record),training=False)['predictions'].numpy()
                records.append(pd.DataFrame(values,index=record[10]))
            if not records:raise ValueError('Prediction population has no accepted windows')
            log_predictions.append(pd.concat(records))
        models.append(average_checkpoint_predictions(log_predictions))
    return finalize_predictions(models,runtime_pages),models
