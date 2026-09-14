"""Complete Community s32 runtime with declared stochastic/scheduling behavior.

Source architecture, feature order, loss, optimizer, schedule and ensemble retained.
CPU NumPy/PyTorch RNG replaces legacy TF streams. EMA observation phases are explicit.
"""
from dataclasses import dataclass
import itertools
import numpy as np
import pandas as pd
import torch
from .webtraffic_assembly import assemble_features
from .webtraffic_batches import window_batches
from .webtraffic_initialization import uniform_shapes,initialize_from_uniforms
from .webtraffic_lifecycle import batch_inputs,initial_lifecycle,advance,save_checkpoint,predict_checkpoints
from .webtraffic_schedule import training_schedule


@dataclass(frozen=True)
class RuntimeConfig:
    seed:int=2
    batch_size:int=256
    max_epoch:int=100
    max_steps:int=11500
    save_from_step:int=10500
    prediction_batch_size:int=2048
    train_completeness:float=.01
    ema_parameter_phase:str='after'
    ema_step_phase:str='after'

    def validate(self):
        for name in ['seed','max_steps','save_from_step']:
            value=getattr(self,name)
            if type(value) is not int or value<0:raise ValueError('Nonnegative integer '+name+' required')
        if self.seed>2**32-4:raise ValueError('Seed must support three uint32 model streams')
        for name in ['batch_size','max_epoch','prediction_batch_size']:
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('Positive integer '+name+' required')
        if not np.isfinite(self.train_completeness) or not 0<=self.train_completeness<=1:
            raise ValueError('Completeness must be finite and in [0,1]')
        if self.ema_parameter_phase not in {'before','after'} or self.ema_step_phase not in {'before','after'}:
            raise ValueError('EMA phases must be before or after')


@dataclass
class PreparedRuntime:
    tensors:dict
    plain:dict
    config:RuntimeConfig
    schedule:list


def prepare_runtime(frame,config):
    config.validate()
    if len(frame.columns)<=409:
        raise ValueError('Source training requires more than train_window+2*predict_window days')
    tensors,plain=assemble_features(frame,add_days=63)
    # Source can hang when an infinite repeated population rejects every window.
    # Require at least one potentially nonzero training series before creating streams.
    values=np.asarray(tensors['hits'])
    zero=(~np.isfinite(values))|(values==0)
    prefix=np.pad(np.cumsum(zero,axis=1),((0,0),(1,0)))
    high=plain['data_days']-346
    empties=prefix[:,283:283+high]-prefix[:,:high]
    if not np.any(empties<=round(283*(1-config.train_completeness))):
        raise ValueError('No possible source training window passes completeness')
    schedule=list(training_schedule(plain['n_pages'],batch_size=config.batch_size,max_epoch=config.max_epoch,
                                   max_steps=config.max_steps,save_from_step=config.save_from_step))
    if not any(save for _,_,save in schedule):raise ValueError('Schedule produces no checkpoints for inference')
    return PreparedRuntime(tensors,plain,config,schedule)


def train_runtime(prepared):
    cfg=prepared.config;tensors=prepared.tensors;days=prepared.plain['data_days'];n=prepared.plain['n_pages']
    streams=[];generators=[];first=[];parameters=[]
    for model_index in range(3):
        seed=cfg.seed+model_index
        rng=np.random.RandomState(seed)
        order=rng.permutation(n)
        def offsets(random=rng):
            for _ in itertools.count():yield int(random.randint(0,days-346))
        stream=window_batches(tensors,order,offsets(),data_days=days,batch_size=cfg.batch_size,
                              epochs=None,completeness=cfg.train_completeness)
        record=next(stream);first.append(record);streams.append(stream)
        generator=torch.Generator(device='cpu').manual_seed(seed);generators.append(generator)
        xdepth,ydepth=record[1].shape[-1],record[5].shape[-1]
        draws={name:torch.rand(shape,generator=generator) for name,shape in uniform_shapes(xdepth,ydepth).items()}
        parameters.append(initialize_from_uniforms(draws,xdepth,ydepth))
    state=initial_lifecycle(parameters);checkpoints=[]
    for step,_,save in prepared.schedule:
        records=first if step==1 else [next(stream) for stream in streams]
        batches=[]
        for record,generator in zip(records,generators):
            batch=len(record[0]);gate=torch.rand(batch,267,generator=generator)<.9967589439360334
            masks=[dict(state=torch.rand(63,batch,267,generator=generator)<.99,
                        output=torch.rand(63,batch,267,generator=generator)<.975)]
            batches.append(batch_inputs(record,gate,masks))
        observed=step if cfg.ema_step_phase=='after' else step-1
        observations=[dict(parameter_phase=cfg.ema_parameter_phase,global_step=observed) for _ in range(3)]
        state,_=advance(state,batches,observations)
        if save:checkpoints=save_checkpoint(checkpoints,state)
    return dict(state=state,checkpoints=checkpoints)


def forecast_runtime(prepared,trained):
    tensors=prepared.tensors;plain=prepared.plain
    records=list(window_batches(tensors,np.arange(plain['n_pages']),[],data_days=plain['data_days'],
                                epochs=1,training=False,batch_size=prepared.config.prediction_batch_size,completeness=.01))
    result,_=predict_checkpoints(trained['checkpoints'],records,pd.Index(tensors['page_ix']))
    result.columns=pd.date_range(plain['data_end']+pd.Timedelta(days=1),periods=63)
    return result


def execute_runtime(frame,config):
    prepared=prepare_runtime(frame,config)
    trained=train_runtime(prepared)
    return forecast_runtime(prepared,trained)
