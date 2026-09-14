"""CHAMPS population epochs, checkpoint selection and complete private state.

Artifacts contain runtime vocabulary/scaling and must stay in caller-owned
private storage. Checkpoints use tensor/basic types, never pickled model classes.
"""
import copy
from dataclasses import asdict,replace
import math
import os
from pathlib import Path
import tempfile

import torch

from sciona.champs_population import population_batches
from sciona.champs_preprocessing import PreprocessingState
from sciona.champs_source_runtime import EXECUTION_VERSION
from sciona.champs_training import ChampsTrainer,TrainingOptions


CHECKPOINT_VERSION='champs-lifecycle.v1'


def write_checkpoint(path,trainer,preprocessing,*,epoch,metric):
    path=Path(path)
    payload={'format':CHECKPOINT_VERSION,'training':trainer.state_dict(),
             'preprocessing':asdict(preprocessing),'epoch':epoch,'selection_metric':metric}
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.champs-',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path,runtime,variant):
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if payload.get('format')!=CHECKPOINT_VERSION:
        raise ValueError('Unknown CHAMPS checkpoint format')
    training=payload['training']
    if (training['execution_version']!=EXECUTION_VERSION or training['commit']!=runtime.commit
            or training['variant']!=variant):
        raise ValueError('CHAMPS checkpoint source/model identity mismatch')
    options=TrainingOptions(**training['options'])
    options.validate()
    return payload,PreprocessingState(**copy.deepcopy(payload['preprocessing']))


def selected_metric(validation_result):
    """Keep original eight-type macro log-MAE; never select on missing types."""
    per_type=validation_result[2]
    if per_type.shape!=(8,) or not torch.isfinite(per_type).all():
        raise ValueError('CHAMPS checkpoint selection requires finite metrics for all eight types')
    metric=float(validation_result[1])
    if not math.isfinite(metric):
        raise ValueError('CHAMPS selection metric is not finite')
    return metric


def fit_variant(runtime,variant,packed,preprocessing,options,*,epochs,checkpoint_path,validation=None):
    """Train source full-population mode or retain best validation checkpoint.

    Full mode saves the last epoch, as upstream. Validation mode uses strict
    lower macro log-MAE and keeps the earlier checkpoint on ties. Per-epoch
    batch generators isolate shuffle from model randomness explicitly.
    """
    if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs<=0:
        raise ValueError('CHAMPS epochs must be a positive integer')
    options.validate()
    training_loader=population_batches(packed,batch_size=options.batch_size,
        shuffle=True,drop_last=True,seed=options.seed)
    if not len(training_loader):
        raise ValueError('CHAMPS training population has no full batches')
    if validation is None and options.scheduler=='dev_perf':
        raise ValueError('Validation-driven scheduling requires a validation population')
    if validation is not None:
        loader=population_batches(validation,batch_size=options.batch_size,shuffle=True,drop_last=True,seed=options.seed)
        if not len(loader):
            raise ValueError('CHAMPS validation population has no full batches')
    options=replace(options,max_step=epochs*len(training_loader))
    model=runtime.create_model(variant,seed=options.seed)
    trainer=ChampsTrainer(runtime,variant,model,options)
    best=float('inf')
    selected_epoch=None
    history=[]
    for epoch in range(epochs):
        training=trainer.epoch(population_batches(packed,batch_size=options.batch_size,
            shuffle=True,drop_last=True,seed=(options.seed+2*epoch)%2**32))
        metric=None
        if validation is not None:
            result=trainer.epoch(population_batches(validation,batch_size=options.batch_size,
                shuffle=True,drop_last=True,seed=(options.seed+2*epoch+1)%2**32),training=False)
            metric=selected_metric(result)
        save=validation is None or metric<best
        # Store post-scheduler state so resumption begins at the next epoch.
        if metric is not None:
            trainer.validation_schedule_step(metric)
        if save:
            write_checkpoint(checkpoint_path,trainer,preprocessing,epoch=epoch,metric=metric)
            selected_epoch=epoch
            if metric is not None: best=metric
        history.append({'epoch':epoch,'training_mae':float(training[0]),'validation_log_mae':metric,'selected':save})
    # Release source namespace/model reference cycle before caller loads a model.
    trainer._namespace.clear()
    return {'selected_epoch':selected_epoch,'epochs':epochs,'history':history}


def load_for_prediction(runtime,variant,checkpoint_path):
    payload,preprocessing=read_checkpoint(checkpoint_path,runtime,variant)
    model=runtime.create_model(variant,seed=payload['training']['options']['seed'])
    model.load_state_dict(payload['training']['model'],strict=True)
    model.eval()
    return model,preprocessing
