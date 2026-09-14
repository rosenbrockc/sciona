"""Consecutive source epochs from decoded runtime populations and model state."""
import torch
from sciona.hubmap_batches import training_batches
from sciona.hubmap_loader import ordered_batches
from sciona.hubmap_epoch import run_epoch
from sciona.hubmap_checkpoint_policy import CheckpointPolicy


def run_training_range(model,optimizer,scheduler,training,validation,*,start_epoch,end_epoch,
        maximum_bin,training_batch_size,validation_batch_size,input_side,
        python_rng,numpy_rng,torch_generator,dice_threshold=.5,early_stopping=True,patience=10,snapshot_period=19):
    """Balance, shuffle, augment, train, validate and retain source checkpoints.

    Model/Adam/CosineLR are freshly restored/constructed by the caller. Skipped
    source epochs advance the fresh scheduler only. Continuity holds within this
    range; model-only checkpoint inputs do not resume old optimizer/RNG state.
    """
    if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in [start_epoch,end_epoch]):
        raise ValueError('Positive epoch range required')
    if end_epoch<start_epoch or scheduler.last_epoch!=0:
        raise ValueError('Ordered range and freshly constructed scheduler required')
    if set(training)!={'images_bgr','rles','present','bins'} or set(validation)!={'images_bgr','rles'}:
        raise ValueError('Explicit training and validation collections required')
    if len(validation['images_bgr'])!=len(validation['rles']) or not len(validation['rles']):
        raise ValueError('Aligned nonempty validation population required')
    if isinstance(validation_batch_size,bool) or not isinstance(validation_batch_size,int) or validation_batch_size<1:
        raise ValueError('Positive validation batch size required')
    policy=CheckpointPolicy();retained={};reports=[]
    for _ in range(1,start_epoch):scheduler.step()
    for epoch in range(start_epoch,end_epoch+1):
        training_data=training_batches(**training,maximum_bin=maximum_bin,batch_size=training_batch_size,
            input_side=input_side,python_rng=python_rng,numpy_rng=numpy_rng,torch_generator=torch_generator)
        # Validation consumes its own source DataLoader iterator base seed even
        # without shuffle; share the explicit generator with training.
        def validation_data():
            indices=torch.utils.data.DataLoader(list(range(len(validation['rles']))),batch_size=validation_batch_size,
                shuffle=False,num_workers=0,generator=torch_generator)
            order=torch.cat(list(indices)).numpy()
            yield from ordered_batches(**validation,indices=order,batch_size=validation_batch_size,input_side=input_side,
                training=False,python_rng=python_rng,numpy_rng=numpy_rng)
        result=run_epoch(model,optimizer,scheduler,policy,training_data['batches'],validation_data(),
            epoch=epoch,training_population_size=training_data['population_size'],training_batch_size=training_batch_size,
            validation_population_size=len(validation['rles']),dice_threshold=dice_threshold,
            early_stopping=early_stopping,patience=patience,snapshot_period=snapshot_period)
        for role in result['decision']['save']:
            key='snapshot_'+str(epoch) if role=='snapshot' else role
            retained[key]=result['checkpoint']
        reports.append(dict(epoch=epoch,training=result['training'],validation=result['validation'],decision=result['decision']))
        if result['decision']['stop']:break
    return dict(epochs=reports,checkpoints=retained)
