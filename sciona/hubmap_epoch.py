"""Connected CPU epoch over explicit minibatches, preserving source ordering."""
import numpy as np
import torch
from sciona.hubmap_training import train_batch
from sciona.hubmap_validation import validate_batches
from sciona.hubmap_checkpoint import checkpoint_bytes


def run_epoch(model,optimizer,scheduler,policy,training_batches,validation_batches,*,
              epoch,training_population_size,training_batch_size,validation_population_size,
              dice_threshold=.5,early_stopping=True,patience=10,snapshot_period=19):
    """Train, validate, decide saves and advance scheduler unless source stops.

    Training minibatches must implement source drop_last. Training loss divides
    by the entire sampled population, including the dropped tail. Checkpoints
    contain model state only. This call retains the supplied optimizer/runtime.
    """
    for value in [training_population_size,training_batch_size,validation_population_size,epoch]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive population, batch and epoch integers required')
    if training_batch_size<2 or training_population_size<training_batch_size:
        raise ValueError('At least one full training batch of two required')
    if scheduler.optimizer is not optimizer:
        raise ValueError('Scheduler must belong to the training optimizer')
    if not 0<=dice_threshold<=1 or not np.isfinite(dice_threshold):
        raise ValueError('Finite probability threshold required')
    expected_batches=training_population_size//training_batch_size
    # Materialize only the iterable of tensor references to check complete batch
    # coverage before mutating training state; arrays are not copied.
    batches=list(training_batches)
    if len(batches)!=expected_batches or any(b[0].shape[0]!=training_batch_size for b in batches):
        raise ValueError('Training batches do not implement exact drop_last coverage')
    loss_total=0;numerator=0;denominator=0
    for images,masks,labels in batches:
        loss,logits=train_batch(model,optimizer,images,masks,labels,return_logits=True)
        loss_total+=loss.item()*training_batch_size
        probabilities=torch.sigmoid(logits).numpy().reshape(training_batch_size,1,-1)
        target=masks.numpy().reshape(training_batch_size,1,-1)
        batch_numerator=0;batch_denominator=0
        for i in range(training_batch_size):
            predicted=(probabilities[i,0]>dice_threshold).astype(np.float32)
            batch_numerator+=2*np.sum(predicted*target[i,0])
            batch_denominator+=np.sum(predicted)+np.sum(target[i,0])
        numerator+=batch_numerator;denominator+=batch_denominator
    if denominator==0:
        raise ValueError('Source training Dice denominator is zero')
    validation=validate_batches(model,validation_batches,expected_examples=validation_population_size,dice_threshold=dice_threshold)
    decision=policy.decide(epoch=epoch,validation_loss=validation['loss'],validation_score=validation['dice'],
        early_stopping=early_stopping,patience=patience,snapshot_period=snapshot_period)
    payload=checkpoint_bytes(model) if decision['save'] else None
    # Source best checkpoints precede scheduler.step and snapshot checkpoints
    # follow it. Both contain identical model tensors: scheduler mutates LR only.
    if decision['step_scheduler']:
        scheduler.step()
    return dict(training=dict(loss=loss_total/training_population_size,dice=float(numerator/denominator),
                             examples=expected_batches*training_batch_size,population=training_population_size),
                validation=validation,decision=decision,checkpoint=payload)
