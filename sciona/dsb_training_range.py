"""Consecutive source epochs over one shared model and optimizer runtime."""
from sciona.dsb_components import _integer
from sciona.dsb_schedule import epoch_steps
from sciona.dsb_training import run_epoch_lifecycle


def run_training_range(runtime,training_factory,validation_factory,*,end_epoch,
                       classifier_variant=3,freeze_batchnorm=False,prepare_epoch=None):
    """Keep momentum, shared model buffers and RNG streams across the epoch range.

    runtime comes from one restore_training_runtime call. prepare_epoch, when
    supplied, receives each epoch before its passes and can install caller-owned
    sample orders. The factory is never recreated or reseeded here. Returned
    checkpoint bytes contain only the final model; metrics contain no model states.
    """
    start=_integer(runtime['start_epoch'],'start_epoch')
    end=_integer(end_epoch,'end_epoch')
    if end<start:raise ValueError('end_epoch precedes restored start_epoch')
    # Reject unsupported ranges before the first optimizer update.
    for epoch in range(start,end+1):epoch_steps(epoch,start,classifier_variant)
    reports=[]
    for epoch in range(start,end+1):
        if prepare_epoch is not None:prepare_epoch(epoch)
        result=run_epoch_lifecycle(**runtime,training_factory=training_factory,
            validation_factory=validation_factory,epoch=epoch,classifier_variant=classifier_variant,
            freeze_batchnorm=freeze_batchnorm,save_frequency=1)
        reports.append(dict(epoch=epoch,training=result['training'],validation=result['validation']))
    return dict(checkpoint=result['checkpoint'],epochs=reports)
