"""Pinned alternating-training controls; caller owns batches and checkpoints."""
import math
import torch

from sciona.dsb_components import _integer

DETECTOR_STAGES = (50, 100, 140, 160)
DETECTOR_RATES = (.01, .001, .0001, .00001)
CLASSIFIER_PROFILES = {
    3: dict(stages=(50,100,140,160), rates=(.01,.001,.0001,.00001),
            flip=True, swap=False, rotate=False, scale=False),
    4: dict(stages=(50,100,140,160,180), rates=(.01,.001,.0001,.00001,.000001),
            flip=True, swap=True, rotate=True, scale=True),
}


def learning_rate(epoch, stages, rates, override=None):
    """Source strict epoch>boundary schedule, including its last-epoch bound."""
    epoch = _integer(epoch, 'epoch', 0)
    stages = tuple(_integer(s, 'stage') for s in stages)
    if (not stages or len(stages) != len(rates) or any(a >= b for a,b in zip(stages,stages[1:]))
            or epoch > stages[-1] or any(not math.isfinite(r) or r < 0 for r in rates)):
        raise ValueError('invalid learning-rate schedule or epoch')
    if override is not None:
        if not math.isfinite(override) or override < 0:
            raise ValueError('learning-rate override must be finite and nonnegative')
        return override
    return rates[sum(epoch > stage for stage in stages)]


def epoch_steps(epoch, start_epoch, classifier_variant=3, learning_rate_override=None):
    """Source main-loop ordering including first-epoch zero-LR classifier pass.

    That initial debug pass includes up to five batches and can change momentum
    and BatchNorm state even though its learning rate is zero.
    """
    epoch, start_epoch = _integer(epoch,'epoch'), _integer(start_epoch,'start_epoch')
    if classifier_variant not in CLASSIFIER_PROFILES or epoch < start_epoch:
        raise ValueError('invalid classifier variant or epoch range')
    profile = CLASSIFIER_PROFILES[classifier_variant]
    if epoch > profile['stages'][-1]:
        raise ValueError('epoch exceeds classifier schedule')
    steps = []
    if epoch == start_epoch:
        steps.append(dict(task='classifier', learning_rate=0., max_batches=5))
    if epoch < DETECTOR_STAGES[-1]:
        steps.append(dict(task='detector', learning_rate=learning_rate(epoch,DETECTOR_STAGES,DETECTOR_RATES,learning_rate_override),max_batches=None))
    if epoch > 20:
        steps.append(dict(task='classifier', learning_rate=learning_rate(epoch,profile['stages'],profile['rates'],learning_rate_override),max_batches=None))
    return steps


def training_mode(model, freeze_batchnorm=False):
    """Freeze running statistics only; retain source affine parameter gradients."""
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                module.eval()
