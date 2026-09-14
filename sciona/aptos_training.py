"""Smooth-L1 training primitive for both stages of the APTOS reference.

The source establishes the regression loss and full model fine-tuning.
Optimizer, schedule, batch construction and first-stage budget remain explicit
caller choices. This primitive does not select labels or skip augmentation.
"""
import torch


def train_epochs(model, optimizer, epoch_batches, *, epochs):
    """Train every model parameter using a fresh batch iterable for each epoch.

    epoch_batches(epoch_index) supplies normalized NCHW tensors and N-by-1
    continuous targets. Returns aggregate epoch losses only. No private input
    identities or contents are logged. Optimizer state persists across epochs.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('Positive integer epoch budget required')
    parameters = list(model.parameters())
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if (not parameters or not all(p.requires_grad for p in parameters)
            or len(optimized) != len(parameters)
            or {id(p) for p in optimized} != {id(p) for p in parameters}):
        raise ValueError('Optimizer must cover every trainable model parameter exactly once')
    loss_function = torch.nn.SmoothL1Loss(reduction='mean', beta=1.)
    losses = []
    model.train()
    for epoch in range(epochs):
        weighted_loss, count = 0., 0
        for images, targets in epoch_batches(epoch):
            if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3
                    or targets.shape != (images.shape[0], 1)
                    or not images.is_floating_point() or not targets.is_floating_point()
                    or not torch.isfinite(images).all() or not torch.isfinite(targets).all()):
                raise ValueError('Finite normalized NCHW RGB inputs and N-by-1 soft targets required')
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            if predictions.shape != targets.shape or not torch.isfinite(predictions).all():
                raise ValueError('Finite scalar regression output per image required')
            loss = loss_function(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite regression loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('All trainable parameters must receive finite gradients')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise ValueError('Nonfinite updated model parameters')
            weighted_loss += float(loss.detach()) * images.shape[0]
            count += images.shape[0]
        if count == 0:
            raise ValueError('Every epoch must contain training observations')
        losses.append(weighted_loss / count)
    return losses
