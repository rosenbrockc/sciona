"""DFDC training epoch with an explicit CPU float32 execution adaptation.

Preserves the active B7 configuration's loss, optimizer and scheduler ordering.
MIT source: Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. Apex/distributed numerical equivalence is not claimed.
"""

import torch

from sciona.dfdc_loss import balanced_bce


def train_epoch(model, optimizer, scheduler, batches, *, epoch, batches_per_epoch=2500):
    """Train on prepared image/label tensors and return aggregate observations.

    Labels enter already smoothed. Source optional OHEM/only-valid branches are
    inactive in the selected configuration. Dataset preparation is separate.
    """
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    if type(batches_per_epoch) is not int or batches_per_epoch <= 0:
        raise ValueError('batches_per_epoch must be a positive integer')
    for parameter in model.parameters():
        if parameter.device.type != 'cpu' or parameter.dtype != torch.float32:
            raise ValueError('epoch adaptation requires CPU float32 model parameters')
    model.train()
    observations = []
    loss_sum = 0.
    examples = 0
    for i, sample in enumerate(batches):
        images, labels = sample['image'], sample['labels']
        if (not isinstance(images, torch.Tensor) or images.device.type != 'cpu'
                or images.dtype != torch.float32 or not torch.isfinite(images).all()):
            raise ValueError('prepared images must be finite CPU float32 tensors')
        labels = labels.to(device='cpu', dtype=torch.float32)
        outputs = model(images)
        loss = balanced_bce(outputs, labels)
        optimizer.zero_grad()
        # Source logging calls this mutating method before backward/optimizer.step.
        observed_lr = float(scheduler.get_lr()[-1])
        applied_lr = [float(group['lr']) for group in optimizer.param_groups]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step(i + epoch * batches_per_epoch)
        count = len(images)
        loss_sum += loss.item() * count
        examples += count
        observations.append({'batch': i, 'observed_lr': observed_lr, 'applied_lr': applied_lr})
        if i == batches_per_epoch - 1:
            break
    if not observations:
        raise ValueError('training epoch requires at least one batch')
    return {'optimizer_updates': len(observations), 'examples': examples,
            'loss': loss_sum / examples, 'observations': observations}
