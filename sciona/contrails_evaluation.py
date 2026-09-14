"""Source validation metrics for prepared Contrails batches.

Adapted from Jun Koda's MIT-licensed final evaluate.py at
08a15beb36f9cbed4c3990e74c625c1332b61fe8. Timing/logging are excluded.
"""
import torch
from segmentation_models_pytorch.losses import DiceLoss

from sciona.contrails_losses import BCELoss


def evaluate(model, batches, *, threshold=.45, augment_fraction=.95, device='cpu'):
    """Global hard Dice plus source sample-weighted BCE and batch soft Dice.

    Targets must be present in every batch. Undefined empty-positive hard Dice
    raises explicitly; model mode is restored even on invalid evaluation input.
    """
    if not 0 <= threshold <= 1 or not 0 <= augment_fraction <= 1:
        raise ValueError('Invalid threshold or augmentation fraction')
    criterion = BCELoss()
    dice = DiceLoss('binary', from_logits=True)
    was_training = model.training
    model.eval()
    n_sum = 0
    loss_sum = dice_sum = 0.
    tp = positives_pred = positives_true = 0.
    try:
        for batch in batches:
            x, y, y_sym, label = (batch[k].to(device) for k in ['x', 'y', 'y_sym', 'label'])
            if len(x) == 0:
                raise ValueError('Empty evaluation batch')
            with torch.no_grad():
                sym, pred = model(x)
                loss = criterion(sym, y_sym, pred, y, 1 - augment_fraction)
                soft_loss = dice(pred, y)
            if not torch.isfinite(loss) or not torch.isfinite(soft_loss):
                raise ValueError('Nonfinite evaluation loss')
            n_sum += len(x)
            loss_sum += loss.item() * len(x)
            dice_sum += soft_loss.item() * len(x)
            hard = pred.sigmoid() > threshold
            tp += (hard * label).sum().item()
            positives_pred += hard.sum().item()
            positives_true += label.sum().item()
        denominator = positives_pred + positives_true
        if n_sum == 0 or denominator == 0:
            raise ValueError('Undefined evaluation for empty population or empty positive union')
        return {'score': 2 * tp / denominator, 'loss': loss_sum / n_sum,
                'dice': 1 - dice_sum / n_sum}
    finally:
        model.train(was_training)
