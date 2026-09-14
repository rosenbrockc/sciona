"""Binary training loss for the pinned tikutikutiku HuBMAP realization.

Source: tikutikutiku/kaggle-hubmap, commit
615444e86f4fb5ab916c0b66d311a307f4e4dd27 (MIT).
Float32 CPU computation; full model/training execution is a separate obligation.
"""
import torch
from torch.nn import functional as F


def _mask_pair(logits, target):
    if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ValueError('Tensor logits and binary targets required')
    if logits.dtype != torch.float32 or target.dtype != torch.float32 or logits.device.type != 'cpu' or target.device.type != 'cpu':
        raise ValueError('Float32 CPU tensors required')
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1 or not logits.numel():
        raise ValueError('Equal nonempty N1HW shapes required')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('Finite logits and binary targets required')


def binary_lovasz_hinge(logits, target):
    """Mean per-image Lovasz hinge; preserve source float32 sort/subgradient."""
    _mask_pair(logits, target)
    losses = []
    for scores, labels in zip(logits.reshape(logits.shape[0], -1), target.reshape(target.shape[0], -1)):
        errors = 1. - scores * (2. * labels - 1.)
        ordered, indices = torch.sort(errors, descending=True)
        foreground = labels[indices]
        total = foreground.sum()
        intersection = total - foreground.cumsum(0)
        union = total + (1. - foreground).cumsum(0)
        gradient = 1. - intersection / union
        if gradient.numel() > 1:
            gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
        losses.append(torch.dot(F.relu(ordered), gradient.detach()))
    # Source mean accumulates sequentially rather than using torch.stack.mean.
    result = losses[0]
    for value in losses[1:]:
        result = result + value
    return result / len(losses)


def training_loss(logits, target, *, deep_logits=(), classification_logits=None, classification_targets=None):
    """BCE + binary hinge, nonempty-only deep losses and optional image BCE.

    Every auxiliary segmentation head contributes 0.1 times BCE plus hinge on
    nonempty images only. Empty auxiliary batches have no gradient connection,
    matching source behavior. Classification labels are supplied explicitly.
    """
    _mask_pair(logits, target)
    if not isinstance(deep_logits, (tuple, list)):
        raise ValueError('Explicit sequence of deep logits required')
    for deep in deep_logits:
        _mask_pair(deep, target)
    if (classification_logits is None) != (classification_targets is None):
        raise ValueError('Both classification tensors required')
    if classification_logits is not None:
        for value in [classification_logits, classification_targets]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
                raise ValueError('Finite float32 CPU classification tensors required')
        if classification_logits.shape != (target.shape[0], 1) or classification_targets.shape != (target.shape[0],):
            raise ValueError('Classification logits N1 and labels N required')
        if not ((classification_targets == 0) | (classification_targets == 1)).all():
            raise ValueError('Binary classification labels required')
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss = loss + binary_lovasz_hinge(logits, target)
    nonempty = target.reshape(target.shape[0], -1).sum(1) != 0
    for deep in deep_logits:
        if nonempty.any():
            auxiliary = F.binary_cross_entropy_with_logits(deep[nonempty].reshape(int(nonempty.sum()), -1), target[nonempty].reshape(int(nonempty.sum()), -1))
            auxiliary = auxiliary + binary_lovasz_hinge(deep[nonempty], target[nonempty])
            loss = loss + 0.1 * auxiliary
        else:
            loss = loss + 0.1 * torch.tensor(0)
    if classification_logits is not None:
        loss = loss + F.binary_cross_entropy_with_logits(classification_logits.squeeze(-1), classification_targets)
    return loss
