"""Bi-tempered loss preserving the winning PyTorch floating-point operation order.

The generic Google-reference component is mathematically equivalent; this
variant retains normalization-level autodiff for source-sensitive Adam updates.
"""
import torch
from torch.autograd.function import once_differentiable


def _exp(value, temperature):
    return (1. + (1. - temperature) * value).relu().pow(1. / (1. - temperature))


def _log(value, temperature):
    return (value.pow(1. - temperature) - 1.) / (1. - temperature)


class _Normalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        mu = logits.max(-1, keepdim=True).values
        initial = logits - mu
        step = initial
        for _ in range(5):
            partition = _exp(step, 1.4).sum(-1, keepdim=True)
            step = initial * partition.pow(1. - 1.4)
        partition = _exp(step, 1.4).sum(-1, keepdim=True)
        normalizer = -_log(1. / partition, 1.4) + mu
        ctx.save_for_backward(logits, normalizer)
        return normalizer

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        logits, normalizer = ctx.saved_tensors
        escort = _exp(logits - normalizer, 1.4).pow(1.4)
        return escort / escort.sum(-1, keepdim=True) * upstream


def loss(logits, targets):
    """Fixed five-class, five-iteration, smoothed per-row objective."""
    if (logits.ndim != 2 or logits.shape[1] != 5 or targets.shape != logits.shape
            or targets.requires_grad or targets.dtype != logits.dtype
            or not torch.isfinite(logits).all() or not torch.isfinite(targets).all()
            or (targets < 0).any() or (targets > 1).any()
            or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]))):
        raise ValueError('Finite five-class logits and fixed probability targets required')
    labels = (1. - .06 * 5 / 4) * targets + .06 / 4
    p = _exp(logits - _Normalizer.apply(logits), 1.4)
    terms = labels * _log(labels + 1e-10, .8) - labels * _log(p, .8)
    terms = terms - labels.pow(2. - .8) / (2. - .8) + p.pow(2. - .8) / (2. - .8)
    return terms.sum(-1)
