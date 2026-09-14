# Copyright 2019 The Google Research Authors.
# Licensed under the Apache License, Version 2.0.
# See docs/licenses/Cassava-bi-tempered-Apache-2.0.txt.
"""PyTorch adaptation of the pinned Google bi-tempered reference.

Restricted to the reported ViT settings: t1=.8, t2=1.4, smoothing=.06.
Five iterations, smoothing and first derivatives have been compared with the
pinned winning ViT notebook's numerical helpers. No higher derivatives or
target gradients are supported. This component does not qualify full training.
"""
import torch
from torch.autograd.function import once_differentiable


class _Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, iterations):
        centered = logits - logits.amax(dim=-1, keepdim=True)
        step = centered
        for _ in range(iterations):
            partition = (1 - .4 * step).pow(-2.5).sum(-1, keepdim=True)
            step = centered * partition.pow(-.4)
        partition = (1 - .4 * step).pow(-2.5).sum(-1, keepdim=True)
        normalizer = (partition.pow(.4) - 1) / .4
        probabilities = (1 - .4 * (centered - normalizer)).pow(-2.5)
        labels = .925 * targets + .015
        terms = labels * ((labels + 1e-10).pow(.2) - probabilities.pow(.2)) / .2
        terms -= (labels.pow(1.2) - probabilities.pow(1.2)) / 1.2
        ctx.save_for_backward(probabilities, labels)
        return terms.sum(-1)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        probabilities, labels = ctx.saved_tensors
        delta = (probabilities - labels) * probabilities.pow(.6)
        escort = probabilities.pow(1.4)
        escort = escort / escort.sum(-1, keepdim=True)
        gradient = delta - escort * delta.sum(-1, keepdim=True)
        return upstream.unsqueeze(-1) * gradient, None, None


def vit_loss(logits, targets, *, iterations=5):
    """Return one loss per row for finite N-by-5 logits and soft targets.

Callers choose the batch reduction. Targets must be fixed probabilities;
only float32/float64 are accepted. Iterations control reference fixed-point
normalization, whose finite-iteration gradient is intentionally analytical.
"""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 1000:
        raise ValueError('iterations must be an integer in [1, 1000]')
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError('Tensor logits and targets required')
    if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != 5
            or targets.shape != logits.shape or targets.dtype != logits.dtype
            or targets.device != logits.device or targets.requires_grad
            or logits.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Aligned floating N-by-5 tensors with fixed targets required')
    if (not torch.isfinite(logits).all() or not torch.isfinite(targets).all()
            or (targets < 0).any() or (targets > 1).any()
            or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-7, rtol=1e-6)):
        raise ValueError('Finite logits and normalized probability targets required')
    # Prevent overflow in centering even when both endpoints are finite.
    if not torch.isfinite(logits.detach().amax(-1) - logits.detach().amin(-1)).all():
        raise ValueError('Logit range exceeds dtype capacity')
    return _Loss.apply(logits, targets, iterations)
