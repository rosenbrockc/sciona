"""Frozen font-classifier guidance for the Bengali CycleGAN branch.

The source epoch loop calls classifier.eval() immediately after model.train().
This wrapper enforces that same fixed/eval behavior while retaining the gradient
through generated pixels. It does not implement the full pipeline.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FrozenFontGuidance(nn.Module):
    """Take ownership of a separately qualified 14784-class font classifier."""
    def __init__(self, classifier, *, weight):
        super().__init__()
        if not isinstance(classifier, nn.Module) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('classifier module and positive finite guidance weight required')
        self.classifier = classifier
        self.weight = float(weight)
        self.classifier.requires_grad_(False)
        self.classifier.eval()

    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self

    def forward(self, generated_images, labels):
        if (not isinstance(generated_images, torch.Tensor) or generated_images.dtype != torch.float32
                or generated_images.device.type != 'cpu' or generated_images.ndim != 4
                or generated_images.shape[0] < 1 or generated_images.shape[1:] != (3,224,224)
                or not torch.isfinite(generated_images).all()):
            raise ValueError('finite float32 CPU N3x224x224 generated images required')
        if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.device.type != 'cpu'
                or labels.shape != (len(generated_images),) or (labels < 0).any() or (labels >= 14784).any()):
            raise ValueError('aligned integer joint-class labels in0..14783 required')
        if self.classifier.training or any(p.requires_grad for p in self.classifier.parameters()):
            raise ValueError('font classifier must remain frozen and in eval mode')
        # No no_grad/detach: the generator must receive classification gradients.
        logits = self.classifier(generated_images)
        if logits.shape != (len(generated_images),14784) or not torch.isfinite(logits).all():
            raise ValueError('finite14784-way font logits required')
        return F.cross_entropy(logits, labels) * self.weight
