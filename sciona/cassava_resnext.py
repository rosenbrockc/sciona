"""Winning ResNeXt topology and epoch mechanics in the current timm runtime."""
import numpy as np
import torch
from torch import nn


class Classifier(nn.Module):
    """Require an explicit pretrained choice; False is an untrained diagnostic."""
    def __init__(self, *, pretrained, weights_path=None, expected_sha256=None):
        super().__init__()
        from sciona.cassava_pretrained import build_backbone
        self.model = build_backbone('resnext', pretrained=pretrained,
                                    weights_path=weights_path, expected_sha256=expected_sha256)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, images):
        return self.model(images)


def _check(images, labels):
    if (not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, 512, 512)
            or images.dtype != torch.float32 or labels.dtype != torch.int64
            or labels.shape != (images.shape[0],) or labels.device != images.device
            or not torch.isfinite(images).all() or (labels < 0).any() or (labels >= 5).any()):
        raise ValueError('Finite float32 NCHW 512-pixel images and aligned integer labels required')


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)


def train_epoch(model, batches, optimizer):
    """One optimizer step per batch; source gradient norm cap is 1000."""
    model.train()
    total, count = 0., 0
    for images, labels in batches:
        _check(images, labels)
        loss = nn.functional.cross_entropy(model(images), labels)
        total += loss.item() * len(labels)
        count += len(labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1000.)
        optimizer.step()
        optimizer.zero_grad()
    if not count:
        raise ValueError('Nonempty training epoch required')
    return total / count


def validate(model, batches):
    """Population-weighted mean CE and ordered softmax probabilities."""
    model.eval()
    total, count, probabilities = 0., 0, []
    with torch.no_grad():
        for images, labels in batches:
            _check(images, labels)
            logits = model(images)
            total += nn.functional.cross_entropy(logits, labels).item() * len(labels)
            count += len(labels)
            probabilities.append(logits.softmax(1).cpu().numpy())
    if not count:
        raise ValueError('Nonempty validation epoch required')
    return total / count, np.concatenate(probabilities)
