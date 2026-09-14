"""Winner-described Bengali seen/OOD heads over supplied B7 feature extraction.

The caller owns backbone identity, ImageNet initialization, image normalization,
training and threshold qualification. No default for those is implied here.
"""
import torch
from torch import nn


class GraphemeB7Head(nn.Module):
    def __init__(self, backbone, *, branch):
        super().__init__()
        if branch not in ('seen', 'ood'):
            raise ValueError('explicit seen or ood branch required')
        if not isinstance(backbone, nn.Module) or not callable(getattr(backbone, 'extract_features', None)):
            raise ValueError('supplied B7 feature extractor required')
        self.branch = branch
        self.backbone = backbone
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # Preserve source head construction order for explicit RNG replay.
        self.fc = nn.Linear(2560, 14784 if branch == 'seen' else 1295)
        self.ln = nn.LayerNorm(2560)

    def forward(self, images):
        if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32
                or images.device.type != 'cpu' or images.ndim != 4 or images.shape[0] < 1
                or images.shape[1:] != (3, 137, 236) or not torch.isfinite(images).all()):
            raise ValueError('finite float32 CPU N3x137x236 input required')
        features = self.backbone.extract_features(images)
        if (features.ndim != 4 or features.shape[:2] != (len(images), 2560)
                or min(features.shape[2:]) < 1 or not torch.isfinite(features).all()):
            raise ValueError('finite B7 spatial features with2560channels required')
        pooled = self._avg_pooling(features).flatten(1)
        logits = self.fc(self.ln(pooled))
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite classifier logits')
        return logits
