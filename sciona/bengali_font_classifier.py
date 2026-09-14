"""Bengali font head on an explicitly supplied EfficientNet-B0 backbone.

The caller must qualify the backbone implementation and any trained checkpoint.
This module downloads no weights and makes no pretrained-accuracy claim.
"""
import torch
from torch import nn


class BengaliFontClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if not isinstance(backbone, nn.Module) or not callable(getattr(backbone,'extract_features',None)):
            raise ValueError('feature-extracting B0 backbone required')
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Preserve source parameter construction order for reproducible initialization.
        self.output = nn.Linear(1280,14784)
        self.normalization = nn.LayerNorm(1280)

    def forward(self, images):
        if (not isinstance(images,torch.Tensor) or images.dtype != torch.float32
                or images.device.type != 'cpu' or images.ndim != 4
                or images.shape[0] < 1 or images.shape[1:] != (3,224,224)
                or not torch.isfinite(images).all()):
            raise ValueError('finite float32 CPU N3x224x224 images required')
        features = self.backbone.extract_features(images)
        if (features.ndim != 4 or features.shape[:2] != (len(images),1280)
                or not torch.isfinite(features).all()):
            raise ValueError('finite B0 features with 1280 channels required')
        return self.output(self.normalization(self.pool(features).flatten(1)))
