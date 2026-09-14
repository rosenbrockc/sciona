"""Explicit local initialization for the legacy APTOS reference backbones.

A caller-supplied digest proves file identity, not publisher provenance or
identity with the winning solution. Qualification must establish those claims
separately. No download, pickle fallback, or random-weight fallback is used.
"""
import hashlib
import re

import torch
from torch import nn

from sciona.aptos_models import FAMILIES
from sciona.aptos_pooling import GeM


def build_from_checkpoint(family, checkpoint, *, expected_sha256, checkpoint_format='torch'):
    """Strictly load the original classifier before replacing pooling/head.

    Inception publisher files contain 1001 classifier rows; SE-ResNeXt files
    contain 1000. Loading the original head validates the entire state before
    discarding classification parameters for a new scalar regression head.
    This head is an explicit reference choice, not recovered winner code.
    """
    if family not in FAMILIES:
        raise ValueError('A specified APTOS architecture family is required')
    if checkpoint_format not in ('torch', 'safetensors'):
        raise ValueError('Explicit torch or safetensors checkpoint format required')
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A full lowercase SHA256 is required')
    # Hash and deserialize through the same open file; do not reopen its path.
    with open(checkpoint, 'rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected_sha256:
            raise ValueError('Checkpoint SHA256 mismatch')
        stream.seek(0)
        if checkpoint_format == 'safetensors':
            from safetensors.torch import load
            state = load(stream.read())
        else:
            state = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError('A plain tensor state dictionary is required')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Checkpoint contains nonfinite parameters')
    import pretrainedmodels
    factory, pool, width, _ = FAMILIES[family]
    classes = 1001 if family.startswith('inception_') else 1000
    model = getattr(pretrainedmodels, factory)(num_classes=classes, pretrained=None)
    if (not hasattr(model, pool) or not isinstance(model.last_linear, nn.Linear)
            or model.last_linear.in_features != width
            or model.last_linear.out_features != classes):
        raise ValueError('Legacy backbone pooling or head contract differs')
    model.load_state_dict(state, strict=True)
    setattr(model, pool, GeM())
    model.last_linear = nn.Linear(width, 1)
    return model
