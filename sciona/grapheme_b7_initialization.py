"""Pinned publisher standard B7 initialization, loaded as tensors only.

This is a candidate ImageNet initialization. The Bengali winner did not publish
an exact checkpoint digest, so it is not an attestation of their original model.
"""
import hashlib
import io
from pathlib import Path

from efficientnet_pytorch import EfficientNet
import torch


STANDARD_B7_SHA256 = 'dcc49843b8ec5a097a8cb166eded1d7a09291093eff7735f603bcfcafce12c8e'
STANDARD_B7_URL = 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'


def load_standard_b7(path):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != STANDARD_B7_SHA256:
        raise ValueError('standard B7 publisher checkpoint content differs')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    model = EfficientNet.from_name('efficientnet-b7')
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('standard B7 checkpoint tensor inventory differs')
    for name, value in state.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all()):
            raise ValueError('standard B7 checkpoint tensor contract differs')
    model.load_state_dict(state, strict=True)
    return model
