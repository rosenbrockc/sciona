"""Verified tensor-only classifier checkpoint handoff into frozen GAN guidance."""
import hashlib
import io
from pathlib import Path
import re

import torch
from torch import nn


def load_frozen_classifier(model,path,*,expected_sha256):
    if not isinstance(model,nn.Module) or not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256):
        raise ValueError('classifier module and explicit checkpoint SHA256 required')
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256:
        raise ValueError('classifier checkpoint content differs from receipt')
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    expected=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(expected):
        raise ValueError('classifier checkpoint tensor inventory differs')
    for key,value in state.items():
        target=expected[key]
        if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided
                or value.shape!=target.shape or value.dtype!=target.dtype
                or target.device.type!='cpu' or not torch.isfinite(value).all()):
            raise ValueError('classifier checkpoint tensor contract differs')
    # All content and tensor checks finish before changing the supplied model.
    model.load_state_dict(state,strict=True)
    model.requires_grad_(False)
    model.eval()
    return model
