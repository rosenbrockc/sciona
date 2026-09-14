"""Explicit in-memory checkpoint boundaries for Contrails execution.

No files, deserialization or implicit pretrained downloads. Encoder inputs are
feature-only MaxViT state dictionaries, not unreviewed classifier checkpoints.
"""
from collections.abc import Mapping

import torch


def checked_load(module, state):
    """Preflight the entire mapping before copying any weights into a module."""
    expected = module.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError('Checkpoint keys differ from model state')
    for key, target in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Checkpoint tensor shape or dtype mismatch')
        if value.layout != torch.strided or not torch.isfinite(value).all():
            raise ValueError('Checkpoint tensors must be finite and strided')
    module.load_state_dict(state, strict=True)


def initialize_model(model, *, policy, state=None):
    """Choose supplied feature encoder, supplied full model, or explicit random init.

    A random run is valid execution evidence, not reproduction of pretrained
    competition performance. Checkpoint provenance belongs to the runtime caller.
    """
    if policy == 'encoder':
        checked_load(model.encoder, state)
    elif policy == 'full_model':
        checked_load(model, state)
    elif policy == 'random':
        if state is not None:
            raise ValueError('Random initialization must not include a checkpoint')
    else:
        raise ValueError('Explicit initialization policy required')
    return model


def terminal_checkpoint(model):
    """Capture independent CPU tensors after final training, without best-epoch selection."""
    model.eval()
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
