"""Explicit initialization for the complete DFDC MTCNN detection cascade."""

from collections.abc import Mapping

import torch

from sciona.dfdc_mtcnn_networks import MTCNN


def build_detector(*, initialization, seed, state=None):
    """Build a CPU detector from supplied tensors or explicitly random weights.

    Random initialization is for synthetic execution checks, not a claim of useful
    face detection. Caller states require all PNet/RNet/ONet parameters. No files
    are loaded and no weights are downloaded. Caller Torch RNG state is restored.
    """
    if initialization not in ('random', 'state'):
        raise ValueError('initialization must be random or state')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if initialization == 'random' and state is not None:
        raise ValueError('random initialization does not accept state')
    if initialization == 'state' and not isinstance(state, Mapping):
        raise ValueError('state initialization requires a tensor mapping')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = MTCNN(margin=0, thresholds=[.7, .8, .8], device=torch.device('cpu'))
    if initialization == 'state':
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError('state keys must exactly match all three detector networks')
        for key, template in expected.items():
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != template.shape or value.dtype != template.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('detector state tensor has invalid shape, dtype or values')
        model.load_state_dict(state, strict=True)
    return model.eval()
