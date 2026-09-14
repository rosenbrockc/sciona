"""Full source-selected DFDC EfficientNet-B7 classifier with explicit state policy.

MIT 2020 Selim Seferbekov; docs/licenses/DFDC-MIT.txt. Source commit
89c6290490bac96b29193a4061b3db9dd3933e36. Uses installed timm's historical-name
adapter with historical stochastic-depth restoration. Historical binary-runtime
equivalence is not claimed; synthetic CPU source parity is validated separately.
"""

from collections.abc import Mapping

import torch
from torch import nn
from timm.models.efficientnet import tf_efficientnet_b7_ns
from sciona.dfdc_stochastic_depth import restore_stochastic_depth


class DeepFakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = tf_efficientnet_b7_ns(pretrained=False, drop_path_rate=.2)
        restore_stochastic_depth(self.encoder)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.)
        self.fc = nn.Linear(2560, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_classifier(*, initialization, seed, state=None):
    """Build a CPU model using random, encoder-only, or complete supplied state.

    Encoder-only state includes the source encoder's unused classification head.
    State keys are explicit and never automatically stripped or remapped.
    """
    if initialization not in ('random', 'encoder', 'state'):
        raise ValueError('initialization must be random, encoder or state')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if initialization == 'random' and state is not None:
        raise ValueError('random initialization does not accept state')
    if initialization != 'random' and not isinstance(state, Mapping):
        raise ValueError('state initialization requires a tensor mapping')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = DeepFakeClassifier()
    if initialization != 'random':
        target = model.encoder if initialization == 'encoder' else model
        expected = target.state_dict()
        if set(state) != set(expected):
            raise ValueError('state keys must exactly match the selected target')
        for key, template in expected.items():
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != template.shape or value.dtype != template.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError('classifier state tensor has invalid shape, dtype or values')
        target.load_state_dict(state, strict=True)
    return model
