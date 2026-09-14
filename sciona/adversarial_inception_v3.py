"""Full pinned competition Inception-v3, including its inference auxiliary head.

Topology derived from TensorFlow Authors' Apache-2.0 source; see
 docs/licenses/Adversarial-non_targeted-Apache-2.0.txt.
Torch CPU/NCHW arithmetic, initializers and state naming are explicit adaptations.
"""
import json
from pathlib import Path

import torch
from torch import nn
from sciona.adversarial_inception_ops import InceptionConv, check_input, pool2d


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.topology = json.loads(Path(__file__).with_name('adversarial_inception_v3_topology.json').read_text())
        self.layers = nn.ModuleDict()
        for i, node in enumerate(self.topology['nodes']):
            if node['op'] == 'conv':
                self.layers[str(i)] = InceptionConv(**{k:node[k] for k in
                    ('in_channels','out_channels','kernel','stride','padding','normalized','activation')})

    def forward(self, x):
        check_input(x)
        if tuple(x.shape[1:]) != (3,299,299):
            raise ValueError('source full-resolution RGB shape (N,3,299,299) required')
        values = {'input':x}
        for i, node in enumerate(self.topology['nodes']):
            args = [values[key] for key in node['inputs']]
            op = node['op']
            if op == 'conv': out = self.layers[str(i)](args[0])
            elif op == 'pool': out = pool2d(args[0], **{k:node[k] for k in ('kernel','stride','padding','mode')})
            elif op == 'concat': out = torch.cat(args, dim=1)
            elif op == 'identity': out = args[0]
            elif op == 'squeeze':
                if args[0].shape[-2:] != (1,1): raise ValueError('head spatial dimensions must equal one')
                out = args[0][:,:,0,0]
            elif op == 'softmax': out = torch.softmax(args[0], dim=1)
            else: raise ValueError('unknown topology operation')
            expected = node['shape'][1:]
            expected = [expected[2], *expected[:2]] if len(expected)==3 else expected
            if list(out.shape[1:]) != expected: raise ValueError('source topology shape mismatch')
            values[node['name']] = out
        endpoints = {k:values[v] for k,v in self.topology['endpoints'].items()}
        return values[self.topology['logits']], endpoints


def build_inception_v3(*, seed, state=None):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('explicit nonnegative int64 seed required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = InceptionV3()
    if state is not None:
        expected = model.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys():
            raise ValueError('complete exact model state required')
        for key, value in state.items():
            if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                    or value.layout!=torch.strided or value.shape!=expected[key].shape
                    or value.dtype!=expected[key].dtype or not torch.isfinite(value).all()):
                raise ValueError('state tensor must match finite CPU shape and dtype')
            if key.endswith('moving_variance') and torch.any(value<0):
                raise ValueError('nonnegative normalization variance required')
        model.load_state_dict(state, strict=True)
    return model.eval()
