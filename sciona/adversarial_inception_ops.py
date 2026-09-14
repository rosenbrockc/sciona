"""CPU NCHW primitives for source Slim Inception inference.

TensorFlow Authors' Apache-2.0 reference semantics; see
docs/licenses/Adversarial-TensorFlow-reference-Apache-2.0.txt.
This is a Torch adaptation, not historical TensorFlow binary equivalence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from sciona.adversarial_normalization import FrozenBatchNorm


def pair(value):
    result = (value, value) if type(value) is int else value
    if (not isinstance(result, (tuple, list)) or len(result) != 2
            or any(type(v) is not int or v <= 0 for v in result)):
        raise ValueError('positive integer or pair required')
    return tuple(result)


def padding_for(shape, kernel, stride, padding):
    """Return Torch left/right/top/bottom padding with TF SAME alignment."""
    if padding not in ('SAME', 'VALID'):
        raise ValueError('padding must be SAME or VALID')
    kernel, stride = pair(kernel), pair(stride)
    pads = []
    for size, width, step in zip(shape, kernel, stride):
        if size <= 0 or (padding == 'VALID' and size < width):
            raise ValueError('nonempty spatial output required')
        total = max(((size + step - 1) // step - 1) * step + width - size, 0) if padding == 'SAME' else 0
        pads.append((total // 2, total - total // 2))
    return (*pads[1], *pads[0])


def check_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 4
            or x.device.type != 'cpu' or x.dtype not in (torch.float32, torch.float64)
            or any(d <= 0 for d in x.shape) or not torch.isfinite(x).all()):
        raise ValueError('nonempty finite CPU NCHW float32/64 input required')


def pool2d(x, kernel, *, stride=1, padding='SAME', mode='avg'):
    check_input(x)
    kernel, stride = pair(kernel), pair(stride)
    pads = padding_for(x.shape[-2:], kernel, stride, padding)
    if mode == 'max':
        return F.max_pool2d(F.pad(x, pads, value=-torch.inf), kernel, stride)
    if mode != 'avg':
        raise ValueError('pool mode must be avg or max')
    # Eigen TF1 uses lowest finite value as an excluded padding sentinel.
    # Reject collisions explicitly instead of interpreting real values as padding.
    if torch.any(x == torch.finfo(x.dtype).min):
        raise ValueError('input collides with historical average-pool sentinel')
    numerator = F.avg_pool2d(F.pad(x, pads), kernel, stride, divisor_override=1)
    mask = torch.ones((1, 1, *x.shape[-2:]), dtype=x.dtype, device=x.device)
    denominator = F.avg_pool2d(F.pad(mask, pads), kernel, stride, divisor_override=1)
    return numerator / denominator


class InceptionConv(nn.Module):
    """Slim conv + frozen BN + ReLU, or an explicit linear biased head.

    Defaults match Inception arg scopes: epsilon .001, center True, scale False.
    Torch initializers and state names are an explicit adaptation.
    """
    def __init__(self, in_channels, out_channels, kernel, *, stride=1,
                 padding='SAME', normalized=True, activation=True):
        super().__init__()
        if any(type(c) is not int or c <= 0 for c in (in_channels, out_channels)):
            raise ValueError('positive channel counts required')
        if type(normalized) is not bool or type(activation) is not bool:
            raise ValueError('explicit normalization and activation flags required')
        if padding not in ('SAME', 'VALID'):
            raise ValueError('padding must be SAME or VALID')
        self.kernel, self.stride = pair(kernel), pair(stride)
        self.padding, self.activation = padding, activation
        self.conv = nn.Conv2d(in_channels, out_channels, self.kernel,
                              stride=self.stride, bias=not normalized)
        self.norm = FrozenBatchNorm(out_channels, epsilon=.001, scale=False) if normalized else nn.Identity()

    def forward(self, x):
        check_input(x)
        if x.shape[1] != self.conv.in_channels or x.dtype != self.conv.weight.dtype:
            raise ValueError('input must match convolution channels and dtype')
        x = self.conv(F.pad(x, padding_for(x.shape[-2:], self.kernel, self.stride, self.padding)))
        x = self.norm(x)
        return F.relu(x) if self.activation else x
