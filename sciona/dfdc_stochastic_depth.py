"""Historical timm0.4.12 stochastic depth for DFDC's B7 encoder.

Derived from Ross Wightman's timm, Apache-2.0; docs/licenses/Timm-Apache-2.0.txt.
The source function is unchanged. Module replacement adapts the current timm
encoder without altering parameter names, state tensors or shared library code.
"""
import torch
from torch import nn
from timm.layers import DropPath

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class HistoricalDropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def restore_stochastic_depth(encoder):
    """Replace current DropPath modules in-place; return replacement count."""
    replaced = 0
    for module in list(encoder.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, DropPath):
                if not child.scale_by_keep:
                    raise ValueError('historical stochastic depth requires keep scaling')
                replacement = HistoricalDropPath(child.drop_prob)
                replacement.train(child.training)
                setattr(module, name, replacement)
                replaced += 1
    return replaced
