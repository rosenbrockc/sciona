"""Pinned Slim ResNet conv2d_same geometry, adapted to Torch NCHW/OIHW.

Apache-2.0 TensorFlow Authors source vendored by dongyp13. See
Adversarial-non_targeted-Apache-2.0.txt. Explicit padding preserves feature
alignment independently of stride; it is not generic strided SAME padding.
"""
import torch
from torch.nn import functional as F


def conv2d_same(inputs, weight, bias=None, *, stride=1, rate=1):
    if type(stride) is not int or stride<=0 or type(rate) is not int or rate<=0:
        raise ValueError('stride and dilation must be positive integers')
    if (not isinstance(inputs,torch.Tensor) or not isinstance(weight,torch.Tensor)
            or inputs.ndim!=4 or weight.ndim!=4 or min(inputs.shape)<=0 or min(weight.shape)<=0
            or weight.shape[1]!=inputs.shape[1] or weight.shape[2]!=weight.shape[3]
            or inputs.dtype not in (torch.float32,torch.float64) or weight.dtype!=inputs.dtype
            or inputs.device.type!='cpu' or weight.device!=inputs.device):
        raise ValueError('expected matching CPU floating NCHW input and square OIHW kernel')
    if bias is not None and (not isinstance(bias,torch.Tensor) or bias.shape!=(weight.shape[0],)
                            or bias.dtype!=inputs.dtype or bias.device!=inputs.device):
        raise ValueError('invalid convolution bias')
    kernel=weight.shape[2]
    total=(kernel-1)*rate
    before=total//2;after=total-before
    return F.conv2d(F.pad(inputs,(before,after,before,after)),weight,bias,stride=stride,dilation=rate)
