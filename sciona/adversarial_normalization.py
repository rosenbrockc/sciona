"""Frozen Slim-style normalization for differentiable CPU attack inference.

Arithmetic follows explicit TensorFlow1.4 batch_normalization reference; Apache2
TensorFlow Authors; docs/licenses/Adversarial-TensorFlow-reference-Apache-2.0.txt.
Original competition binary version/fused kernels unspecified.
"""
import math
import torch
from torch import nn


class FrozenBatchNorm(nn.Module):
    def __init__(self,channels,*,epsilon=.001,scale=False):
        super().__init__()
        if type(channels) is not int or channels<=0 or type(scale) is not bool:
            raise ValueError('positive channels and explicit scale flag required')
        if type(epsilon) not in (float,int) or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('normalization epsilon must be finite and positive')
        self.epsilon=epsilon
        self.register_buffer('moving_mean',torch.zeros(channels))
        self.register_buffer('moving_variance',torch.ones(channels))
        self.register_buffer('beta',torch.zeros(channels))
        self.register_buffer('gamma',torch.ones(channels) if scale else None)

    def forward(self,x):
        if (not isinstance(x,torch.Tensor) or x.ndim!=4 or x.shape[1]!=len(self.beta)
                or x.dtype not in (torch.float32,torch.float64) or x.device.type!='cpu'
                or x.dtype!=self.beta.dtype or x.device!=self.beta.device):
            raise ValueError('normalization requires matching CPU NCHW floating input')
        if torch.any(self.moving_variance<0) or any(not torch.isfinite(v).all() for v in self.buffers()):
            raise ValueError('normalization state must be finite with nonnegative variance')
        inv=torch.rsqrt(self.moving_variance+self.epsilon)
        if self.gamma is not None:inv=inv*self.gamma
        return x*inv[None,:,None,None]+(self.beta-self.moving_mean*inv)[None,:,None,None]
