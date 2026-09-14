"""Full source-shaped ResNet-v2-101 for momentum-attack inference.

Apache2 TensorFlow Authors/dongyp13; Adversarial-non_targeted license notice.
CPU NCHW adaptation with frozen statistics and explicit random/supplied state.
Torch parameter keys and random initialization are not TensorFlow checkpoint names
or historical initializers. No automatic weight downloads or checkpoint decoding.
"""
import torch
from torch import nn
from torch.nn import functional as F

from sciona.adversarial_normalization import FrozenBatchNorm
from sciona.adversarial_resnet_spatial import conv2d_same


def same_max_pool(x):
    height,width=x.shape[-2:]
    ph=max(((height+1)//2-1)*2+3-height,0)
    pw=max(((width+1)//2-1)*2+3-width,0)
    return F.max_pool2d(F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2),value=float('-inf')),3,2)


class Bottleneck(nn.Module):
    def __init__(self,in_channels,depth,base,stride):
        super().__init__();self.stride=stride
        self.preact=FrozenBatchNorm(in_channels,epsilon=1e-5,scale=True)
        self.shortcut=nn.Conv2d(in_channels,depth,1,stride=stride) if in_channels!=depth else None
        self.conv1=nn.Conv2d(in_channels,base,1,bias=False)
        self.bn1=FrozenBatchNorm(base,epsilon=1e-5,scale=True)
        self.conv2=nn.Conv2d(base,base,3,bias=False)
        self.bn2=FrozenBatchNorm(base,epsilon=1e-5,scale=True)
        self.conv3=nn.Conv2d(base,depth,1)

    def forward(self,x):
        pre=F.relu(self.preact(x))
        shortcut=self.shortcut(pre) if self.shortcut is not None else x[:,:,::self.stride,::self.stride]
        residual=F.relu(self.bn1(self.conv1(pre)))
        residual=conv2d_same(residual,self.conv2.weight,stride=self.stride)
        residual=F.relu(self.bn2(residual))
        return shortcut+self.conv3(residual)


class ResNetV2_101(nn.Module):
    def __init__(self):
        super().__init__()
        self.root=nn.Conv2d(3,64,7)
        self.blocks=nn.ModuleList();channels=64
        for base,units,stride in ((64,3,2),(128,4,2),(256,23,2),(512,3,1)):
            block=nn.ModuleList()
            for i in range(units):
                block.append(Bottleneck(channels,base*4,base,stride if i==units-1 else 1))
                channels=base*4
            self.blocks.append(block)
        self.postnorm=FrozenBatchNorm(2048,epsilon=1e-5,scale=True)
        self.logits=nn.Conv2d(2048,1001,1)

    def forward(self,x):
        x=same_max_pool(conv2d_same(x,self.root.weight,self.root.bias,stride=2))
        for block in self.blocks:
            for unit in block:x=unit(x)
        x=F.relu(self.postnorm(x)).mean(dim=(2,3),keepdim=True)
        logits=self.logits(x).squeeze(-1).squeeze(-1)
        return logits,{'predictions':logits.softmax(1)}


def build_resnet101(*,seed,state=None):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('invalid initialization seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=ResNetV2_101()
    if state is not None:
        if not isinstance(state,dict) or state.keys()!=model.state_dict().keys():
            raise ValueError('state keys must exactly match ResNet adaptation')
        for key,expected in model.state_dict().items():
            value=state[key]
            if (not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.dtype!=expected.dtype
                    or value.layout!=torch.strided or not torch.isfinite(value).all()
                    or key.endswith('moving_variance') and torch.any(value<0)):
                raise ValueError('invalid ResNet state tensor')
        model.load_state_dict(state,strict=True)
    return model.eval()
