"""Fixed CycleGAN image networks reconstructed from the pinned Bengali notebook.

Two independent instances of each network are required for the two domains.
No classifier, pretrained weights or competition records are included.
"""
import torch
from torch import nn


def _norm(channels):
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)


class ResidualImageBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(256,256,3),
            _norm(256), nn.ReLU(inplace=True), nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,3), _norm(256))

    def forward(self, images):
        return images + self.path(images)


class BengaliGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.ReflectionPad2d(3),nn.Conv2d(3,64,7),_norm(64),nn.ReLU(inplace=True)]
        for incoming,outgoing in [(64,128),(128,256)]:
            layers.extend([nn.Conv2d(incoming,outgoing,3,stride=2,padding=1),_norm(outgoing),nn.ReLU(inplace=True)])
        layers.extend(ResidualImageBlock() for _ in range(9))
        for incoming,outgoing in [(256,128),(128,64)]:
            layers.extend([nn.ConvTranspose2d(incoming,outgoing,3,stride=2,padding=1,output_padding=1),
                           _norm(outgoing),nn.ReLU(inplace=True)])
        layers.extend([nn.ReflectionPad2d(3),nn.Conv2d(64,3,7),nn.Tanh()])
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class BengaliDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3,64,4,stride=2,padding=1),nn.LeakyReLU(.2,inplace=True)]
        for incoming,outgoing,stride in [(64,128,2),(128,256,2),(256,512,1)]:
            layers.extend([nn.Conv2d(incoming,outgoing,4,stride=stride,padding=1),
                           _norm(outgoing),nn.LeakyReLU(.2,inplace=True)])
        layers.append(nn.Conv2d(512,1,4,padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


def initialize_image_network(network, *, generator):
    """Source normal(0,.02) convolution weights and zero biases, explicit RNG."""
    if not isinstance(network,(BengaliGenerator,BengaliDiscriminator)):
        raise ValueError('Bengali image network required')
    if not isinstance(generator,torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('explicit CPU initialization generator required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in network.parameters()):
        raise ValueError('float32 CPU network required')
    with torch.no_grad():
        for layer in network.modules():
            if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight,mean=0.,std=.02,generator=generator)
                nn.init.zeros_(layer.bias)
    return network
