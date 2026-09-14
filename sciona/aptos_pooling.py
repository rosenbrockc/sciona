"""Trainable generalized-mean pooling for the APTOS reconstruction.

Adapted from cnnimageretrieval-pytorch (Copyright 2016-2019 VRG, CTU Prague),
under MIT; see docs/licenses/APTOS-pooling-MIT.txt. Local changes simplify the
module wrapper and document its numerical scope. The exponent remains an
unconstrained learned parameter, as in the cited source. Zero exponents and
overflow are not made safe by clipping or a different parameterization.
"""
import torch
from torch import nn
from torch.nn import functional as F


def gem(x, p=3, eps=1e-6):
    """Clamp, exponentiate, spatially average and apply the reciprocal power.

    Intended input is a nonempty floating NCHW activation tensor. Retains the
    source arithmetic order and a singleton output for each spatial dimension.
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
