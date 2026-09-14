"""Fused-feature binary MLP with explicit dropout and label-smoothed log loss."""
import math
import torch
from torch import nn


class HybridHead(nn.Module):
    def __init__(self,features,hidden=32,dropout=.5):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (features,hidden)):raise ValueError('Positive network dimensions required')
        if type(dropout) not in (int,float) or not 0<=dropout<1:raise ValueError('Dropout must lie in [0,1)')
        self.layers=nn.Sequential(nn.Linear(features,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,features):
        if features.ndim!=2 or not all(features.shape) or not torch.isfinite(features).all():raise ValueError('Finite fused feature matrix required')
        return self.layers(features).squeeze(-1)


def smoothed_loss(logits,labels,*,smoothing):
    if logits.ndim!=1 or labels.shape!=logits.shape or not len(logits) or not torch.isfinite(logits).all() or not torch.isfinite(labels).all() or not ((labels==0)|(labels==1)).all():raise ValueError('Aligned finite binary labels and logits required')
    if type(smoothing) not in (int,float) or not 0<=smoothing<1:raise ValueError('Smoothing must lie in [0,1)')
    targets=labels*(1-smoothing)+.5*smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
