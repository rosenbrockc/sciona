"""Hard-negative BCE for the winner's 1295-way OOD classifier.

Every positive target contributes. Negatives contribute only when detached
sigmoid confidence is strictly above 0.1. Normalize each sample by its selected
class count before averaging over the batch.
"""
import torch
from torch.nn import functional as F


def ood_training_loss(logits,labels):
    if (not isinstance(logits,torch.Tensor) or logits.device.type!='cpu'
            or logits.dtype!=torch.float32 or logits.ndim!=2 or logits.shape[0]<1
            or logits.shape[1]!=1295 or not torch.isfinite(logits).all()):
        raise ValueError('finite float32 CPU N1295 OOD logits required')
    if (not isinstance(labels,torch.Tensor) or labels.device.type!='cpu' or labels.dtype!=torch.int64
            or labels.shape!=(len(logits),) or (labels<0).any() or (labels>=1295).any()):
        raise ValueError('aligned integer OOD-class targets in0..1294 required')
    targets=torch.zeros_like(logits).scatter_(1,labels[:,None],1.)
    selected=(1-targets)*(logits.detach().sigmoid()>.1)+targets
    losses=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return ((losses*selected).sum(1)/selected.sum(1)).mean()
