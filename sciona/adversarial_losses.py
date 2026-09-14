"""Branch-specific source momentum-attack ensemble objectives.

Apache-2.0 dongyp13 winning attack sources; see Adversarial license notices.
CPU Torch numerical adaptation, explicit main/aux tensors; no model loading.
"""
import torch
from torch.nn import functional as F


def ensemble_loss(main, auxiliary, labels, *, branch):
    """Mean main cross entropy plus0.4 mean auxiliary cross entropy.

    Ordering: non-targeted V3,advV3,ens3V3,ens4V3,V4,ResV2,ensAdvResV2,ResNet101
    (last has no auxiliary); targeted large V3,advV3,ens3V3,ens4V3,ensAdvResV2;
    targeted small V3,ensAdvResV2. Labels are frozen or supplied by caller.
    """
    if branch=='non_targeted':
        weights=(1.,.25,1.,1.,1.,1.,1.,1.);aux_weights=weights[:-1]
    elif branch=='targeted_large':weights=aux_weights=(4.,1.,1.,1.,4.)
    elif branch=='targeted_small':weights=aux_weights=(1.,2.)
    else:raise ValueError('unknown attack branch')
    main=tuple(main);auxiliary=tuple(auxiliary)
    if len(main)!=len(weights) or len(auxiliary)!=len(aux_weights):
        raise ValueError('branch requires exact main and auxiliary head counts')
    first=main[0]
    if (not isinstance(first,torch.Tensor) or first.ndim!=2 or min(first.shape)<1
            or first.dtype not in (torch.float32,torch.float64) or first.device.type!='cpu'):
        raise ValueError('ensemble logits must be nonempty CPU floating matrices')
    for value in main+auxiliary:
        if (not isinstance(value,torch.Tensor) or value.shape!=first.shape or value.dtype!=first.dtype
                or value.device!=first.device or not torch.isfinite(value).all()):
            raise ValueError('ensemble head shapes, dtypes and finite values must match')
    if (not isinstance(labels,torch.Tensor) or labels.dtype!=torch.int64 or labels.device.type!='cpu'
            or labels.shape!=(len(first),) or torch.any(labels<0) or torch.any(labels>=first.shape[1])):
        raise ValueError('labels must be valid per-example class indices')
    def fuse(values,weights):
        # Retain source left-associative summation order.
        result=values[0]*weights[0]
        for value,weight in zip(values[1:],weights[1:]):result=result+value*weight
        return result/sum(weights)
    return F.cross_entropy(fuse(main,weights),labels)+.4*F.cross_entropy(fuse(auxiliary,aux_weights),labels)


def infer_labels(probabilities, *, iteration, previous=None):
    """Unweighted sum of eight prediction heads, frozen after iteration zero."""
    if type(iteration) is not int or iteration<0:
        raise ValueError('iteration must be nonnegative')
    values=tuple(probabilities)
    if len(values)!=8:raise ValueError('non-targeted label inference requires eight prediction heads')
    first=values[0]
    if (not isinstance(first,torch.Tensor) or first.ndim!=2 or min(first.shape)<1
            or first.dtype not in (torch.float32,torch.float64) or first.device.type!='cpu'):
        raise ValueError('prediction heads must be nonempty CPU floating matrices')
    for value in values:
        if (not isinstance(value,torch.Tensor) or value.shape!=first.shape or value.dtype!=first.dtype
                or value.device!=first.device or not torch.isfinite(value).all()
                or torch.any(value<0) or torch.any(value>1)):
            raise ValueError('invalid prediction probabilities')
    if iteration==0:
        total=values[0]
        for value in values[1:]:total=total+value
        return total.argmax(dim=1).detach()
    if (not isinstance(previous,torch.Tensor) or previous.dtype!=torch.int64 or previous.device.type!='cpu'
            or previous.shape!=(len(first),) or torch.any(previous<0) or torch.any(previous>=first.shape[1])):
        raise ValueError('subsequent iteration requires valid frozen labels')
    return previous.detach().clone()
