"""Class-balanced BCE used by the active DFDC B7 training configuration.

Source: MIT 2020 Selim Seferbekov, commit
89c6290490bac96b29193a4061b3db9dd3933e36; docs/licenses/DFDC-MIT.txt.
Labels are already smoothed by preparation. Each class contributes half its
own mean loss; a missing class contributes zero, retaining the divisor of two.
The optional, inactive source OHEM configuration is not selected here.
"""

import torch
from torch.nn import functional as F


def balanced_bce(logits, labels):
    if (not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or logits.ndim != 2 or logits.shape[1] != 1 or logits.shape != labels.shape
            or len(logits) == 0 or not logits.is_floating_point() or not labels.is_floating_point()
            or logits.device != labels.device):
        raise ValueError('logits and labels must be matching nonempty floating (N,1) tensors')
    if (not torch.isfinite(logits).all() or not torch.isfinite(labels).all()
            or torch.any(labels < 0) or torch.any(labels > 1)):
        raise ValueError('logits and labels must be finite; labels must be in [0,1]')
    fake = labels > .5
    real = labels <= .5
    fake_loss = F.binary_cross_entropy_with_logits(logits[fake], labels[fake]) if fake.any() else 0
    real_loss = F.binary_cross_entropy_with_logits(logits[real], labels[real]) if real.any() else 0
    return (fake_loss + real_loss) / 2
