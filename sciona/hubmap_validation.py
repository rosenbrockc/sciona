"""Source-order validation loss accumulation and global Dice statistics."""
import numpy as np
import torch
from torch.nn import functional as F
from sciona.hubmap_losses import _mask_pair, binary_lovasz_hinge


class ValidationAccumulator:
    def __init__(self, *, dice_threshold=.5):
        if not np.isfinite(dice_threshold) or not 0 <= dice_threshold <= 1:
            raise ValueError('Finite probability threshold required')
        self.threshold = dice_threshold
        self.loss = 0
        self.numerator = 0
        self.denominator = 0
        self.examples = 0

    @torch.no_grad()
    def add(self, logits, masks, *, deep_logits=(), classification_logits=None, labels=None):
        _mask_pair(logits, masks)
        if not isinstance(deep_logits, (tuple, list)):
            raise ValueError('Explicit deep head sequence required')
        for deep in deep_logits:
            _mask_pair(deep, masks)
        n = logits.shape[0]
        if (classification_logits is None) != (labels is None):
            raise ValueError('Both classification tensors required')
        if labels is not None:
            for value in [classification_logits, labels]:
                if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Finite float32 CPU classification tensors required')
            if classification_logits.shape!=(n,1) or labels.shape!=(n,) or not ((labels==0)|(labels==1)).all():
                raise ValueError('Aligned binary image labels required')
        terms = [F.binary_cross_entropy_with_logits(logits,masks).item(), binary_lovasz_hinge(logits,masks).item()]
        selected = masks.reshape(n,-1).sum(1)!=0
        for deep in deep_logits:
            if selected.any():
                d=deep[selected];m=masks[selected]
                value=F.binary_cross_entropy_with_logits(d.reshape(len(d),-1),m.reshape(len(m),-1))
                value=value+binary_lovasz_hinge(d,m)
                terms.append(.1*value.item())
            else:
                terms.append(.1*torch.tensor(0).item())
        if labels is not None:
            terms.append(F.binary_cross_entropy_with_logits(classification_logits.squeeze(-1),labels).item())
        for term in terms:
            self.loss += term*n
        probabilities=torch.sigmoid(logits).numpy().reshape(n,1,-1)
        targets=masks.detach().numpy().reshape(n,1,-1)
        numerator=0;denominator=0
        for i in range(n):
            prediction=(probabilities[i,0]>self.threshold).astype(np.float32)
            numerator += 2*np.sum(prediction*targets[i,0])
            denominator += np.sum(prediction)+np.sum(targets[i,0])
        self.numerator += numerator
        self.denominator += denominator
        self.examples += n

    def finish(self, expected_examples):
        if isinstance(expected_examples,bool) or not isinstance(expected_examples,int) or expected_examples<1 or self.examples!=expected_examples:
            raise ValueError('Complete nonempty validation population required')
        if self.denominator==0:
            raise ValueError('Source global Dice is undefined when both totals are zero')
        return dict(loss=self.loss/expected_examples,dice=float(self.numerator/self.denominator),
                    numerator=float(self.numerator),denominator=float(self.denominator),examples=self.examples)


def validate_batches(model, batches, *, expected_examples, dice_threshold=.5):
    """Evaluate the original configuration with all auxiliary heads enabled."""
    if not getattr(model,'deepsupervision',False) or not getattr(model,'clfhead',False):
        raise ValueError('Original validation configuration requires both heads')
    accumulator=ValidationAccumulator(dice_threshold=dice_threshold)
    model.eval()
    with torch.no_grad():
        for images,masks,labels in batches:
            logits,deep,classification=model(images)
            accumulator.add(logits,masks,deep_logits=deep,classification_logits=classification,labels=labels)
    return accumulator.finish(expected_examples)
