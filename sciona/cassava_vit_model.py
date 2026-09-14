"""ViT-B/16 backbone and source validation semantics in the CPU reference."""
import numpy as np
import torch
from torch import nn

from sciona.cassava_loss import vit_loss


class Classifier(nn.Module):
    def __init__(self, *, pretrained, weights_path=None, expected_sha256=None):
        super().__init__()
        from sciona.cassava_pretrained import build_backbone
        self.model = build_backbone('vit', pretrained=pretrained,
                                    weights_path=weights_path, expected_sha256=expected_sha256)
        self.model.head = nn.Linear(self.model.head.in_features, 5)

    def forward(self, images):
        return self.model(images)


def validate(model, batches):
    """Return last-batch loss and local population accuracy, as the source does.

The winning helper never accumulates validation loss. Its distributed accuracy
is an unweighted mean of replica accuracies; this function covers one CPU
replica only. It must not be interpreted as global weighted TPU accuracy.
"""
    model.eval()
    labels_all, predictions, last = [], [], None
    with torch.no_grad():
        for images, labels in batches:
            if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, 384, 384)
                    or images.dtype != torch.float32 or labels.dtype != torch.int64
                    or labels.shape != (images.shape[0],) or (labels < 0).any() or (labels >= 5).any()
                    or not torch.isfinite(images).all()):
                raise ValueError('Finite 384-pixel float32 batches and integer five-class labels required')
            logits = model(images)
            targets = nn.functional.one_hot(labels, 5).to(logits.dtype)
            last = vit_loss(logits, targets).mean().item()
            labels_all.append(labels.cpu().numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
    if last is None:
        raise ValueError('Nonempty validation required')
    return last, float(np.mean(np.concatenate(labels_all) == np.concatenate(predictions)))
