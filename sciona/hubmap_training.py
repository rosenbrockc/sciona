"""Full-network CPU training steps for the pinned HuBMAP configuration."""
import torch
from sciona.hubmap_losses import training_loss


def train_batch(model, optimizer, images, masks, labels, *, return_logits=False):
    """Execute configured heads, exact composite loss, backward and Adam step.

    Inputs are already normalized/augmented runtime tensors. Optimizer state is
    retained by the caller. CPU float32 is explicit; historical CUDA mixed
    precision and complete epoch/validation orchestration are separate work.
    """
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError('Source Adam optimizer required')
    if not getattr(model, 'deepsupervision', False) or not getattr(model, 'clfhead', False):
        raise ValueError('Original training configuration requires both heads')
    for value in [images, masks, labels]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or not torch.isfinite(value).all():
            raise ValueError('Finite CPU float32 training tensors required')
    if images.ndim != 4 or images.shape[0] < 2 or images.shape[1] != 3 or any(n < 32 or n % 32 for n in images.shape[2:]):
        raise ValueError('N3HW images, batch at least two and spatial multiples of32 required')
    n, _, h, w = images.shape
    if masks.shape != (n,1,h,w) or labels.shape != (n,) or not ((masks == 0) | (masks == 1)).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Aligned binary masks and classification labels required')
    if masks.requires_grad or labels.requires_grad:
        raise ValueError('Targets must be detached')
    model.train()
    optimizer.zero_grad()
    logits, deep, classification = model(images)
    loss = training_loss(logits, masks, deep_logits=deep,
                         classification_logits=classification, classification_targets=labels)
    loss.backward()
    optimizer.step()
    if return_logits:
        return loss.detach(), logits.detach()
    return loss.detach()
