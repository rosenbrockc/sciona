"""Ten-epoch CPU ViT fold with an explicitly corrected validation population."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler

from sciona.cassava_resnext_fold import EncodedPopulation
from sciona.cassava_vit_images import transformations
from sciona.cassava_vit_model import Classifier, validate
from sciona.cassava_vit_ordered_loss import loss


def train_stream(model, loader, optimizer, scheduler):
    """Source two-batch gradient sum and final partial accumulation flush."""
    if not len(loader):
        raise ValueError('Nonempty training loader required')
    model.train()
    updates = 0
    for index, (images, labels) in enumerate(loader):
        logits = model(images.float())
        targets = torch.nn.functional.one_hot(labels.long(), 5).to(logits.dtype)
        loss(logits, targets).mean().backward()
        if (index + 1) % 2 == 0 or index + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            updates += 1
    scheduler.step()
    return updates


def train_fold(train_images, train_labels, val_images, val_labels, *, pretrained,
               batch_size, workers, output_directory, weights_path=None, expected_sha256=None):
    """Require separately selected training and validation populations.

Correction: pinned notebook concatenates validation into training. This runner
does not; caller must establish disjoint sample membership. Source sampler
epoch stays zero, loaders drop incomplete batches, and accuracy ties replace
the checkpoint. CPU replica one; original batch ten and TPU execution remain
separate qualification requirements. False pretrained is diagnostic only.
"""
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(workers, bool) or not isinstance(workers, int) or workers < 0):
        raise ValueError('Valid explicit batch size and worker count required')
    training = EncodedPopulation(train_images, train_labels, training=False)
    validation = EncodedPopulation(val_images, val_labels, training=False)
    training.transform = transformations('train')
    validation.transform = transformations('validation')
    if min(len(training), len(validation)) < batch_size:
        raise ValueError('Both populations must produce a full source batch')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / 'best.pt'
    model = Classifier(pretrained=pretrained, weights_path=weights_path,
                       expected_sha256=expected_sha256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 / 7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
    history, best, updates = [], 0., 0
    for epoch in range(10):
        train_loader = DataLoader(training, batch_size=batch_size, num_workers=workers, drop_last=True,
            sampler=DistributedSampler(training, num_replicas=1, rank=0, shuffle=True))
        val_loader = DataLoader(validation, batch_size=batch_size, num_workers=workers, drop_last=True,
            sampler=DistributedSampler(validation, num_replicas=1, rank=0, shuffle=False))
        rate = optimizer.param_groups[0]['lr']
        updates += train_stream(model, train_loader, optimizer, scheduler)
        validation_loss, accuracy = validate(model, val_loader)
        history.append(dict(epoch=epoch, lr=rate, last_validation_batch_loss=validation_loss, accuracy=accuracy))
        if accuracy >= best:
            best = accuracy
            torch.save({'model': model.state_dict(), 'epoch': epoch}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    return model, history, checkpoint, updates
