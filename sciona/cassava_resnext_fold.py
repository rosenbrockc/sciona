"""Source 15-epoch ResNeXt fold lifecycle, with explicit CPU diagnostics."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from sciona.cassava_resnext import Classifier, optimizer_for, train_epoch, validate
from sciona.cassava_resnext_images import decode_rgb, transformations, transform_rgb


class EncodedPopulation(Dataset):
    def __init__(self, images, labels, *, training):
        if (not isinstance(images, (tuple, list)) or not images
                or not all(isinstance(x, bytes) and x for x in images)
                or not isinstance(labels, (tuple, list)) or len(images) != len(labels)
                or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 5 for x in labels)):
            raise ValueError('Aligned nonempty encoded images and five-class labels required')
        self.images, self.labels = tuple(images), tuple(labels)
        self.transform = transformations(training=training)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = transform_rgb(decode_rgb(self.images[index]), self.transform)
        return image, torch.tensor(self.labels[index], dtype=torch.int64)


def scheduler_for(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=5, eps=1e-6)


def train_fold(train_images, train_labels, val_images, val_labels, *, pretrained,
               batch_size, workers, output_directory, weights_path=None, expected_sha256=None):
    """Run all 15 epochs and load the first strict best-accuracy checkpoint.

Source best score starts at zero. An all-zero validation run fails explicitly
instead of inventing a checkpoint. Original batch size is 16/workers 4;
smaller settings and pretrained=False are diagnostic choices only.
Caller must establish disjoint fold populations and seed runtime RNGs.
"""
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(workers, bool) or not isinstance(workers, int) or workers < 0
            or not isinstance(pretrained, bool)):
        raise ValueError('Explicit valid batch size, worker count and pretrained flag required')
    training = EncodedPopulation(train_images, train_labels, training=True)
    validation = EncodedPopulation(val_images, val_labels, training=False)
    if len(training) < batch_size:
        raise ValueError('Training population must produce a full batch')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / 'best.pt'
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)
    model = Classifier(pretrained=pretrained, weights_path=weights_path,
                       expected_sha256=expected_sha256)
    optimizer = optimizer_for(model)
    scheduler = scheduler_for(optimizer)
    history, best_score = [], 0.
    for epoch in range(15):
        lr = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, train_loader, optimizer)
        val_loss, probabilities = validate(model, val_loader)
        scheduler.step(val_loss)
        accuracy = float(np.mean(probabilities.argmax(1) == np.asarray(val_labels)))
        history.append(dict(epoch=epoch, loss=loss, val_loss=val_loss, accuracy=accuracy, lr=lr))
        if accuracy > best_score:
            best_score = accuracy
            torch.save({'model': model.state_dict(), 'probabilities': torch.from_numpy(probabilities), 'epoch': epoch}, checkpoint)
    if not checkpoint.exists():
        raise RuntimeError('No validation accuracy exceeded source initial best score of zero')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'])
    return model, history, checkpoint
