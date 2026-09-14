"""Fivefold pretrained torch-family training and identity-aligned inference."""
import gc
from pathlib import Path

import numpy as np
import torch

from sciona.cassava_fold_contract import members, average_folds, _keys
from sciona.cassava_fold_dispatch import train_planned_fold
from sciona.cassava_resnext_images import decode_rgb, transformations as resnext_transforms
from sciona.cassava_vit_images import transformations as vit_transforms


def predict_images(family, model, encoded_images, *, batch_size):
    if family not in ('vit', 'resnext'):
        raise ValueError('A supported torch family is required')
    if (type(batch_size) is not int or batch_size < 1
            or not isinstance(encoded_images, (list, tuple)) or not encoded_images
            or not all(isinstance(image, bytes) and image for image in encoded_images)):
        raise ValueError('Nonempty encoded images and positive batch size required')
    transform = vit_transforms('inference') if family == 'vit' else resnext_transforms(training=False)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(encoded_images), batch_size):
            images = torch.stack([transform(image=decode_rgb(image))['image']
                                  for image in encoded_images[start:start + batch_size]])
            predictions.append(model(images).softmax(-1).cpu().numpy())
    return np.concatenate(predictions)


def train_five_folds(family, plan, training_images, prediction_keys, prediction_images,
                     *, weights_path, expected_sha256, batch_size, workers, output_directory):
    """Run all source epochs per fold; explicit reference weights are required.

The corrected population contract excludes each held-out fold from training.
CPU settings remain explicit; this does not claim source TPU/budget parity.
All private checkpoints are retained under the caller's output directory.
"""
    if family not in ('vit', 'resnext'):
        raise ValueError('A supported torch family is required')
    members(plan, 0)
    keys = _keys(prediction_keys)
    for images, length in ((training_images, len(plan.keys)), (prediction_images, len(keys))):
        if (not isinstance(images, (list, tuple)) or len(images) != length
                or not all(isinstance(image, bytes) and image for image in images)):
            raise ValueError('Encoded populations must match their identity contracts')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    folds, histories, checkpoints = {}, {}, {}
    for fold in range(5):
        result = train_planned_fold(family, plan, training_images, fold,
            pretrained=True, weights_path=weights_path, expected_sha256=expected_sha256,
            batch_size=batch_size, workers=workers, output_directory=directory / f'fold-{fold}')
        model, history, checkpoint = result[:3]
        probabilities = predict_images(family, model, prediction_images, batch_size=batch_size)
        folds[fold] = (keys, probabilities)
        histories[fold], checkpoints[fold] = history, checkpoint
        del model, result
        gc.collect()
    return average_folds(keys, folds), folds, histories, checkpoints
