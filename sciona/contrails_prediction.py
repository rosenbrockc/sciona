"""Contrails inference preparation and source final ensemble semantics.

Copyright (c) 2023 Jun Koda; MIT, docs/licenses/Contrails-MIT.txt.
Source 08a15beb36f9cbed4c3990e74c625c1332b61fe8.
"""
import numpy as np
import torch
from torchvision.transforms import Resize

from sciona.contrails_preprocessing import ash_color


VARIANTS = {
    'v43': ((.6465366913319027, 2), (.4531613974630522, 2)),
    'v47': ((.6669411076255138, 5), (.42019871465275055, 5)),
}


def prepare_inference(thermal, *, branch):
    """Accept four selected time steps/three thermal channels in source order."""
    x = np.asarray(thermal)
    if x.shape != (4, 3, 256, 256) or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('Expected finite source-sized thermal input')
    if branch == 'single':
        return Resize(1024, antialias=False)(ash_color(torch.from_numpy(x[3].copy())))
    if branch != 'temporal':
        raise ValueError('Unknown branch')
    # Source temporal inference performs false color in NumPy before torch resize.
    r = (x[:, 2] - x[:, 1] + 4) / 6
    g = (x[:, 1] - x[:, 0] + 4) / 9
    b = (x[:, 1] - 243) / 60
    color = 1 - np.stack((r, g, b), axis=1)
    return Resize(512, antialias=False)(torch.from_numpy(color))


def ensemble_probabilities(temporal, single, *, variant='v47'):
    """Sum per-fold sigmoid outputs in source order without weight normalization.

    Each input has shape (folds, examples, 1, 256, 256). Fold counts must match
    the selected final submission; runtime handles example-order alignment.
    """
    if variant not in VARIANTS:
        raise ValueError('Unknown final submission variant')
    groups = [np.asarray(temporal), np.asarray(single)]
    configs = VARIANTS[variant]
    for values, (_, folds) in zip(groups, configs):
        if values.ndim != 5 or values.shape[0] != folds or values.shape[2:] != (1, 256, 256) or values.shape[1] < 1:
            raise ValueError('Invalid fold probability shape')
        if values.dtype != np.float32 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Expected finite float32 sigmoid probabilities')
    if groups[0].shape[1:] != groups[1].shape[1:]:
        raise ValueError('Branch populations differ')
    result = np.zeros(groups[0].shape[1:], dtype=np.float32)
    for values, (weight, folds) in zip(groups, configs):
        for prediction in values:
            result += (weight / folds) * prediction
    return result


def encode_mask(probability, threshold=.5):
    """Source column-major, one-based run lengths using a strict threshold."""
    x = np.asarray(probability)
    if x.shape not in ((256, 256), (1, 256, 256)) or not np.isfinite(x).all():
        raise ValueError('Expected finite source-sized prediction')
    if not np.isfinite(threshold):
        raise ValueError('Expected finite threshold')
    dots = np.where(x.T.flatten() > threshold)[0]
    runs = []
    previous = -2
    for position in dots:
        if position > previous + 1:
            runs.extend((position + 1, 0))
        runs[-1] += 1
        previous = position
    # NumPy 2 scalar repr changed; explicit integers preserve historical plain RLE.
    return ' '.join(str(int(value)) for value in runs) if runs else '-'
