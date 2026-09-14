"""Contrails temporal panels from Jun Koda's MIT-licensed winning source.

Source: 08a15beb36f9cbed4c3990e74c625c1332b61fe8, src/submit/vit4/model.py.
Copyright (c) 2023 Jun Koda; see docs/licenses/Contrails-MIT.txt.
"""
import torch


def _check(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 5 or x.shape[1:3] != (4, 3):
        raise ValueError('Expected B,4,3,H,W tensor')
    if min(x.shape) < 1 or x.shape[-2] != x.shape[-1]:
        raise ValueError('Expected nonempty square frames')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating point frames')


def create_four_panels(x: torch.Tensor) -> torch.Tensor:
    """Place frames 3,0 / 1,2 in spatial quadrants; preserve source float32 cast."""
    _check(x)
    batch, _, _, height, width = x.shape
    result = torch.zeros((batch, 3, 2 * height, 2 * width), dtype=torch.float32, device=x.device)
    result[:, :, :height, :width] = x[:, 3]
    result[:, :, :height, width:] = x[:, 0]
    result[:, :, height:, :width] = x[:, 1]
    result[:, :, height:, width:] = x[:, 2]
    return result


def create_four_panels_tta(x: torch.Tensor, rotation: int) -> torch.Tensor:
    """Rotate each frame, then form unflipped/flipped panel batches.

    The quadrant assignment stays fixed: rotating the completed panel would
    incorrectly move temporal frames into different quadrants.
    """
    _check(x)
    if type(rotation) is not int or rotation not in range(4):
        raise ValueError('Rotation must be an integer from 0 to 3')
    rotated = torch.rot90(x, rotation, dims=[3, 4])
    return torch.cat((create_four_panels(rotated),
                      create_four_panels(rotated.flip(dims=[4]))), dim=0)
