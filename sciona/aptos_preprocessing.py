"""Plain resize and publisher normalization for the APTOS reference.

The winner specifies plain resizing and architecture resolutions. Bilinear
Pillow interpolation and legacy publisher normalization are explicit reference
choices; the writeup does not establish its exact preprocessing library.
Training augmentation is a separate, still-required lifecycle operation.
"""
import numpy as np
from PIL import Image
import torch

from sciona.aptos_models import FAMILIES


def prepare_rgb(rgb, family):
    """Convert a private runtime RGB uint8 HWC image into normalized CHW.

    No content is retained or logged. Resizing uses the entire image without
    aspect-preserving crop, retinal crop or local-color subtraction.
    """
    if family not in FAMILIES:
        raise ValueError('A specified APTOS architecture family is required')
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError('Nonempty RGB uint8 HWC image required')
    size = FAMILIES[family][3]
    resized = Image.fromarray(rgb).resize((size, size), resample=Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / np.float32(255.)
    if family.startswith('inception_'):
        mean, std = (.5, .5, .5), (.5, .5, .5)
    else:
        mean, std = (.485, .456, .406), (.229, .224, .225)
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))
