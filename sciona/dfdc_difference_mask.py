"""Source DFDC SSIM mask construction over explicit RGB crop pairs.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. Modern structural_similarity(channel_axis=-1)
replaces the removed compare_ssim(multichannel=True) entrypoint.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def difference_mask(original_rgb, altered_rgb):
    """Return the source uint8 grayscale mask, or None for uncomputable pairs.

    None represents the source's missing mask artifact; training substitutes zeros.
    The uint8 conversion deliberately wraps rather than clips values above 255.
    """
    for image in (original_rgb, altered_rgb):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[-1] != 3):
            raise ValueError('crop pairs must be uint8 RGB arrays')
    if original_rgb.shape != altered_rgb.shape or min(original_rgb.shape[:2]) < 7:
        return None
    # Original file reader produces BGR; retain its SSIM channel order and cast.
    original_bgr = original_rgb[:, :, ::-1].copy()
    altered_bgr = altered_rgb[:, :, ::-1].copy()
    _, similarity = structural_similarity(original_bgr, altered_bgr,
                                          channel_axis=-1, full=True, data_range=255)
    difference = ((1 - similarity) * 255).astype(np.uint8)
    return cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
