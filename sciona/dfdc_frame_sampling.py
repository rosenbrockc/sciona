"""Uniform DFDC frame selection from caller-decoded RGB arrays.

Source: selimsef/dfdc_deepfake_challenge at
89c6290490bac96b29193a4061b3db9dd3933e36 (MIT; docs/licenses/DFDC-MIT.txt).
The original reader stops selecting after a repeated requested index. Preserve
that observable behavior, including first-frame-only selection for short clips.
File decoding and partial decoder failures are outside this array boundary.
"""

import numpy as np


def select_frames(decoded_rgb: np.ndarray, num_frames: int = 32):
    """Return selected RGB frames and source indices, or None for an empty clip."""
    if type(num_frames) is not int or num_frames <= 0:
        raise ValueError('num_frames must be a positive integer')
    if (not isinstance(decoded_rgb, np.ndarray) or decoded_rgb.dtype != np.uint8
            or decoded_rgb.ndim != 4 or decoded_rgb.shape[-1] != 3
            or min(decoded_rgb.shape[1:3]) <= 0):
        raise ValueError('decoded_rgb must be a uint8 array shaped (T,H,W,3)')
    count = len(decoded_rgb)
    if count == 0:
        return None
    requested = np.linspace(0, count - 1, num_frames, endpoint=True, dtype=int)
    # The source keeps its request cursor at the duplicate and never catches up.
    # Later frames cannot be selected; avoid scanning those already-decoded frames.
    repeats = np.flatnonzero(np.diff(requested) == 0)
    selected = requested[:int(repeats[0]) + 1] if len(repeats) else requested
    return decoded_rgb[selected].copy(), selected.tolist()
