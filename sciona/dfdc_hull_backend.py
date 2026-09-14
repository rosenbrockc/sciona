"""Explicit dlib backend loading for DFDC's68-point polygon augmentation."""

from pathlib import Path

import numpy as np


def load_hull_backend(*, predictor_path):
    """Load a caller-specified native predictor file and bundled frontal detector.

    The caller must supply a model using the source68-point landmark convention.
    No default path, model download or Python pickle loading is used. Native
    predictor deserialization errors are reported without exposing the input path.
    """
    import dlib

    if not isinstance(predictor_path, (str, Path)) or not str(predictor_path):
        raise ValueError('an explicit predictor file is required')
    if not Path(predictor_path).is_file():
        raise ValueError('predictor file is unavailable')
    try:
        predictor = dlib.shape_predictor(str(predictor_path))
        probe = predictor(np.zeros((32,32,3),dtype=np.uint8), dlib.rectangle(0,0,31,31))
    except Exception:
        raise ValueError('native predictor could not be loaded or evaluated') from None
    if probe.num_parts != 68:
        raise ValueError('predictor must contain68 landmark parts')
    return dlib.get_frontal_face_detector(), predictor
