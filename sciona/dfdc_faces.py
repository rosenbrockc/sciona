"""DFDC inference face preparation with an explicitly supplied detector.

Adapted from MIT-licensed selimsef/dfdc_deepfake_challenge, commit
89c6290490bac96b29193a4061b3db9dd3933e36; see docs/licenses/DFDC-MIT.txt.
This is the repository inference variant: half-size PIL input, expanded boxes,
no landmark alignment. Detector implementation and weights are separate inputs.
"""

import numpy as np
from numbers import Real
from PIL import Image


def detector_image(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[-1] != 3
            or min(frame.shape[:2]) < 2):
        raise ValueError('frame must be uint8 RGB with both spatial dimensions >= 2')
    image = Image.fromarray(frame)
    return image.resize(size=[s // 2 for s in image.size])


def expanded_crop(frame, box):
    """Retain source integer truncation, margin and NumPy slicing conventions."""
    coordinates = np.asarray(box)
    # MTCNN.detect returns object arrays even when all coordinates are numeric.
    if coordinates.dtype.kind == 'O' and coordinates.shape == (4,):
        if all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in coordinates):
            coordinates = coordinates.astype(np.float64)
    if (coordinates.shape != (4,) or coordinates.dtype.kind not in 'fiu'
            or not np.isfinite(coordinates).all()):
        raise ValueError('box must contain four finite numeric coordinates')
    xmin, ymin, xmax, ymax = [int(b * 2) for b in coordinates]
    width, height = xmax - xmin, ymax - ymin
    margin_h, margin_w = height // 3, width // 3
    return frame[max(ymin - margin_h, 0):ymax + margin_h,
                 max(xmin - margin_w, 0):xmax + margin_w]


def extract_faces(frames, detector):
    """Return ordered per-frame faces/scores; frames without boxes are omitted.

The detector must implement detect(PIL_image, landmarks=False). No download or
implicit detector construction occurs. Source bookkeeping dimensions are omitted:
they are overwritten with the last box dimensions and unused by classification.
"""
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8
            or frames.ndim != 4 or frames.shape[-1] != 3
            or min(frames.shape[1:3]) < 2):
        raise ValueError('frames must be a uint8 array shaped (T,H,W,3)')
    results = []
    for index, frame in enumerate(frames):
        boxes, probabilities = detector.detect(detector_image(frame), landmarks=False)
        if boxes is None:
            continue
        faces, scores = [], []
        # Retain source zip behavior and skip None boxes; do not add a threshold.
        for box, score in zip(boxes, probabilities):
            if box is not None:
                faces.append(expanded_crop(frame, box))
                scores.append(score)
        results.append({'frame_index': index, 'faces': faces, 'scores': scores})
    return results
