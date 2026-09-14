"""DFDC training detections and paired crops from caller-decoded RGB arrays.

MIT source: Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. Original detections are reused for altered clips.
Complete decoded arrays exclude source codec failures and partial-decode indexing.
"""
import numpy as np

from sciona.dfdc_detector import build_detector
from sciona.dfdc_faces import detector_image, expanded_crop


def build_preprocessing_detector(*, stage, initialization, seed, state=None):
    if stage not in ('boxes','landmarks'):
        raise ValueError('stage must be boxes or landmarks')
    detector=build_detector(initialization=initialization,seed=seed,state=state)
    detector.thresholds=[.85,.95,.95] if stage=='boxes' else [.65,.75,.75]
    return detector


def _validate_frames(frames):
    if (not isinstance(frames,np.ndarray) or frames.dtype!=np.uint8 or frames.ndim!=4
            or frames.shape[-1]!=3 or min(frames.shape[1:3])<2):
        raise ValueError('decoded frames must be uint8 RGB with spatial dimensions >=2')


def detect_training_boxes(original_rgb, detector):
    """Detect every original frame, in source batches of32, on half-size images."""
    _validate_frames(original_rgb)
    result={}
    for start in range(0,len(original_rgb),32):
        images=[detector_image(frame) for frame in original_rgb[start:start+32]]
        boxes,*_=detector.detect(images,landmarks=False)
        if len(boxes)!=len(images):
            raise ValueError('detector batch result length mismatch')
        for offset,box in enumerate(boxes):
            result[start+offset]=None if box is None else box.tolist()
    return result


def crop_training_clip(decoded_rgb, original_boxes):
    """Use source every-tenth-frame sampling and all detected actor positions."""
    _validate_frames(decoded_rgb)
    result=[]
    for frame_index in range(0,len(decoded_rgb),10):
        boxes=original_boxes.get(frame_index)
        if boxes is None:
            continue
        for actor,box in enumerate(boxes):
            crop=expanded_crop(decoded_rgb[frame_index],box)
            if min(crop.shape[:2])==0:
                raise ValueError('source crop cannot be encoded because it is empty')
            result.append({'frame':frame_index,'actor':actor,'image':crop.copy()})
    return result


def crop_pair(original_rgb, altered_rgb, original_boxes):
    return {'original':crop_training_clip(original_rgb,original_boxes),
            'altered':crop_training_clip(altered_rgb,original_boxes)}
