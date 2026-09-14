"""DFDC source-limited landmark and difference-mask annotations for crop pairs.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. Source generation covers frames0..310 by10,actors0/1.
"""
import numpy as np
from PIL import Image

from sciona.dfdc_difference_mask import difference_mask


def _index(crops):
    result={}
    for crop in crops:
        key=(crop['frame'],crop['actor'])
        if any(type(value) is not int or value<0 for value in key) or key in result:
            raise ValueError('crop frame/actor positions must be unique nonnegative integers')
        image=crop['image']
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
            raise ValueError('crops must be uint8 RGB arrays')
        result[key]=image
    return result


def annotate_pair(original_crops, altered_crops, landmark_detector):
    """Return sparse annotation maps keyed by computational frame/actor position.

    Landmarks belong to original crops and are reused for corresponding altered
    crops. Missing/failed landmark results remain absent as in source file output.
    """
    originals,altered=_index(original_crops),_index(altered_crops)
    landmarks={};differences={}
    for frame in range(0,320,10):
        for actor in range(2):
            key=(frame,actor)
            if key not in originals:
                continue
            image=originals[key]
            try:
                _,_,points=landmark_detector.detect(Image.fromarray(image),landmarks=True)
                if points is not None:
                    selected=np.asarray(points[0])
                    # Current MTCNN's object arrays make np.around fail. Explicit
                    # numeric conversion restores the source rounding operation.
                    if selected.dtype.kind=='O':
                        selected=selected.astype(np.float64)
                    landmarks[key]=np.around(selected).astype(np.int16)
            except Exception:
                pass
            if key in altered:
                mask=difference_mask(image,altered[key])
                if mask is not None:
                    differences[key]=mask
    return {'landmarks':landmarks,'differences':differences}
