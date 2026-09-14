"""Source RGB batching and SciPy-reference PNG quantization, without file paths.

SciPy Authors BSD-3 reference: docs/licenses/Adversarial-SciPy-reference-BSD.txt.
imsave byte scaling stretches per-image min/max; it does not preserve a raw pixel
perturbation bound. Historical SciPy/Pillow versions are unspecified.
"""
from io import BytesIO
import numpy as np
from PIL import Image


def batches(rgb, *, targets=None):
    """Yield source default batches of ten, float64 normalize then float32 feed."""
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=4
            or rgb.shape[1:]!=(299,299,3) or len(rgb)<1):
        raise ValueError('nonempty299RGB uint8 batch required')
    if targets is not None and (not isinstance(targets,np.ndarray) or targets.dtype!=np.int64
            or targets.shape!=(len(rgb),) or np.any(targets<0) or np.any(targets>=1001)):
        raise ValueError('matching int64 source target classes required')
    for start in range(0,len(rgb),10):
        count=min(10,len(rgb)-start)
        values=np.zeros((10,299,299,3),dtype=np.float64)
        values[:count]=rgb[start:start+count].astype(np.float64)/255.*2.-1.
        labels=None
        if targets is not None:
            labels=np.zeros(10,dtype=np.int64);labels[:count]=targets[start:start+count]
        yield count,values.astype(np.float32),labels


def source_rgb(normalized):
    if (not isinstance(normalized,np.ndarray) or normalized.dtype!=np.float32
            or normalized.shape!=(299,299,3) or not np.isfinite(normalized).all()
            or np.any(normalized < -1) or np.any(normalized > 1)):
        raise ValueError('normalized299RGBfloat32 image required')
    values=(normalized+1.)*.5
    minimum=values.min();span=values.max()-minimum
    if span==0:span=1
    scaled=(values-minimum)*(255./span)
    return (scaled.clip(0,255)+.5).astype(np.uint8)


def encode_real_entries(normalized, count):
    if (not isinstance(normalized,np.ndarray) or normalized.shape!=(10,299,299,3)
            or type(count) is not int or not 1<=count<=10):
        raise ValueError('source ten-entry batch and real entry count required')
    result=[]
    for x in normalized[:count]:
        stream=BytesIO();Image.fromarray(source_rgb(x)).save(stream,format='PNG')
        result.append(stream.getvalue())
    return result
