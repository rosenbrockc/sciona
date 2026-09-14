"""Selected source detector training Crop with explicit shared RNG.

Copyright (c) 2023 lfz. MIT notice: docs/licenses/DSB2017-MIT.txt.
Source commit 0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2.
Python 2 integer shape/bound quotients are preserved; target geometry stays float.
"""
import numpy as np
import warnings
from scipy.ndimage import zoom


def crop_detection(volume, target, boxes, *, crop_size=128, bound_size=12, stride=4,
                   scale=False, random_crop=False, pad_value=170, rng=None):
    """Source training crop and transformed labels, using an explicit shared RNG."""
    from sciona.dsb_components import _integer
    crop_size = _integer(crop_size, 'crop_size')
    bound_size = _integer(bound_size, 'bound_size', 2)
    stride = _integer(stride, 'stride')
    missing_target = target is None
    if missing_target and (not random_crop or scale):
        raise ValueError('absent target requires an unscaled random crop')
    volume, target, boxes = np.asarray(volume), np.asarray([] if missing_target else target, dtype=float), np.asarray(boxes, dtype=float)
    if (volume.ndim != 4 or min(volume.shape) < 1 or volume.dtype.kind not in 'fiu'
            or (not missing_target and (target.shape != (4,) or not np.all(np.isfinite(target)) or target[3] <= 0))
            or boxes.ndim != 2 or boxes.shape[1] != 4 or not np.all(np.isfinite(boxes))
            or np.any(boxes[:,3] <= 0) or crop_size % stride or not np.isfinite(pad_value)):
        raise ValueError('invalid detector crop inputs')
    config = dict(crop_size=[crop_size]*3, bound_size=bound_size, stride=stride,
                  pad_value=pad_value, rng=np.random.RandomState() if rng is None else rng)
    result = Crop(config)(volume, target, boxes, isScale=scale, isRand=random_crop)
    if result[0].shape != (volume.shape[0], crop_size, crop_size, crop_size):
        raise ValueError('source crop did not produce the requested shape')
    return result

class Crop(object):

    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.rng = config['rng']

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            radiusLim = [8.0, 100.0]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([radiusLim[0] / target[3], scaleLim[0]]), 1]), np.max([np.min([radiusLim[1] / target[3], scaleLim[1]]), 1])]
            scale = self.rng.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] // 2, imgs.shape[i + 1] // 2 + bound_size])
                e = np.min([crop_size[i] // 2, imgs.shape[i + 1] // 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(self.rng.randint(e, s))
            else:
                start.append(int(target[i]) - crop_size[i] // 2 + self.rng.randint(-bound_size // 2, bound_size // 2))
        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] // self.stride), np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] // self.stride), np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] // self.stride), indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:, max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]), max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]), max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]
        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return (crop, target, bboxes, coord)
