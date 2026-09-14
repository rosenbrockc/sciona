"""Pinned source augmentation with explicit RNG and copy-preserving wrappers.

Copyright (c) 2023 lfz. MIT notice: docs/licenses/DSB2017-MIT.txt.
Source commit 0ac3eb9f383bf0127c587e0502c59ac84d9ba6a2.
Source conventions retained: detection flips transform target as size-coordinate;
classifier rotation rotates the image but not its coordinate grid. Source default
classifier rotation is disabled. Neither is silently presented as a correction.
"""
import numpy as np
from scipy.ndimage import rotate


def _arrays(sample, coord):
    sample, coord = np.asarray(sample), np.asarray(coord)
    if (sample.ndim != 4 or coord.ndim != 4 or coord.shape[0] != 3
            or min(sample.shape) < 1 or min(coord.shape) < 1
            or sample.dtype.kind not in 'fiu' or coord.dtype.kind != 'f'
            or not np.all(np.isfinite(sample)) or not np.all(np.isfinite(coord))):
        raise ValueError('expected finite image and three-channel coordinate tensors')
    return sample.copy(), coord.copy()


def augment_detection(sample, target, boxes, coord, *, flip=True, rotate=True, swap=True, rng=None):
    sample, coord = _arrays(sample, coord)
    target, boxes = np.asarray(target, dtype=float).copy(), np.asarray(boxes, dtype=float).copy()
    if target.shape != (4,) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('invalid target or boxes')
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(boxes)) or target[3] <= 0 or np.any(boxes[:,3] <= 0):
        raise ValueError('augmentation requires a finite positive-size target and boxes')
    return _detector_augment(sample, target, boxes, coord, ifflip=flip, ifrotate=rotate,
                             ifswap=swap, rng=np.random.RandomState() if rng is None else rng)


def augment_classifier(sample, coord, *, flip=True, rotate=False, swap=False, filling_value=160, rng=None):
    sample, coord = _arrays(sample, coord)
    if not np.isfinite(filling_value):
        raise ValueError('filling value must be finite')
    return _classifier_augment(sample, coord, ifflip=flip, ifrotate=rotate, ifswap=swap,
                               filling_value=filling_value, rng=np.random.RandomState() if rng is None else rng)

def _detector_augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True, *, rng):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = (rng.rand() - 0.5) * 20
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)], [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = rng.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]
    if ifflip:
        flipid = np.array([1, rng.randint(2), rng.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return (sample, target, bboxes, coord)

def _classifier_augment(sample, coord, ifflip=True, ifrotate=True, ifswap=True, filling_value=0, *, rng):
    if ifrotate:
        validrot = False
        counter = 0
        angle1 = rng.rand() * 180
        size = np.array(sample.shape[2:4]).astype('float')
        rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)], [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
        sample = rotate(sample, angle1, axes=(2, 3), reshape=False, cval=filling_value)
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = rng.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
    if ifflip:
        flipid = np.array([rng.randint(2), rng.randint(2), rng.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
    return (sample, coord)
