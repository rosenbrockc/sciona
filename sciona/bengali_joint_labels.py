"""Joint-label coding from the pinned Bengali CycleGAN software reference.

The reference explicitly uses168 roots,11 vowels and8 consonant states.
No mapping from class indices to real graphemes or records is bundled.
"""
import numpy as np


def encode_components(components):
    values = np.asarray(components)
    if (values.ndim != 2 or not len(values) or values.shape[1] != 3 or values.dtype.kind not in 'iu'
            or (values < 0).any() or (values >= np.array([168,11,8])).any()):
        raise ValueError('nonempty integer N3 components within168/11/8 class bounds required')
    values = values.astype(np.int64)
    return (values[:,0] * 11 + values[:,1]) * 8 + values[:,2]


def decode_joint_classes(classes):
    values = np.asarray(classes)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
            or (values < 0).any() or (values >= 14784).any()):
        raise ValueError('nonempty integer joint classes in0..14783 required')
    values = values.astype(np.int64)
    return np.stack((values // 88, (values // 8) % 11, values % 8), axis=1)
