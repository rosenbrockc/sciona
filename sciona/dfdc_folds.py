"""DFDC source-part fold assignment without identities or filesystem discovery.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. Positional inputs are supplied at runtime.
"""

import numpy as np


def assign_folds(part_positions, original_positions, *, n_splits=16):
    """Partition source's50 part positions and enforce pair-fold agreement.

    original_positions indexes the original clip for every clip; originals point
    to themselves. This function neither sorts nor shuffles caller record order.
    """
    parts=np.asarray(part_positions)
    originals=np.asarray(original_positions)
    if type(n_splits) is not int or not 1<=n_splits<=50:
        raise ValueError('n_splits must be an integer in [1,50]')
    if (parts.ndim!=1 or parts.size==0 or parts.dtype.kind not in 'iu'
            or np.any(parts<0) or np.any(parts>=50)):
        raise ValueError('part positions must be a nonempty integer vector in [0,49]')
    if (originals.shape!=parts.shape or originals.dtype.kind not in 'iu'
            or np.any(originals<0) or np.any(originals>=len(parts))):
        raise ValueError('original positions must reference input clips')
    if np.any(originals[originals]!=originals):
        raise ValueError('each original reference must point directly to a self-referencing original')
    width=50//n_splits
    folds=np.minimum(parts//width,n_splits-1).astype(np.int64)
    if np.any(folds!=folds[originals]):
        raise ValueError('original and altered clips must belong to the same fold')
    return folds


def eligible_crop_positions(crops):
    """Select source inventory positions in frame-major/actor-minor order."""
    index={}
    for position,crop in enumerate(crops):
        key=(crop['frame'],crop['actor'])
        if any(type(value) is not int or value<0 for value in key) or key in index:
            raise ValueError('crop positions must be unique nonnegative integers')
        index[key]=position
    return np.array([index[(frame,actor)] for frame in range(0,320,10)
                    for actor in range(2) if (frame,actor) in index],dtype=np.int64)
