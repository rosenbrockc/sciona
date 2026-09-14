"""DFDC epoch population selection over caller-supplied computational arrays.

MIT source: Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. No source validation-membership lists are used.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PopulationSelection:
    indices: np.ndarray
    numpy_state_after_shuffle: tuple


def select_population(labels, folds, frame_numbers, *, mode, epoch, seed,
                      fold=0, rebalance=True, reduce_val=True):
    """Retain source class downsampling, validation thinning and epoch shuffle.

    Caller NumPy RNG is unchanged. The returned RNG state lets the outer training
    lifecycle preserve source continuation into augmentation within its own scope.
    All arrays refer to caller-owned records by position, without identities.
    """
    labels, folds, frame_numbers = map(np.asarray, (labels, folds, frame_numbers))
    if (labels.ndim != 1 or len(labels) == 0 or labels.dtype.kind not in 'fiub'
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError('labels must be a nonempty binary vector')
    for values in (folds, frame_numbers):
        if (values.shape != labels.shape or values.dtype.kind not in 'iu'
                or np.any(values < 0)):
            raise ValueError('folds and frame numbers must be matching nonnegative integer vectors')
    if mode not in ('train', 'val') or type(fold) is not int or fold < 0:
        raise ValueError('mode must be train/val and fold a nonnegative integer')
    if type(epoch) is not int or epoch < 0 or type(seed) is not int or seed < 0:
        raise ValueError('epoch and seed must be nonnegative integers')
    effective_seed = (epoch + 1) * seed
    if effective_seed >= 2**32:
        raise ValueError('effective epoch seed must be below 2**32')
    if type(rebalance) is not bool or type(reduce_val) is not bool:
        raise ValueError('population flags must be booleans')
    indices = np.flatnonzero(folds != fold if mode == 'train' else folds == fold)
    if rebalance:
        real = indices[labels[indices] == 0]
        fake = indices[labels[indices] == 1]
        if mode == 'train':
            if len(fake) < len(real):
                raise ValueError('source downsampling requires at least as many fake as real examples')
            # pandas.sample(n=..., random_state=int) uses a separate RandomState.
            choice = np.random.RandomState(effective_seed).choice(len(fake), size=len(real), replace=False)
            fake = fake[choice]
        indices = np.concatenate([real, fake])
    if mode == 'val' and reduce_val:
        indices = indices[frame_numbers[indices] % 20 == 0]
    rng = np.random.RandomState(effective_seed)
    rng.shuffle(indices)
    return PopulationSelection(indices, rng.get_state())
