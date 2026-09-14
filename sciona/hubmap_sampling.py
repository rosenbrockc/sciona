"""Source-order epoch sampling from explicit synthetic/runtime population arrays."""
import numpy as np


def balanced_epoch_indices(present, bins, *, maximum_bin, rng):
    """Reproduce the two-stage replacement sampling in pinned HuBMAP run.py.

    Background and foreground first sample the smaller population count. Then
    each observed foreground bin samples the floored mean bin count, retaining
    first-appearance bin order. The result need not be exactly class balanced.
    This function returns original positional indices; caller controls shuffling.
    """
    present = np.asarray(present)
    bins = np.asarray(bins)
    if present.ndim != 1 or present.dtype != np.bool_ or not len(present) or bins.shape != present.shape:
        raise ValueError('Aligned nonempty Boolean presence and bin arrays required')
    if not np.issubdtype(bins.dtype, np.integer) or np.any(bins < 0):
        raise ValueError('Nonnegative integer bins required')
    if isinstance(maximum_bin, bool) or not isinstance(maximum_bin, (int, np.integer)) or maximum_bin < 0:
        raise ValueError('Nonnegative maximum bin required')
    if not isinstance(rng, np.random.RandomState):
        raise ValueError('Explicit NumPy RandomState required')
    negative = np.flatnonzero(~present)
    positive = np.flatnonzero(present)
    if not len(negative) or not len(positive):
        raise ValueError('Both foreground and background populations required')
    capped = np.minimum(bins, maximum_bin)
    count = min(len(negative), len(positive))
    background = negative[rng.choice(len(negative), size=count, replace=True)]
    foreground = positive[rng.choice(len(positive), size=count, replace=True)]
    ordered_bins = list(dict.fromkeys(capped[foreground].tolist()))
    per_bin = int(count / len(ordered_bins))
    groups = []
    for selected_bin in ordered_bins:
        group = foreground[capped[foreground] == selected_bin]
        groups.append(group[rng.choice(len(group), size=per_bin, replace=True)])
    return np.concatenate([*groups, background])
