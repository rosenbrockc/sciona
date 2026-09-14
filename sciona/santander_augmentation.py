"""Independent paired-group augmentation for the pending Santander pipeline."""
import numpy as np


def shuffle_feature_groups(categories, raw, substituted, targets, *, rng, training):
    """Shuffle feature triplets within class, independently for each feature.

Returns copied categories, raw values, substituted values and targets. Training
output groups positive rows before negative rows. Evaluation preserves input
order and consumes no random numbers. The caller owns the Generator so training
can advance its state reproducibly across batches. No rows are added here.
"""
    cat, original, replacement, labels = map(np.asarray,
        (categories, raw, substituted, targets))
    if cat.ndim != 2 or not all(cat.shape):
        raise ValueError('Expected nonempty feature matrices')
    if original.shape != cat.shape or replacement.shape != cat.shape:
        raise ValueError('Feature group shapes differ')
    if cat.dtype.kind not in 'iu' or np.any(cat > 5) or np.any(cat < 0):
        raise ValueError('Categories must be integers in the six-token vocabulary')
    for values in (original, replacement):
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
            raise ValueError('Continuous features must be finite real numbers')
    if labels.shape != (cat.shape[0],) or labels.dtype.kind not in 'biu' or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected aligned binary labels')
    if type(training) is not bool or not isinstance(rng, np.random.Generator):
        raise ValueError('Expected boolean training control and explicit Generator')
    if not training:
        return tuple(value.copy() for value in (cat, original, replacement, labels))
    order = np.concatenate([np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)])
    result = [value[order].copy() for value in (cat, original, replacement)]
    output_labels = labels[order].copy()
    for label in (1, 0):
        destination = np.flatnonzero(output_labels == label)
        source = np.flatnonzero(labels == label)
        for column in range(cat.shape[1]):
            selected = rng.permutation(source)
            for output, values in zip(result, (cat, original, replacement)):
                output[destination, column] = values[selected, column]
    return (*result, output_labels)
