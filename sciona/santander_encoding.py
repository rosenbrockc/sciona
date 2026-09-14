"""Independent implementation of the recovered supervised uniqueness encoding.

The supplied labeled population is the reference. Training class-presence codes
exclude each row itself, not an entire held-out fold. Callers must not report
cross-validation on globally encoded labeled rows as label-isolated validation.
"""
from collections import Counter
import math
import numpy as np
from sciona.santander_features import real_test_mask


def encode_populations(training, labels, test):
    """Return training/test categories and substituted values plus test mask.

Codes 0..3 record other labeled class presence: twice positive presence plus
negative presence. Code 4 overrides values occurring at most once in training
plus retained test rows. Substitution uses the mean of the labeled population.
Test predictions retain all test rows, including rows excluded from statistics.
Inputs are finite floating matrices; binary labels align with training rows.
"""
    train, target, query = map(np.asarray, (training, labels, test))
    for matrix in (train, query):
        if matrix.ndim != 2 or not all(matrix.shape) or matrix.dtype.kind != 'f' or not np.isfinite(matrix).all():
            raise ValueError('Expected nonempty finite floating matrices')
    if train.shape[1] != query.shape[1]:
        raise ValueError('Population feature widths differ')
    if target.shape != (len(train),) or target.dtype.kind not in 'biu' or not np.isin(target, [0, 1]).all():
        raise ValueError('Expected aligned binary labels')
    retained = real_test_mask(query)
    categories = [np.empty(matrix.shape, dtype=np.int64) for matrix in (train, query)]
    substituted = [np.empty(matrix.shape, dtype=np.float64) for matrix in (train, query)]
    for column in range(train.shape[1]):
        positive = Counter(train[target == 1, column])
        negative = Counter(train[target == 0, column])
        reference = Counter(train[:, column])
        reference.update(query[retained, column])
        try:
            mean = math.fsum(float(v) / len(train) for v in train[:, column])
        except OverflowError as error:
            raise ValueError('Nonfinite feature mean') from error
        if not math.isfinite(mean):
            raise ValueError('Nonfinite feature mean')
        for population, matrix in enumerate((train, query)):
            for row, value in enumerate(matrix[:, column]):
                pos, neg = positive[value], negative[value]
                if population == 0:
                    pos -= int(target[row] == 1)
                    neg -= int(target[row] == 0)
                repeated = reference[value] > 1
                categories[population][row, column] = (2 * int(pos > 0) + int(neg > 0)) if repeated else 4
                substituted[population][row, column] = value if repeated else mean
    return dict(training_categories=categories[0], test_categories=categories[1],
        training_substituted=substituted[0], test_substituted=substituted[1], real_test_mask=retained)
