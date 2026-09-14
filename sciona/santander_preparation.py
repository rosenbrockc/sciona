"""Source-compared input preparation for the pending neural lifecycle."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sciona.santander_encoding import encode_populations


def prepare_neural_populations(training, labels, test):
    """Encode, map per-column tokens, and standardize combined populations.

Token zero is unknown; observed training categories receive sorted tokens from
one. Scaling fits all training and test rows, including rows excluded only from
uniqueness statistics. This is deliberately transductive, matching recovered
source behavior. No fold-isolated or streaming inference claim is made.
"""
    encoded = encode_populations(training, labels, test)
    train, query = np.asarray(training), np.asarray(test)
    train_tokens = np.empty(train.shape, dtype=np.int64)
    test_tokens = np.zeros(query.shape, dtype=np.int64)
    for column in range(train.shape[1]):
        vocabulary = np.unique(encoded['training_categories'][:, column])
        for token, category in enumerate(vocabulary, start=1):
            train_tokens[encoded['training_categories'][:, column] == category, column] = token
            test_tokens[encoded['test_categories'][:, column] == category, column] = token
    continuous = np.concatenate((
        np.concatenate((train, encoded['training_substituted']), axis=1),
        np.concatenate((query, encoded['test_substituted']), axis=1)), axis=0)
    scaled = StandardScaler().fit_transform(continuous).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError('Nonfinite standardized features')
    groups = train.shape[1]
    return dict(training_categories=train_tokens, test_categories=test_tokens,
        training_raw=scaled[:len(train), :groups].copy(),
        training_substituted=scaled[:len(train), groups:].copy(),
        test_raw=scaled[len(train):, :groups].copy(),
        test_substituted=scaled[len(train):, groups:].copy(),
        labels=np.asarray(labels, dtype=np.int64).copy(),
        real_test_mask=encoded['real_test_mask'].copy())
