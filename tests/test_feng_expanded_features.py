import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feng_expanded_features import feng_expanded_features


def test_feature_order_time_correlation_and_nonmutation():
    x = np.random.default_rng(250).normal(size=(20000, 16)).astype(np.float32)
    before = x.copy()
    actual = feng_expanded_features(x)
    assert actual.shape == (1, 384)
    for channel in range(16):
        assert actual[0, channel*7+6] == np.std(x[:, channel])
    a = x.T.astype(float)
    normalized = (a-a.mean(axis=0))/a.std(axis=0)
    centered = normalized-normalized.mean(axis=1, keepdims=True)
    gram = centered@centered.T
    correlation = gram/np.sqrt(np.outer(np.diag(gram), np.diag(gram)))
    np.testing.assert_allclose(actual[0, 112:232], correlation[np.triu_indices(16, 1)], atol=1e-14)
    assert np.all(np.diff(actual[0, 232:248]) >= 0)
    assert np.all(np.diff(actual[0, 368:384]) >= 0)
    np.testing.assert_array_equal(x, before)


def test_independent_windows_preserve_order():
    rng = np.random.default_rng(251)
    a, b = rng.normal(size=(20000, 16)), rng.normal(size=(20000, 16))
    np.testing.assert_array_equal(feng_expanded_features(np.r_[a, b]), np.r_[feng_expanded_features(a), feng_expanded_features(b)])


@pytest.mark.parametrize('x', [np.zeros((20000, 15)), np.zeros((19999, 16)), np.zeros((20001, 16)),
                             np.full((20000, 16), np.nan), np.ones((20000, 16), dtype=complex)])
def test_invalid_contract(x):
    with pytest.raises(ValueError):
        feng_expanded_features(x)
