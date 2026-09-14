import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_distribution_features import andriy_distribution_features


def test_uncorrected_moments_and_no_mutation():
    x = np.tile([-1., 1.], 30)[None, :]
    original = x.copy()
    actual = andriy_distribution_features(x)
    np.testing.assert_array_equal(actual[0, :2], [1., 0.])
    np.testing.assert_array_equal(x, original)


def test_final_sample_excluded_from_svd_but_included_in_moments():
    x = np.random.default_rng(381).normal(size=(1, 100))
    changed = x.copy()
    changed[0, -1] = 1000
    before, after = andriy_distribution_features(x), andriy_distribution_features(changed)
    np.testing.assert_array_equal(before[:, 2:], after[:, 2:])
    assert np.all(before[:, :2] != after[:, :2])


def test_negative_scale_moment_and_svd_invariance():
    x = np.random.default_rng(382).normal(size=(2, 100))
    before = andriy_distribution_features(x)
    after = andriy_distribution_features(-3*x)
    np.testing.assert_allclose(after, before * [1, -1, 1, 1], rtol=1e-12, atol=1e-13)


def test_zero_input_preserves_undefined_results():
    assert np.isnan(andriy_distribution_features(np.zeros((1, 100)))).all()


def test_single_embedding_column_has_zero_entropy_and_fisher():
    actual = andriy_distribution_features(np.arange(21.)[None, :])
    np.testing.assert_array_equal(actual[0, 2:], [0., 0.])


@pytest.mark.parametrize('x', [np.ones(30), np.ones((1, 20)), np.ones((0, 30)), np.full((1, 30), np.nan), np.full((1, 30), np.inf), np.ones((1, 30), dtype=complex)])
def test_invalid_windows(x):
    with pytest.raises(ValueError):
        andriy_distribution_features(x)
