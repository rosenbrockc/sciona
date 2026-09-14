import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.basic_statistics import channel_basic_statistics


def test_ordered_statistics_match_scalar_moments_and_interpolated_quantiles():
    a = np.array([[[1., 2., 4., 7., 9.], [-4., -2., 0., 1., 8.]]])
    before = a.copy()
    actual = channel_basic_statistics(a)
    assert actual.shape == (1, 2, 6)
    for ci, x in enumerate(a[0]):
        mean = sum(x) / len(x)
        variance = sum((v - mean) ** 2 for v in x) / len(x)
        third = sum((v - mean) ** 3 for v in x) / len(x)
        fourth = sum((v - mean) ** 4 for v in x) / len(x)
        ordered = sorted(x)
        q10 = .6 * ordered[0] + .4 * ordered[1]
        q90 = .4 * ordered[3] + .6 * ordered[4]
        np.testing.assert_allclose(actual[0, ci], [mean, variance ** .5, fourth / variance ** 2 - 3,
                                                  third / variance ** 1.5, q90, q10], atol=1e-14)
    np.testing.assert_array_equal(a, before)


def test_positive_affine_transform_preserves_standardized_moments():
    a = np.random.default_rng(131).normal(size=(2, 3, 121))
    original, transformed = channel_basic_statistics(a), channel_basic_statistics(a * 3 + 7)
    np.testing.assert_allclose(transformed[..., 2:4], original[..., 2:4], atol=1e-13)
    np.testing.assert_allclose(transformed[..., 1], original[..., 1] * 3)
    np.testing.assert_allclose(transformed[..., [0, 4, 5]], original[..., [0, 4, 5]] * 3 + 7)


@pytest.mark.parametrize('a', [np.ones((1, 2, 20)), np.zeros((1, 0, 5)), np.ones((1, 2, 1)),
                              np.full((1, 2, 10), np.nan), np.ones((2, 10)), np.ones((1, 2, 10), dtype=complex)])
def test_invalid_moment_domains_rejected(a):
    with pytest.raises(ValueError):
        channel_basic_statistics(a)
