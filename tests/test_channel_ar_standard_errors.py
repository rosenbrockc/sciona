import numpy as np
import pytest
from scipy.linalg import solve_triangular
from sciona.atoms.riemannian_bci.signal_processing.ar_standard_errors import channel_ar_standard_errors


def test_standard_errors_match_independent_qr_and_lag_order():
    a = np.random.default_rng(707).normal(size=(2, 3, 121))
    before = a.copy()
    actual = channel_ar_standard_errors(a, 3, 2)
    assert actual.shape == (2, 3, 4)
    for wi in range(2):
        for ci in range(3):
            values = a[wi, ci, ::2]
            design = np.array([[1., values[t-1], values[t-2], values[t-3]] for t in range(3, len(values))])
            q, r = np.linalg.qr(design, mode='reduced')
            coefficients = solve_triangular(r, q.T @ values[3:])
            residuals = values[3:] - design @ coefficients
            inverse_r = solve_triangular(r, np.eye(4))
            variance = sum(residuals ** 2) / (len(values) - 7)
            expected = np.sqrt(np.diag(inverse_r @ inverse_r.T) * variance)
            np.testing.assert_allclose(actual[wi, ci], expected, atol=1e-14)
    np.testing.assert_array_equal(a, before)


def test_order_one_matches_scalar_regression_standard_errors():
    values = np.array([1., 4., 2., 8., 3., 9., 7., 5.])
    x, y = values[:-1], values[1:]
    xx = sum((x - x.mean()) ** 2)
    slope = sum((x - x.mean()) * (y - y.mean())) / xx
    intercept = y.mean() - slope * x.mean()
    variance = sum((y - intercept - slope*x) ** 2) / (len(x) - 2)
    expected = [np.sqrt(variance * (1/len(x) + x.mean()**2/xx)), np.sqrt(variance/xx)]
    np.testing.assert_allclose(channel_ar_standard_errors(values[None, None], 1, 1)[0, 0], expected)


@pytest.mark.parametrize('order,subsample', [(0, 1), (True, 1), (2, 0), (2, True), (20, 1)])
def test_invalid_ar_geometry_rejected(order, subsample):
    with pytest.raises(ValueError):
        channel_ar_standard_errors(np.arange(30.)[None, None], order, subsample)


def test_rank_deficiency_and_nonfinite_input_rejected():
    for values in [np.ones(80), np.arange(80.), np.full(80, np.nan)]:
        with pytest.raises(ValueError):
            channel_ar_standard_errors(values[None, None], 5, 1)
