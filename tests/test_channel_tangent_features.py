import numpy as np
import pytest
from sciona.atoms.riemannian_bci.covariance_features.source_tangent import channel_tangent_features


def diagonal(values):
    return np.stack([np.diag(row) for row in values])[..., None]


def test_diagonal_log_reference_and_prediction_update_have_analytic_solution():
    train = diagonal([[1., 4.], [9., 16.]])
    pred = diagonal([[4., 25.], [16., 49.]])
    def logs(a):
        d = np.diagonal(a[..., 0], axis1=1, axis2=2)
        return np.log((1 - 1e-9) * d + 1e-9 * d.mean(axis=1, keepdims=True))
    for update in [True, False]:
        tr, te = channel_tangent_features(train, pred, tsupdate=update)
        np.testing.assert_allclose(tr[:, [0, 2]], logs(train) - logs(train).mean(axis=0), atol=1e-12)
        reference = logs(pred).mean(axis=0) if update else logs(train).mean(axis=0)
        np.testing.assert_allclose(te[:, [0, 2]], logs(pred) - reference, atol=1e-12)
        np.testing.assert_array_equal(tr[:, 1], 0.)


def test_identity_reference_layout_offdiagonal_weight_and_no_mutation():
    q = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)
    a = np.stack([q @ np.diag([2., 5.]) @ q.T, np.diag([3., 7.])])[None, ...].transpose(0, 2, 3, 1)
    before = a.copy()
    train, pred = channel_tangent_features(a, a, 'identity', False)
    expected = []
    for matrix in a[0].transpose(2, 0, 1):
        values, vectors = np.linalg.eigh(matrix)
        log = vectors @ np.diag(np.log((1 - 1e-9) * values + 1e-9 * values.mean())) @ vectors.T
        expected.extend([log[0, 0], np.sqrt(2.) * log[0, 1], log[1, 1]])
    np.testing.assert_allclose(train[0], expected, atol=1e-12)
    np.testing.assert_array_equal(train, pred)
    np.testing.assert_array_equal(a, before)


def test_singular_psd_is_explicitly_shrunk():
    a = np.ones((2, 3, 3, 1))
    tr, te = channel_tangent_features(a, a, 'identity', False)
    assert tr.shape == (2, 6) and np.all(np.isfinite(tr))
    np.testing.assert_array_equal(tr, te)


@pytest.mark.parametrize('matrix', [np.zeros((2, 2)), np.diag([-1., 2.]), np.array([[1., .2], [.3, 1.]]), np.full((2, 2), np.nan)])
def test_invalid_matrices_rejected(matrix):
    a = matrix[None, ..., None]
    with pytest.raises(ValueError):
        channel_tangent_features(a, a)
