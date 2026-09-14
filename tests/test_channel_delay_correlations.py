import numpy as np
import pytest
from sciona.atoms.riemannian_bci.covariance_features.delay_correlation import channel_delay_correlations


def test_delay_correlation_matches_scalar_circular_reference_and_layout():
    rng=np.random.default_rng(6);a=rng.normal(size=(2,3,80));before=a.copy()
    result=channel_delay_correlations(a,[1,2,4],2)
    assert result.shape==(2,4,4,3)
    for wi in range(2):
        for ci in range(3):
            values=a[wi,ci,::2];centered=values-values.mean()
            for i,di in enumerate([0,1,2,4]):
                for j,dj in enumerate([0,1,2,4]):
                    expected=sum(centered[(k-di)%40]*centered[(k-dj)%40] for k in range(40))/sum(centered**2)
                    assert result[wi,i,j,ci]==pytest.approx(expected)
    np.testing.assert_array_equal(a,before)
    np.testing.assert_allclose(channel_delay_correlations(a,4,2),channel_delay_correlations(a,[1,2,3],2))


def test_singular_correlation_is_not_mislabeled_as_strictly_positive_definite():
    a=np.tile([0.,1.],20).reshape(1,1,-1)
    matrix=channel_delay_correlations(a,[1,2],1)[0,:,:,0]
    assert np.linalg.matrix_rank(matrix)==1
    np.testing.assert_allclose(np.diag(matrix),1.)


@pytest.mark.parametrize('delays,subsample',[(0,1),([0,1],1),([1,1],1),([100],1),([1],0),([1],True)])
def test_invalid_delay_contracts_rejected(delays,subsample):
    with pytest.raises(ValueError):channel_delay_correlations(np.arange(40.).reshape(1,1,40),delays,subsample)
