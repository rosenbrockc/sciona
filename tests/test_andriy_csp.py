import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_csp import andriy_csp_filters, andriy_csp_project


def classes():
    rng = np.random.default_rng(591)
    return (rng.normal(size=(16,300))*np.linspace(1,3,16)[:,None],
            rng.normal(size=(16,400))*np.linspace(3,1,16)[:,None])


def test_filters_diagonalize_classes_and_sort_ratios_without_mutation():
    a,b = classes()
    originals = a.copy(),b.copy()
    filters = andriy_csp_filters(a,b)
    ra,rb = a@a.T,b@b.T
    ra,rb = ra/np.trace(ra),rb/np.trace(rb)
    da,db = filters@ra@filters.T,filters@rb@filters.T
    np.testing.assert_allclose(da,np.diag(np.diag(da)),atol=1e-12)
    np.testing.assert_allclose(db,np.diag(np.diag(db)),atol=1e-12)
    assert np.all(np.diff(np.diag(da)/np.diag(db))>0)
    assert np.all(filters[np.arange(16),np.argmax(abs(filters),axis=1)]>0)
    np.testing.assert_array_equal(a,originals[0])
    np.testing.assert_array_equal(b,originals[1])


def test_independent_class_amplitude_scaling_is_removed():
    a,b=classes()
    np.testing.assert_allclose(andriy_csp_filters(-3*a,7*b),andriy_csp_filters(a,b),atol=1e-10,rtol=1e-10)


def test_projection_uses_first_filter_and_preserves_inputs():
    x=np.arange(16*20.).reshape(16,20)
    coefficients=np.eye(16)[::-1].copy()
    original=coefficients.copy()
    np.testing.assert_array_equal(andriy_csp_project(x,coefficients),x[-1:])
    np.testing.assert_array_equal(coefficients,original)


def test_singular_and_repeated_eigenvalue_cases_fail_explicitly():
    a,b=classes()
    singular=a.copy();singular[1]=singular[0]
    with pytest.raises(ValueError,match='singular'):
        andriy_csp_filters(singular,b)
    with pytest.raises(ValueError,match='distinct'):
        andriy_csp_filters(a,a)
    with pytest.raises(ValueError,match='distinct'):
        andriy_csp_filters(np.eye(16),np.eye(16))


@pytest.mark.parametrize('bad',[np.zeros((16,30)),np.ones((15,30)),np.ones((16,15)),np.full((16,30),np.nan),np.ones((16,30),dtype=complex)])
def test_invalid_training_inputs(bad):
    _,b=classes()
    with pytest.raises(ValueError):
        andriy_csp_filters(bad,b)


def test_invalid_projection_filters():
    with pytest.raises(ValueError):
        andriy_csp_project(np.ones((16,30)),np.eye(15))
