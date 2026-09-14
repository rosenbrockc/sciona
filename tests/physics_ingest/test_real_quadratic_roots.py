import numpy as np
import pytest
from sciona.atoms.physical_quantities.quadratic_roots import real_quadratic_roots


@pytest.mark.parametrize('a,b,c,expected',[(2.,3.,1.,(-1.,-.5)),(-2.,-3.,-1.,(-1.,-.5)),
    (1.,2.,1.,(-1.,-1.)),(1.,0.,0.,(0.,0.)),(1.,-3.,0.,(0.,3.)),
    (1e308,0.,-1e308,(-1.,1.)),(1e-300,0.,-1e-300,(-1.,1.)),
    (1.,-1e16,1.,(1e-16,1e16))])
def test_roots(a,b,c,expected):
    lo,hi=real_quadratic_roots(a,b,c)
    np.testing.assert_allclose([lo,hi],expected,rtol=2e-15,atol=0)
    assert lo.shape==() and hi.shape==()


def test_exact_discriminant_rejects_tiny_negative():
    # Naive floating arithmetic rounds b*b to the same value as 4*a*c.
    b=1.+2**-27
    c=np.nextafter(b*b/4,np.inf)
    with pytest.raises(ValueError,match='Negative discriminant'):real_quadratic_roots(1.,b,c)


def test_large_coefficients_small_root_and_subnormal():
    lo,hi=real_quadratic_roots(1.,-1e308,1.)
    np.testing.assert_allclose(lo,1e-308,rtol=1e-15,atol=0)
    assert hi==1e308


def test_shapes_and_no_mutation():
    a=np.array([[1.,-1.]]);b=np.array([[-3.,3.]]);c=np.array([[2.,-2.]])
    originals=[x.copy() for x in (a,b,c)]
    lo,hi=real_quadratic_roots(a,b,c)
    np.testing.assert_array_equal(lo,[[1.,1.]])
    np.testing.assert_array_equal(hi,[[2.,2.]])
    for value,old in zip((a,b,c),originals):np.testing.assert_array_equal(value,old)


@pytest.mark.parametrize('a,b,c',[(0.,1.,1.),(True,1.,1.),(1j,1.,1.),('1',1.,1.),
    (1.,np.nan,1.),(1.,1.,np.inf),([],[],[]),([1.],1.,[1.]),(1e-308,-1e308,1.),(1e308,-1.,1e-308)])
def test_invalid_or_unrepresentable(a,b,c):
    with pytest.raises((ValueError,FloatingPointError)):real_quadratic_roots(a,b,c)
