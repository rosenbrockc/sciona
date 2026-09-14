from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.atoms.physical_quantities.schwarzschild_radius import schwarzschild_radius,witness_schwarzschild_radius
from sciona.ghost.abstract import AbstractArray


def reference(m,g,c):
    with localcontext() as ctx:
        ctx.prec=100
        return float(2*Decimal.from_float(float(g))*Decimal.from_float(float(m))/Decimal.from_float(float(c))**2)


@pytest.mark.parametrize('m,g,c',[(1.,1.,1.),(1e308,1e308,1e308),(1e-300,1e-300,1e-300),
    (np.nextafter(0.,1.),1.,1.),(1e100,6.6743e-11,299792458.),(1e-100,6.6743e-11,299792458.)])
def test_high_precision_reference_and_extreme_scaling(m,g,c):
    actual=schwarzschild_radius(np.asarray(m),g,c)
    np.testing.assert_array_max_ulp(actual,np.asarray(reference(m,g,c)),maxulp=3)
    assert actual.shape==() and actual.dtype==np.float64


def test_shape_no_mutation_and_linear_mass_scaling():
    masses=np.array([[1.,2.],[4.,8.]])
    copy=masses.copy()
    out=schwarzschild_radius(masses,3.,2.)
    np.testing.assert_array_equal(out,1.5*masses)
    np.testing.assert_array_equal(masses,copy)
    assert not np.shares_memory(out,masses)
    assert witness_schwarzschild_radius(AbstractArray(shape=(2,2),dtype='float64'),3.,2.).shape==(2,2)


@pytest.mark.parametrize('bad',[True,1j,'1',[],0.,-1.,np.nan,np.inf])
def test_invalid_mass(bad):
    with pytest.raises((ValueError,FloatingPointError)):
        schwarzschild_radius(bad,1.,1.)


@pytest.mark.parametrize('g,c',[(True,1.),(1.,False),([1.],1.),(1.,[1.]),(0.,1.),(1.,0.),(np.inf,1.),(1.,np.nan)])
def test_invalid_constants(g,c):
    with pytest.raises((ValueError,FloatingPointError)):
        schwarzschild_radius(np.array([1.]),g,c)


@pytest.mark.parametrize('m,g,c',[(1e308,1e308,1.),(1e-300,1e-300,1e300)])
def test_unrepresentable_output_rejected(m,g,c):
    with pytest.raises((ValueError,FloatingPointError)):
        schwarzschild_radius(np.asarray(m),g,c)
