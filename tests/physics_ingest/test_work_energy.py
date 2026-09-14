from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.work_energy import work_energy


def reference(*args):
    results=[[],[],[]]
    with localcontext() as ctx:
        ctx.prec=2500
        for row in zip(*(np.asarray(a).flat for a in args)):
            m,u,v=map(lambda x:Decimal.from_float(float(x)),row)
            first,last=m*u*u/2,m*v*v/2
            for bucket,value in zip(results,[first,last,m*(v-u)*(v+u)/2]):bucket.append(float(value))
    return tuple(np.asarray(r).reshape(np.asarray(args[0]).shape) for r in results)


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_independent_decimal_and_input_preservation(shape):
    rng=np.random.default_rng(512);args=[rng.uniform(.01,10,size=shape),rng.normal(size=shape),rng.normal(size=shape)]
    originals=[a.copy() for a in args]
    for actual,wanted in zip(work_energy(*args),reference(*args)):np.testing.assert_array_equal(actual,wanted)
    for actual,wanted in zip(args,originals):np.testing.assert_array_equal(actual,wanted)


@pytest.mark.parametrize('row',[(2.,3.,-3.),(2.,3.,1.),(2.,0.,0.),(1e-300,1e300,1e300),
                               (2.,1.,np.nextafter(1.,2.)),(np.nextafter(0.,1.),2.,2.)])
def test_signed_work_reversal_cancellation_and_extremes(row):
    args=list(map(np.array,row))
    for actual,wanted in zip(work_energy(*args),reference(*args)):np.testing.assert_array_equal(actual,wanted)


@pytest.mark.parametrize('u,a,t',[(3.,2.,4.),(3.,-2.,1.),(3.,-2.,3.),(3.,0.,4.)])
def test_independent_force_displacement(u,a,t):
    m=2.;displacement=u*t+a*t*t/2
    assert work_energy(*map(np.array,[m,u,u+a*t]))[2]==m*a*displacement


def test_work_not_subtraction_of_rounded_energies():
    args=list(map(np.array,[1.1,1.,np.nextafter(1.,2.)]))
    first,last,work=work_energy(*args)
    assert work==reference(*args)[2]
    assert work!=last-first


@pytest.mark.parametrize('row',[(0.,1.,2.),(-1.,1.,2.),(np.inf,1.,2.),(1.,np.nan,2.),
                               (1e308,2.,3.),(np.nextafter(0.,1.),.5,1.)])
def test_invalid_or_unrepresentable(row):
    with pytest.raises(ValueError):work_energy(*map(np.array,row))


@pytest.mark.parametrize('bad',[[],[True],['1'],[1j],[None],np.ones(2)])
def test_types_empty_and_shapes(bad):
    with pytest.raises(ValueError):work_energy(np.asarray(bad),np.ones(1),np.ones(1))
