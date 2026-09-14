from decimal import Decimal,localcontext
from dataclasses import replace
import numpy as np
import sympy as sp
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.isobaric_internal_energy import isobaric_internal_energy
from sciona.physics_ingest.isobaric_energy_proof import build_proof,verify_proof,symbols


def reference(args):
    outputs=[]
    with localcontext() as ctx:
        ctx.prec=2500
        for row in zip(*(np.asarray(a).flat for a in args)):
            cv,pi,V,alpha=[Decimal.from_float(float(v)) for v in row]
            outputs.append(float(cv+pi*V*alpha))
    return np.array(outputs).reshape(np.asarray(args[0]).shape)


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_independent_decimal_reference(shape):
    rng=np.random.default_rng(901)
    args=[rng.normal(size=shape),rng.normal(size=shape),rng.uniform(.01,10,size=shape),rng.normal(size=shape)]
    before=[a.copy() for a in args]
    np.testing.assert_array_equal(isobaric_internal_energy(*args),reference(args))
    for a,b in zip(args,before):np.testing.assert_array_equal(a,b)


def test_independent_differentiable_energy_and_isobaric_path():
    T,V=sp.symbols('T V',real=True)
    energy=3*T*T-2*T*V+V**3/4;path=T*T/2
    derivative=sp.diff(energy.subs(V,path),T)
    for t in [.5,1.,2.,4.]:
        volume=path.subs(T,t)
        cv=sp.diff(energy,T).subs({T:t,V:volume});pi=sp.diff(energy,V).subs({T:t,V:volume})
        alpha=sp.diff(path,T).subs(T,t)/volume
        actual=isobaric_internal_energy(*[np.array(float(a)) for a in [cv,pi,volume,alpha]])
        assert actual==float(derivative.subs(T,t))


def test_ideal_gas_limit_is_not_cp():
    assert isobaric_internal_energy(*map(np.array,[3.,0.,2.,.5]))==3.
    # For this synthetic ideal-gas state p*V*alpha=1, so Cp would instead be 4.


@pytest.mark.parametrize('row',[[1e308,-1e308,2.,.5],[0.,1e308,1e308,1e-308],
                              [np.nextafter(0.,1.),0.,1.,1.],[2.,-3.,4.,-5.]])
def test_signed_coefficients_cancellation_and_extremes(row):
    args=list(map(np.array,row))
    np.testing.assert_array_equal(isobaric_internal_energy(*args),reference(args))


@pytest.mark.parametrize('i,value',[(0,np.nan),(1,np.inf),(2,0.),(2,-1.)])
def test_invalid_fields(i,value):
    args=[np.ones(1) for _ in range(4)];args[i][:]=value
    with pytest.raises(ValueError):isobaric_internal_energy(*args)


@pytest.mark.parametrize('bad',[np.array([]),np.ones(2),np.ones((1,1)),np.array([1j]),np.array([True]),np.array(['1'])])
def test_invalid_shapes_and_types(bad):
    with pytest.raises(ValueError):isobaric_internal_energy(bad,np.ones(1),np.ones(1),np.ones(1))


@pytest.mark.parametrize('row',[[1e308,1e308,2.,1.],[0.,np.nextafter(0.,1.),.5,1.]])
def test_unrepresentable_output_rejected(row):
    with pytest.raises(ValueError):isobaric_internal_energy(*map(np.array,row))


def test_local_differential_proof():
    assert all(verify_proof(build_proof())['checks'].values())


@pytest.mark.parametrize('i',range(5))
def test_corrupted_step_rejected(i):
    proof=build_proof();steps=list(proof.steps);steps[i]=sp.Eq(sp.Symbol('bad'),0,evaluate=False)
    with pytest.raises(ValueError):verify_proof(replace(proof,steps=tuple(steps)))
