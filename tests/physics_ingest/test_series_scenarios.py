import numpy as np
import pytest
import sympy as sp
from sciona.physics_ingest.series_scenarios import build_series_scenarios
from sciona.physics_ingest.series_source_review import CURRENT,RESISTANCE,VOLTAGE


def circuit():
    i=sp.Symbol('i');r,a,b,total=sp.symbols('r a b total');v,x,y,z=sp.symbols('v x y z')
    equations={'generic':sp.Eq(v,i*r),'first':sp.Eq(x,i*a),'second':sp.Eq(y,i*b),'total':sp.Eq(z,i*total),'loop':sp.Eq(z,x+y),'balance':sp.Eq(i*total,i*a+i*b),'equivalent':sp.Eq(total,a+b)}
    dims={**{str(s):RESISTANCE for s in [r,a,b,total]},**{str(s):VOLTAGE for s in [v,x,y,z]},'i':CURRENT}
    return equations,dims


def test_independent_circuit_scenarios_and_invalid_perturbation():
    equations,dims=circuit();values=build_series_scenarios(equations,dims)
    for equation in equations.values():
        symbols=sorted(equation.free_symbols,key=str)
        fn=sp.lambdify(symbols,equation.lhs-equation.rhs,'numpy')
        residual=fn(*(values[str(s)] for s in symbols))
        np.testing.assert_allclose(residual,0,atol=1e-12)
    assert np.all(values['i']!=0)
    assert np.max(np.abs(values['z']+1-values['x']-values['y']))>.9


def test_incomplete_model_does_not_invent_missing_premises():
    equations,dims=circuit();equations.pop('generic')
    with pytest.raises(ValueError,match='seven-expression'):build_series_scenarios(equations,dims)
