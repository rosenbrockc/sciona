import pytest
import sympy as sp
from sciona.physics_ingest.series_source_review import classify_series_equation,CURRENT,VOLTAGE,RESISTANCE


def test_classification_requires_equation_and_dimensional_agreement():
    v,i,r=sp.symbols('v i r')
    assert classify_series_equation(sp.Eq(v,i*r),{'v':VOLTAGE,'i':CURRENT,'r':RESISTANCE})=='ohmic_voltage_current_resistance'
    with pytest.raises(ValueError):classify_series_equation(sp.Eq(v,i*r),{'v':RESISTANCE,'i':CURRENT,'r':RESISTANCE})
    with pytest.raises(ValueError):classify_series_equation(sp.Eq(v,2*i*r),{'v':VOLTAGE,'i':CURRENT,'r':RESISTANCE})


def test_series_and_parallel_models_are_not_interchanged():
    a,b,total=sp.symbols('a b total');dims={str(s):RESISTANCE for s in [a,b,total]}
    assert classify_series_equation(sp.Eq(total,a+b),dims)=='series_equivalent_resistance'
    with pytest.raises(ValueError):classify_series_equation(sp.Eq(total,a*b/(a+b)),dims)


def test_voltage_addition_and_intermediate_ohmic_balance():
    a,b,total,i=sp.symbols('a b total i')
    assert classify_series_equation(sp.Eq(total,a+b),{str(s):VOLTAGE for s in [a,b,total]})=='series_voltage_addition'
    assert classify_series_equation(sp.Eq(i*total,i*a+i*b),{**{str(s):RESISTANCE for s in [a,b,total]},'i':CURRENT})=='ohmic_series_voltage_balance'
