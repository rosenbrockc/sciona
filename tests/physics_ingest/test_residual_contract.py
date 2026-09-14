import numpy as np
import pytest
import sympy as sp
from sciona.physics_ingest.equation_runtime import compile_equation_runtime
from sciona.physics_ingest.residual_contract import build_residual_contract


def bundle():
    voltage,current,resistance=sp.symbols('V I R')
    equation=sp.Eq(voltage,current*resistance,evaluate=False)
    runtime=compile_equation_runtime(sp.Eq(sp.Symbol('_validation_residual'),equation.lhs-equation.rhs,evaluate=False))
    expression={'sympy_srepr':sp.srepr(equation),'evidence_json':{'numpy_runtime':{
        'kind':'equation_residual_validator','tests_passed':True,'source':runtime.source,
        'runtime_source_sha256':runtime.source_sha256,'argument_symbols':list(runtime.argument_symbols)}}}
    variables=[{'symbol_name':name,'dim_signature':dim} for name,dim in [('V','M1L2T-3I-1'),('I','I1'),('R','M1L2T-3I-2')]]
    return expression,variables


def test_residual_interface_matches_callable_and_physical_units():
    expression,variables=bundle()
    contract=build_residual_contract(expression,variables)
    namespace={};exec(expression['evidence_json']['numpy_runtime']['source'],namespace)
    values={'I':np.array([2.,3.]),'R':4.,'V':np.array([8.,13.])}
    np.testing.assert_allclose(namespace['evaluate'](**{alias:values[name] for alias,name in contract['argument_symbols'].items()}),[0.,1.])
    assert contract['ports'][-1]['dim_signature']=='M1L2T-3I-1'
    assert contract['ports'][-1]['name']=='residual'


def test_residual_contract_rejects_stale_code_and_wrong_dimensions():
    expression,variables=bundle()
    expression['evidence_json']['numpy_runtime']['source']+='\n# drift'
    with pytest.raises(ValueError,match='no longer matches'):build_residual_contract(expression,variables)
    expression,variables=bundle();variables[0]['dim_signature']='L1'
    with pytest.raises(ValueError,match='matching dimensions'):build_residual_contract(expression,variables)
