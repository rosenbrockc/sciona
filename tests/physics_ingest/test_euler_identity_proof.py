import json
import sympy as sp
from sciona.atoms.physical_quantities.euler_identity_proof import euler_identity_proof,witness_euler_identity_proof
from sciona.physics_ingest.source_symbolic import parse_source_srepr


def test_certificate_contains_exact_computed_equations():
    result=euler_identity_proof()
    assert result==json.loads(json.dumps(result))
    assert len(result['steps'])==4
    for step in result['steps']:
        equation=parse_source_srepr(step['equation'])
        assert isinstance(equation,sp.Equality)
        assert equation.lhs==equation.rhs
        assert not equation.atoms(sp.Float)
    assert result['terminal_equation']=='Equality(Integer(0), Integer(0))'


def test_initial_identity_retains_general_real_angle():
    result=euler_identity_proof();theta=sp.Symbol('theta',real=True)
    assert result['initial_equation']==sp.srepr(sp.Eq(sp.exp(sp.I*theta),sp.cos(theta)+sp.I*sp.sin(theta),evaluate=False))
    assert result['steps'][0]['operation']=='substitute theta=pi'


def test_repeat_execution_independent_and_witness_scope():
    first=euler_identity_proof();first['steps'].clear()
    second=euler_identity_proof()
    assert len(second['steps'])==4
    assert witness_euler_identity_proof()['step_count']==4
