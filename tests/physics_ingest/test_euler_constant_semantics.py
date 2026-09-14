import hashlib
import pytest
import sympy as sp
from sciona.physics_ingest.euler_constant_semantics import REVIEWED,reviewed_mapping,verify_euler_steps
from sciona.physics_ingest.pdg_rule_contracts import RuleContract


def declarations():
    rows=[]
    for identity,fields in REVIEWED.items():
        fields={**fields,'name_latex':fields['name_latex'] or ''}
        fields.update({'dimension_'+k:'0' for k in ['time','electric_charge','luminous_intensity','length','amount_of_substance','mass','temperature']})
        props=','.join(k+':'+(v if k.startswith('dimension_') else '"'+v+'"') for k,v in fields.items())
        rows.append('UNWIND [{id:"'+identity+'", properties:{'+props+'}}] AS row\nCREATE (n:scalar{id: row.id}) SET n += row.properties SET n:symbol;\n')
    return ''.join(rows).encode()


def test_exact_identity_mapping_does_not_coerce_other_i():
    b=declarations();mapping,review=reviewed_mapping(b,hashlib.sha256(b).hexdigest())
    assert mapping[sp.Symbol('pdg0004621')]==sp.I
    assert sp.Symbol('i').xreplace(mapping)==sp.Symbol('i')
    assert sp.Symbol('pdg0001567').xreplace(mapping)==sp.Symbol('pdg0001567')
    assert review['source_discrepancies']


@pytest.mark.parametrize('change',[lambda b:b.replace(b'imaginary unit',b'index'),lambda b:b.replace(b'scope:"imaginary"',b'scope:"integer"'),lambda b:b.replace(b'dimension_mass:0',b'dimension_mass:1'),lambda b:b+b])
def test_changed_declarations_fail_even_with_new_hash(change):
    b=change(declarations())
    with pytest.raises(ValueError):reviewed_mapping(b,hashlib.sha256(b).hexdigest())


def test_wrong_pin_rejected():
    with pytest.raises(ValueError):reviewed_mapping(declarations(),'0'*64)


def test_false_root_identity_rejected():
    x=sp.Symbol('pdg0001464');I=sp.Symbol('pdg0004621')
    mapping={x:sp.Symbol('theta',real=True),I:sp.I}
    nodes=[dict(node_id='s',type_signature=dict(inference_rule_id='simplify',inputs=['a'],output='b',variable_bindings=dict(feeds=[])))]
    bad=sp.Eq(sp.exp(I*x),sp.cos(x)-I*sp.sin(x),evaluate=False)
    with pytest.raises(ValueError,match='Root identity'):
        verify_euler_steps(nodes,{'a':bad,'b':bad},{'simplify':RuleContract('simplify',1,0,1)},mapping)
