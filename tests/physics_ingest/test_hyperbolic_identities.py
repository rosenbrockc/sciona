import json
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.hyperbolic_identities import hyperbolic_identities, _checked_parse, _verify, _derive
from sciona.physics_ingest.hyperbolic_identities_proof import build_proof, verify_proof


def check_outputs(argument, outputs):
    functions, identities, certificate = map(json.loads, outputs)
    expected = dict(sinh=sp.sinh(argument), cosh=sp.cosh(argument),
                    tanh=sp.tanh(argument), sech=sp.sech(argument),
                    sin_imaginary=sp.sin(sp.I*argument), cos_imaginary=sp.cos(sp.I*argument))
    assert set(functions) == set(expected)
    for name, value in expected.items():
        assert sp.simplify((_checked_parse(functions[name]).doit()-value).rewrite(sp.exp)) == 0
    assert len(identities) == 18
    x = sp.Symbol('x', real=True)
    for i, (actual, eq) in enumerate(zip(identities, build_proof())):
        assert actual['step'] == i+1
        for key, side in [('lhs_srepr', eq.lhs), ('rhs_srepr', eq.rhs)]:
            assert actual[key] == sp.srepr(side.xreplace({x: _checked_parse(sp.srepr(argument))}))
        assert sp.simplify((_checked_parse(actual['lhs_srepr']).doit()-_checked_parse(actual['rhs_srepr']).doit()).rewrite(sp.exp)) == 0
    assert certificate['functions'] == functions
    assert certificate['specialized_identities'] == identities
    assert certificate['argument_srepr'] == sp.srepr(argument)
    assert certificate['steps'] == [sp.srepr(e) for e in build_proof()]
    assert all(isinstance(_checked_parse(e), sp.Equality) for e in certificate['steps'])
    report = verify_proof(build_proof())
    del report['source_version_id'], report['source_content_hash']
    assert certificate['verification'] == report


@pytest.mark.parametrize('argument', [sp.Integer(0), -sp.pi/2, sp.Rational(1,10)**100,
    sp.sqrt(2), sp.Symbol('u',real=True), sp.Symbol('u',real=True)**2+sp.Rational(1,3)])
def test_full_collection_and_certificate(argument):
    check_outputs(argument, hyperbolic_identities(sp.srepr(argument)))


@pytest.mark.parametrize('argument', [sp.I, sp.oo, sp.nan, sp.Symbol('unknown'), sp.Tuple(1,2)])
def test_invalid_domain(argument):
    with pytest.raises(ValueError):
        hyperbolic_identities(sp.srepr(argument))


@pytest.mark.parametrize('source', ["Integral('1+1')", "Add('1',Integer(1))", "__import__('os')", ''])
def test_constructor_string_guard(source):
    with pytest.raises(ValueError):
        hyperbolic_identities(source)


def test_conflicting_symbol_assumptions():
    a, b = sp.Symbol('a',positive=True), sp.Symbol('a',negative=True)
    with pytest.raises(ValueError):
        hyperbolic_identities(sp.srepr(a+b))


def test_runtime_certificate_mutation():
    steps = _derive()
    steps[6] = sp.Eq(steps[6].lhs, -steps[6].rhs, evaluate=False)
    with pytest.raises(ValueError):
        _verify(steps)


def test_general_argument_preserved():
    u = sp.Symbol('u', real=True)
    functions = json.loads(hyperbolic_identities(sp.srepr(u))[0])
    assert all(_checked_parse(encoded).has(u) for encoded in functions.values())
