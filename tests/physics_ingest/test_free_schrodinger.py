"""Symbolic mode validation using separable fields and independent derivatives."""
import json
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.free_schrodinger import free_schrodinger
from sciona.physics_ingest.source_symbolic import parse_source_srepr


def exact(expression):
    return expression.xreplace({f: sp.Rational(f) for f in expression.atoms(sp.Float)}).doit()


def equal(lhs, rhs):
    if lhs == rhs:
        return True
    if lhs == 0 or rhs == 0:
        return sp.simplify(lhs-rhs) == 0
    # Compare nonzero analytic expressions by their quotient. All exponential
    # factors are nonzero; the cleared identity extends to coefficient zeros.
    quotient = sp.powsimp(sp.cancel(lhs/rhs), deep=True, combine='exp')
    normalized = quotient.replace(lambda e: e.func == sp.exp,
                                  lambda e: sp.exp(sp.cancel(e.args[0])))
    return sp.simplify(normalized-1) == 0


def check_outputs(args, outputs):
    m, hbar, momentum, amplitude = map(exact, args)
    x, y, z, t = sp.symbols('x y z t', real=True)
    coordinates = (x, y, z)
    # Separable space/time factors, independently differentiated.
    wave = amplitude
    for p, q in zip(momentum, coordinates):
        wave *= sp.exp(sp.I*p*q/hbar)*sp.exp(-sp.I*p*p*t/(2*m*hbar))
    values = [parse_source_srepr(v).doit() for v in outputs[:2]]
    values += [[parse_source_srepr(v).doit() for v in json.loads(outputs[2])]]
    values += [parse_source_srepr(v).doit() for v in outputs[3:5]]
    expected = [wave, sum(p*p/(2*m) for p in momentum),
                [sp.diff(wave, q) for q in coordinates],
                sum(sp.diff(wave, q, 2) for q in coordinates), sp.diff(wave, t)]
    for i in [0, 1, 3, 4]:
        assert equal(values[i],expected[i])
    assert len(values[2]) == 3
    for a, b in zip(values[2], expected[2]):
        assert equal(a,b)
    certificate = json.loads(outputs[5])
    assert certificate['schema'] == 'sciona.free-schrodinger-plane-wave.v1'
    assert certificate['source_ast_parity'] is False
    assert len(certificate['identities']) == 7
    for identity in certificate['identities']:
        lhs, rhs = [parse_source_srepr(identity[k]).doit() for k in ['lhs_srepr', 'rhs_srepr']]
        assert equal(lhs,rhs)
    for name, expected_input in zip(['mass_srepr', 'hbar_srepr', 'momentum_srepr', 'amplitude_srepr'],
                                    [m, hbar, momentum, amplitude]):
        actual = parse_source_srepr(certificate['inputs'][name]).doit()
        assert actual == expected_input
    assert [parse_source_srepr(v).doit() for v in certificate['coordinates_srepr']] == [x, y, z, t]
    action = parse_source_srepr(certificate['hamiltonian_action_srepr']).doit()
    assert equal(action,-hbar**2*expected[3]/(2*m))
    assert equal(action,sp.I*hbar*expected[4])


def cases():
    m, hbar = sp.symbols('mass action', positive=True)
    px, py, pz, ar, ai = sp.symbols('px py pz ar ai', real=True)
    return [(sp.Integer(2), sp.Integer(3), sp.Tuple(1, -2, 3), 1+sp.I),
            (sp.Integer(1), sp.Integer(1), sp.Tuple(0, 0, 0), 2-sp.I),
            (sp.Integer(2), sp.Integer(3), sp.Tuple(1, 2, 3), sp.Integer(0)),
            (sp.Rational(1, 100), sp.Rational(1, 1000), sp.Tuple(sp.Rational(1, 3), 0, -2), sp.sqrt(2)),
            (m, hbar, sp.Tuple(px, py, pz), ar+sp.I*ai),
            (m+1, hbar, sp.Tuple(px**2, sp.sin(py), -pz), sp.exp(sp.I*ar))]


@pytest.mark.parametrize('args', cases())
def test_exact_and_general_three_dimensional_modes(args):
    check_outputs(args, free_schrodinger(*(sp.srepr(a) for a in args)))


def test_float_inputs_become_exact_before_arithmetic():
    args = (sp.Float('0.3'), sp.Float('0.7'), sp.Tuple(sp.Float('0.1'), 2, 0), sp.Float('0.2')+sp.I)
    outputs = free_schrodinger(*(sp.srepr(a) for a in args))
    check_outputs(args, outputs)
    assert 'Float(' not in ''.join(outputs)


@pytest.mark.parametrize('index,value', [(0, sp.Integer(0)), (0, sp.Integer(-1)),
    (1, sp.Integer(0)), (1, sp.Symbol('unknown', real=True)), (0, sp.oo),
    (2, sp.Tuple(1, 2)), (2, sp.Tuple(1, 2, sp.I)), (2, sp.Tuple(1, 2, sp.Symbol('unknown'))),
    (3, sp.oo), (3, sp.Symbol('unknown')), (0, sp.Symbol('x', positive=True)),
    (3, sp.exp(sp.I*sp.Symbol('t', real=True)))])
def test_invalid_or_unproved_parameter_domains(index, value):
    args = list(cases()[0]); args[index] = value
    with pytest.raises(ValueError):
        free_schrodinger(*(sp.srepr(a) for a in args))


def test_conflicting_assumptions_across_parameters():
    args = (sp.Symbol('a', positive=True), sp.Integer(1), sp.Tuple(sp.Symbol('a', real=True), 0, 0), sp.Integer(1))
    with pytest.raises(ValueError, match='Conflicting'):
        free_schrodinger(*(sp.srepr(a) for a in args))


@pytest.mark.parametrize('payload', ["Add('1+2', Integer(1))", "Symbol('bad name')", "__import__('os')", "Integral('x', Symbol('x'))"])
def test_string_constructor_escape_rejected(payload):
    with pytest.raises(ValueError):
        free_schrodinger(payload, 'Integer(1)', 'Tuple(Integer(1),Integer(0),Integer(0))', 'Integer(1)')
