import pytest
import sympy as sp

from sciona.physics_ingest.pdg_rule_contracts import RuleContract
from sciona.physics_ingest.pdg_rational_rules_v4 import replay_rational_step
from sciona.physics_ingest.pdg_graph_replay_v4 import replay_derivation


x, y, z = sp.symbols('x y z')
eq = lambda a, b: sp.Eq(a, b, evaluate=False)


@pytest.mark.parametrize('side', ['LHS', 'RHS'])
def test_substitution_direction(side):
    premise = eq(y, x) if side == 'LHS' else eq(x, y)
    rule = RuleContract(f'substitute {side} of expr 1 into expr 2', 2, 0, 1)
    actual, _ = replay_rational_step(rule, [premise, eq(z, x**2)], [], eq(z, y**2))
    assert actual == eq(z, y**2)
    with pytest.raises(ValueError, match='differs'):
        replay_rational_step(rule, [premise, eq(z, x**2)], [], eq(z, y))


def test_unity_preserves_canceled_denominator():
    rule = RuleContract('multiply RHS by unity', 1, 1, 1)
    feed = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    _, conditions = replay_rational_step(rule, [eq(y, z)], [feed], eq(y, z))
    assert sp.Ne(x, 0, evaluate=False) in conditions


@pytest.mark.parametrize('feed', [sp.Integer(2), x, sp.sqrt(x)])
def test_unity_rejects_unproved_or_unsupported_feed(feed):
    with pytest.raises(ValueError):
        replay_rational_step(RuleContract('multiply RHS by unity', 1, 1, 1), [eq(y, z)], [feed], eq(y, z))


def test_invalid_arity_rejected():
    with pytest.raises(ValueError, match='contract'):
        replay_rational_step(RuleContract('multiply RHS by unity', 2, 0, 1), [eq(x, y), eq(y, z)], [], eq(x, z))


@pytest.mark.parametrize('side', ['LHS', 'RHS'])
def test_composition_transforms_inherited_nonzero_condition(side):
    def node(identity, inputs, output, rule, feeds):
        return dict(node_id=identity, type_signature=dict(inputs=inputs, outputs=[output], output=output,
            inference_rule_id=rule, variable_bindings=dict(feeds=[dict(sympy=sp.srepr(f)) for f in feeds])))
    rules = {'unity': RuleContract('multiply RHS by unity', 1, 1, 1),
             'sub': RuleContract(f'substitute {side} of expr 1 into expr 2', 2, 0, 1)}
    feed = sp.Mul(x, sp.Pow(x, -1, evaluate=False), evaluate=False)
    nodes = [node('a', ['root'], 'mid', 'unity', [feed]), node('b', ['premise', 'mid'], 'out', 'sub', [])]
    expressions = dict(root=eq(z, x), mid=eq(z, x), out=eq(z, y),
                       premise=eq(y, x) if side == 'LHS' else eq(x, y))
    result = replay_derivation(nodes, expressions, rules, edges=[('a', 'b')])
    from sciona.ghost.symbolic import deserialize_expr
    conditions = [deserialize_expr(c) for c in result['required_conditions']]
    assert sp.Ne(x, 0, evaluate=False) in conditions
    assert sp.Ne(y, 0, evaluate=False) in conditions


def test_simplify_does_not_silently_cancel_equation_factor():
    with pytest.raises(ValueError, match='differs'):
        replay_rational_step(RuleContract('simplify', 1, 0, 1), [eq(x*y, x*z)], [], eq(y, z))


def test_reciprocal_requires_nonzero_on_both_sides():
    _, conditions = replay_rational_step(RuleContract('raise both sides to power', 1, 1, 1),
        [eq(x, y)], [sp.Integer(-1)], eq(1/x, 1/y))
    assert sp.Ne(x, 0, evaluate=False) in conditions
    assert sp.Ne(y, 0, evaluate=False) in conditions


def test_reciprocal_does_not_enable_root_or_zero_inversion():
    rule = RuleContract('raise both sides to power', 1, 1, 1)
    with pytest.raises(ValueError):
        replay_rational_step(rule, [eq(sp.sqrt(x), y)], [sp.Integer(-1)], eq(1/sp.sqrt(x), 1/y))
    with pytest.raises(ValueError):
        replay_rational_step(rule, [eq(0, y)], [sp.Integer(-1)], eq(sp.zoo, 1/y))
