"""Conservative two-equation substitutions and checked rational unity feeds.

Earlier replay implementations remain immutable. Substitution is structural and
simultaneous; this module does not infer unstated algebraic rewrites.
"""
import sympy as sp

from sciona.physics_ingest.pdg_real_power_rules import replay_real_power_step
from sciona.physics_ingest.pdg_rational_rules_v2 import replay_rational_step as replay_v2


def replay_rational_step(rule, inputs, feeds, expected):
    # V3 reserved the power rule for real-root squaring. Other integer powers
    # remain valid for the strictly rational grammar already enforced by V2.
    if rule.name == 'raise both sides to power' and len(feeds) == 1 and feeds[0] != 2:
        return replay_v2(rule, inputs, feeds, expected)
    names = {
        'substitute LHS of expr 1 into expr 2',
        'substitute RHS of expr 1 into expr 2',
        'multiply RHS by unity',
    }
    if rule.name not in names:
        return replay_real_power_step(rule, inputs, feeds, expected)
    wanted = (1, 1, 1) if rule.name == 'multiply RHS by unity' else (2, 0, 1)
    if (rule.inputs, rule.feeds, rule.outputs) != wanted or (len(inputs), len(feeds)) != wanted[:2]:
        raise ValueError('invalid operation contract')
    if not all(isinstance(e, sp.Equality) for e in [*inputs, expected]):
        raise ValueError('equations required')
    conditions = set()

    def inspect(expression):
        for node in sp.preorder_traversal(expression):
            if isinstance(node, (sp.Symbol, sp.Rational, sp.Add, sp.Mul, sp.Equality)):
                continue
            if isinstance(node, sp.Pow) and isinstance(node.exp, sp.Integer):
                if node.exp < 0:
                    conditions.add(sp.Ne(node.base, 0, evaluate=False))
                continue
            raise ValueError('unsupported rational expression')

    # Inspect before simplification so a canceled denominator remains a condition.
    for expression in [*inputs, *feeds, expected]:
        inspect(expression)
    if rule.name == 'multiply RHS by unity':
        if sp.cancel(feeds[0] - 1) != 0:
            raise ValueError('feed is not rationally equal to unity')
        actual = sp.Eq(inputs[0].lhs, inputs[0].rhs * feeds[0], evaluate=False)
    else:
        premise, target = inputs
        source, replacement = (premise.rhs, premise.lhs) if 'LHS' in rule.name else (premise.lhs, premise.rhs)
        with sp.evaluate(False):
            actual = target.xreplace({source: replacement})
    inspect(actual)
    if any(sp.simplify(c) == sp.false for c in conditions):
        raise ValueError('impossible domain condition')
    if sp.cancel(actual.lhs - expected.lhs) != 0 or sp.cancel(actual.rhs - expected.rhs) != 0:
        raise ValueError('replayed equation differs from source conclusion')
    return actual, tuple(sorted(conditions, key=sp.srepr))
