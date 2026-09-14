"""Forward squaring of principal real square roots with explicit domain bounds.

This establishes a consequence, never the reverse implication or a choice of
solution branch. All other steps retain the existing rational replay contract.
"""
import sympy as sp
from sciona.physics_ingest.pdg_rational_rules_v2 import replay_rational_step


def replay_real_power_step(rule, inputs, feeds, expected):
    if rule.name != 'raise both sides to power':
        return replay_rational_step(rule, inputs, feeds, expected)
    if (rule.inputs,rule.feeds,rule.outputs) != (1,1,1) or len(inputs)!=1 or feeds != [sp.Integer(2)]:
        raise ValueError('Real-root extension supports forward squaring only')
    if not isinstance(inputs[0],sp.Equality) or not isinstance(expected,sp.Equality):
        raise ValueError('Equations required')
    conditions=set()
    roots=0
    def inspect(expr, allow_roots):
        nonlocal roots
        for node in sp.preorder_traversal(expr):
            if isinstance(node,(sp.Symbol,sp.Rational,sp.Add,sp.Mul,sp.Equality)):
                continue
            if isinstance(node,sp.Pow):
                if isinstance(node.exp,sp.Integer):
                    if node.exp<0: conditions.add(sp.Ne(node.base,0,evaluate=False))
                    continue
                if allow_roots and node.exp==sp.Rational(1,2):
                    # Radicands must themselves be rational; nested branches
                    # and general real powers are outside this extension.
                    inspect(node.base,False)
                    conditions.add(sp.Ge(node.base,0,evaluate=False))
                    roots+=1
                    continue
            raise ValueError('Unsupported expression in real-root squaring')
    inspect(inputs[0],True)
    inspect(expected,False)
    if not roots:
        return replay_rational_step(rule,inputs,feeds,expected)
    actual=sp.Eq(inputs[0].lhs**2,inputs[0].rhs**2,evaluate=False)
    inspect(actual,False)
    if sp.cancel(actual.lhs-expected.lhs)!=0 or sp.cancel(actual.rhs-expected.rhs)!=0:
        raise ValueError('Squared equation differs from source conclusion')
    if any(sp.simplify(c)==sp.false for c in conditions):
        raise ValueError('Impossible real-root domain condition')
    return actual,tuple(sorted(conditions,key=sp.srepr))
