"""Versioned rational inference extensions; no branch selection or implicit inversion."""
import sympy as sp
from sciona.physics_ingest.pdg_rule_contracts import replay_rational_step as replay_v1


def replay_rational_step(rule,inputs,feeds,expected):
    if rule.name in {'change two variables in expr','substitute LHS of three expressions into expr','divide both sides by'}:
        return replay_v1(rule,inputs,feeds,expected)
    if len(inputs)!=rule.inputs or len(feeds)!=rule.feeds or rule.outputs!=1:
        raise ValueError('inference arity mismatch')
    if not all(isinstance(e,sp.Equality) for e in [*inputs,expected]):raise ValueError('equations required')
    conditions=set()
    def inspect(expression):
        for node in sp.preorder_traversal(expression):
            if isinstance(node,(sp.Symbol,sp.Rational,sp.Add,sp.Mul,sp.Equality)):continue
            if isinstance(node,sp.Pow) and isinstance(node.exp,sp.Integer):
                if node.exp<0:conditions.add(sp.Ne(node.base,0,evaluate=False))
                continue
            raise ValueError('unsupported rational expression or branch-sensitive operation')
    for expression in [*inputs,*feeds,expected]:inspect(expression)
    name=rule.name
    if name in {'multiply both sides by','add X to both sides','subtract X from both sides','raise both sides to power'}:
        if (rule.inputs,rule.feeds)!=(1,1):raise ValueError('invalid operation contract')
        factor=feeds[0];left,right=inputs[0].args
        if name=='multiply both sides by':left,right=left*factor,right*factor
        elif name=='add X to both sides':left,right=left+factor,right+factor
        elif name=='subtract X from both sides':left,right=left-factor,right-factor
        else:
            if not isinstance(factor,sp.Integer):raise ValueError('only integer powers supported')
            if factor<0:conditions.update([sp.Ne(left,0,evaluate=False),sp.Ne(right,0,evaluate=False)])
            left,right=left**factor,right**factor
        actual=sp.Eq(left,right,evaluate=False)
    elif name=='change variable X to Y':
        if (rule.inputs,rule.feeds)!=(1,2) or not isinstance(feeds[0],sp.Symbol):raise ValueError('invalid substitution contract')
        with sp.evaluate(False):actual=inputs[0].xreplace({feeds[0]:feeds[1]})
    elif name=='simplify':
        if (rule.inputs,rule.feeds)!=(1,0):raise ValueError('invalid simplify contract')
        actual=sp.Eq(sp.cancel(inputs[0].lhs),sp.cancel(inputs[0].rhs),evaluate=False)
    elif name=='LHS of expr 1 equals LHS of expr 2':
        if (rule.inputs,rule.feeds)!=(2,0) or sp.cancel(inputs[0].rhs-inputs[1].rhs)!=0:raise ValueError('right sides are not equal')
        actual=sp.Eq(inputs[0].lhs,inputs[1].lhs,evaluate=False)
    else:raise ValueError('unsupported inference rule')
    inspect(actual)
    if sp.cancel(actual.lhs-expected.lhs)!=0 or sp.cancel(actual.rhs-expected.rhs)!=0:raise ValueError('replayed equation differs from source conclusion')
    if any(sp.simplify(c)==sp.false for c in conditions):raise ValueError('impossible domain condition')
    return actual,tuple(sorted(conditions,key=sp.srepr))
