"""Pinned inference arities and conservative rational-algebra step replay."""
from dataclasses import dataclass
import hashlib
import re


@dataclass(frozen=True)
class RuleContract:
    name: str
    inputs: int
    feeds: int
    outputs: int


def load_pinned_rule_contracts(content: bytes, expected_sha256: str):
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('inference rule file does not match its ingestion pin')
    result = {}
    for block in re.split(r'(?m)^UNWIND ', content.decode())[1:]:
        match = re.match(r'^\[\{id:"(\d+)",\s*properties:\{(.*?)\}\}\] AS row\s*CREATE \(n:inference_rule', block, re.S)
        if not match:
            continue
        identity, properties = match.groups()
        fields = dict(re.findall(r'(?<!\w)(name_latex|number_of_inputs|number_of_feeds|number_of_outputs):\s*("(?:\\.|[^"\\])*"|\d+)', properties))
        if len(fields) != 4 or identity in result:
            raise ValueError('incomplete or duplicate inference rule')
        result[identity] = RuleContract(fields['name_latex'][1:-1], *(int(fields['number_of_'+key]) for key in ('inputs','feeds','outputs')))
    if not result:
        raise ValueError('no supported inference rules found')
    return result


def contract_blockers(signature, rule):
    blockers = []
    if not isinstance(signature, dict):
        return ['invalid_signature']
    inputs = signature.get('inputs', [])
    outputs = signature.get('outputs', [signature['output']] if signature.get('output') else [])
    bindings = signature.get('variable_bindings') or {}
    if not isinstance(bindings, dict):
        return ['invalid_variable_bindings']
    feeds = bindings.get('feeds', [])
    for key, values, expected in [('inputs',inputs,rule.inputs),('feeds',feeds,rule.feeds),('outputs',outputs,rule.outputs)]:
        if not isinstance(values, list) or len(values) != expected:
            blockers.append(key+'_arity_mismatch')
    return blockers


def replay_rational_step(rule: RuleContract, inputs, feeds, expected):
    """Return a computed equation and necessary algebraic domain conditions.

    Equality under these conditions is not verification of the physical premises.
    Unsupported functions and non-rational powers are rejected, never guessed.
    """
    import sympy as sp
    if len(inputs) != rule.inputs or len(feeds) != rule.feeds or rule.outputs != 1:
        raise ValueError('inference arity mismatch')
    if not all(isinstance(x, sp.Equality) for x in [*inputs,expected]):
        raise ValueError('equations are required')
    conditions = set()

    def inspect(expr):
        for node in sp.preorder_traversal(expr):
            if isinstance(node, (sp.Symbol,sp.Rational,sp.Add,sp.Mul,sp.Equality)):
                continue
            if isinstance(node,sp.Pow) and isinstance(node.exp,sp.Integer):
                if node.exp < 0:
                    conditions.add(sp.Ne(node.base,0,evaluate=False))
                continue
            raise ValueError('unsupported rational-algebra expression')

    for expression in [*inputs,*feeds,expected]:
        inspect(expression)
    if rule.name == 'change two variables in expr':
        if (rule.inputs,rule.feeds)!=(1,4) or not all(isinstance(x,sp.Symbol) for x in feeds) or feeds[0]==feeds[2]:
            raise ValueError('invalid variable-renaming contract')
        with sp.evaluate(False):
            actual=inputs[0].xreplace({feeds[0]:feeds[1],feeds[2]:feeds[3]})
    elif rule.name == 'substitute LHS of three expressions into expr':
        if (rule.inputs,rule.feeds)!=(4,0) or len({x.lhs for x in inputs[:3]})!=3:
            raise ValueError('invalid simultaneous-substitution contract')
        replacements={x.lhs:x.rhs for x in inputs[:3]}
        # Structural replacement is simultaneous, preserving independent premises.
        with sp.evaluate(False):
            actual=inputs[3].xreplace(replacements)
    elif rule.name == 'divide both sides by':
        if (rule.inputs,rule.feeds)!=(1,1) or feeds[0]==0:
            raise ValueError('invalid division contract')
        conditions.add(sp.Ne(feeds[0],0,evaluate=False))
        actual=sp.Eq(inputs[0].lhs/feeds[0],inputs[0].rhs/feeds[0],evaluate=False)
    else:
        raise ValueError('unsupported inference rule')
    inspect(actual)
    if sp.cancel(actual.lhs-expected.lhs)!=0 or sp.cancel(actual.rhs-expected.rhs)!=0:
        raise ValueError('replayed equation differs from the source conclusion')
    return actual, tuple(sorted(conditions,key=sp.srepr))
