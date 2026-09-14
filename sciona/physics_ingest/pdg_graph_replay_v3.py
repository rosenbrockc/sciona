"""Compose rational and forward real-root squaring steps with domain bounds."""
import json
from sciona.physics_ingest.pdg_rule_contracts import contract_blockers
from sciona.physics_ingest.pdg_real_power_rules import replay_real_power_step
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.ghost.symbolic import serialize_expr


def replay_derivation(nodes, expressions, rules, *, edges=None):
    import sympy as sp
    steps = {}
    producers = {}
    for node in nodes:
        identity = node['node_id']
        if identity in steps:
            raise ValueError('duplicate step identity')
        signature = node['type_signature']
        if isinstance(signature,str): signature=json.loads(signature)
        rule=rules.get(signature.get('inference_rule_id'))
        if rule is None or contract_blockers(signature,rule):
            raise ValueError('missing inference contract or invalid arity')
        outputs=signature.get('outputs',[signature.get('output')])
        if len(outputs)!=1 or not outputs[0] or (signature.get('output') and signature['output']!=outputs[0]):
            raise ValueError('one consistent output identity is required')
        output=outputs[0]
        if output in producers:
            raise ValueError('multiple producers for one expression')
        producers[output]=identity
        steps[identity]=(signature,rule,output)
    if not steps:
        raise ValueError('empty graph cannot be replayed')
    needed={i for signature,_,_ in steps.values() for i in signature['inputs']}
    if not needed.union(producers).issubset(expressions):
        raise ValueError('missing source expression')
    expected_edges={(producers[i],identity) for identity,(signature,_,_) in steps.items()
                    for i in signature['inputs'] if i in producers}
    if edges is not None and set(edges)!=expected_edges:
        raise ValueError('projected edges disagree with expression dependencies')
    roots=needed-set(producers)
    computed={identity:expressions[identity] for identity in roots}
    scopes={identity:set() for identity in roots}
    pending=dict(steps)
    records=[]
    all_conditions=set()
    while pending:
        ready=sorted(identity for identity,(signature,_,_) in pending.items()
                     if all(i in computed for i in signature['inputs']))
        if not ready:
            raise ValueError('cycle or unresolved computed dependency')
        for identity in ready:
            signature,rule,output=pending.pop(identity)
            feeds=[parse_source_srepr(f['sympy']) for f in signature['variable_bindings'].get('feeds',[])]
            actual,local=replay_real_power_step(rule,[computed[i] for i in signature['inputs']],feeds,expressions[output])
            inherited=set().union(*(scopes[i] for i in signature['inputs']))
            if rule.name=='change two variables in expr':
                mapping={feeds[0]:feeds[1],feeds[2]:feeds[3]}
            elif rule.name=='change variable X to Y':
                mapping={feeds[0]:feeds[1]}
            elif rule.name=='substitute LHS of three expressions into expr':
                mapping={computed[i].lhs:computed[i].rhs for i in signature['inputs'][:3]}
            else: mapping={}
            with sp.evaluate(False):
                transformed={condition.xreplace(mapping) for condition in inherited}
            conditions=inherited|transformed|set(local)
            if any(sp.simplify(condition)==sp.false for condition in conditions):
                raise ValueError('inference requires an impossible domain condition')
            computed[output]=actual
            scopes[output]=conditions
            all_conditions.update(conditions)
            records.append({'node_id':identity,'output_expression_id':output,
                            'computed_srepr':serialize_expr(actual),
                            'required_conditions':[serialize_expr(c) for c in sorted(conditions,key=sp.srepr)]})
    terminal_outputs=set(producers)-needed
    return {
        'runner_version':'pdg-graph-replay.v3',
        'scope':'composed source algebra including forward real-root squaring under explicit conditions; not physical or publication approval',
        'steps':records,
        'root_expression_ids':sorted(roots),
        'terminal_expression_ids':sorted(terminal_outputs),
        'required_conditions':[serialize_expr(c) for c in sorted(all_conditions,key=sp.srepr)],
    }
