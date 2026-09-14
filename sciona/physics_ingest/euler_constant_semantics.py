"""Explicit reviewed interpretation of exact PDG identities in the Euler proof.

This is a semantic interpretation, not literal source-AST parity. The source
calls its imaginary-unit identity a variable; that discrepancy is preserved.
No lexical replacement of an arbitrary symbol named i or pi is performed.
"""
import hashlib
import re
import sympy as sp
from sciona.physics_ingest.pdg_symbols import _ROW,_PROPERTY

REVIEWED={
    '0003141':dict(latex=r'\pi',name_latex='pi',variable_or_constant='constant',scope='real'),
    '0004621':dict(latex='i',name_latex='imaginary unit',variable_or_constant='variable',scope='imaginary'),
    '0001464':dict(latex='x',name_latex=None,variable_or_constant='variable',scope='real'),
}


def reviewed_mapping(content,expected_sha256):
    if hashlib.sha256(content).hexdigest()!=expected_sha256:raise ValueError('Source symbol pin differs')
    selected={}
    for block in re.split(r'(?m)^UNWIND ',content.decode())[1:]:
        row=_ROW.match(block)
        if not row or row.group(1) not in REVIEWED:continue
        identity,body=row.groups()
        if identity in selected:raise ValueError('Duplicate reviewed identity')
        properties={}
        for item in _PROPERTY.finditer(body):
            key,value=item.groups()
            if key in properties:raise ValueError('Duplicate source property')
            properties[key]=value[1:-1] if value.startswith('"') else value
        for key,wanted in REVIEWED[identity].items():
            if wanted is not None and properties.get(key)!=wanted:raise ValueError('Reviewed symbol declaration differs: '+identity+' '+key)
        dimensions=[value for key,value in properties.items() if key.startswith('dimension_')]
        if len(dimensions)!=7 or any(v!='0' for v in dimensions):raise ValueError('Dimensionless declarations required')
        selected[identity]={k:properties[k] for k in ['latex','name_latex','variable_or_constant','scope']}
    if set(selected)!=set(REVIEWED):raise ValueError('Missing reviewed constants')
    return {sp.Symbol('pdg0003141'):sp.pi,sp.Symbol('pdg0004621'):sp.I,
        sp.Symbol('pdg0001464'):sp.Symbol('theta',real=True)},dict(
        source_declarations=selected,interpretation='Exact pi and imaginary unit I, with real dimensionless theta.',
        source_discrepancies=['Imaginary-unit identity is marked variable in source; this reviewed interpretation fixes it to I.'],
        scope='Exact identity-specific reviewed semantics, not a general symbol-name coercion.')


def verify_euler_steps(nodes,expressions,rules,mapping):
    import json
    from sciona.physics_ingest.source_symbolic import parse_source_srepr
    from sciona.physics_ingest.pdg_rule_contracts import contract_blockers
    from sciona.ghost.symbolic import serialize_expr
    signatures={n['node_id']:json.loads(n['type_signature']) if isinstance(n['type_signature'],str) else n['type_signature'] for n in nodes}
    if len(signatures)!=len(nodes):raise ValueError('Duplicate step identity')
    producers={}
    for identity,s in signatures.items():
        outputs=s.get('outputs',[s['output']])
        if len(outputs)!=1 or outputs[0] in producers:raise ValueError('Unique single output required')
        producers[outputs[0]]=identity
    needed={i for s in signatures.values() for i in s['inputs']}
    def interpret(expr):
        if expr.free_symbols-set(mapping):raise ValueError('Unreviewed source symbol')
        # Reconstruct equality without automatic conversion to a boolean.
        if not isinstance(expr,sp.Equality):raise ValueError('Source equation required')
        return sp.Eq(expr.lhs.xreplace(mapping),expr.rhs.xreplace(mapping),evaluate=False)
    computed={i:interpret(expressions[i]) for i in needed-set(producers)}
    for premise in computed.values():
        if sp.simplify(sp.expand_complex(premise.lhs-premise.rhs))!=0:raise ValueError('Root identity is not established')
    pending=dict(signatures);records=[]
    while pending:
        ready=[i for i,s in pending.items() if all(p in computed for p in s['inputs'])]
        if not ready:raise ValueError('Unresolved or cyclic proof')
        for identity in sorted(ready):
            s=pending.pop(identity);rule=rules[s['inference_rule_id']]
            if contract_blockers(s,rule):raise ValueError('Source rule arity differs')
            if len(s['inputs'])!=1:raise ValueError('Single premise required')
            premise=computed[s['inputs'][0]]
            raw_feeds=[parse_source_srepr(f['sympy']) for f in s['variable_bindings']['feeds']]
            if any(f.free_symbols-set(mapping) for f in raw_feeds):raise ValueError('Unreviewed feed symbol')
            feeds=[f.xreplace(mapping) for f in raw_feeds]
            if rule.name=='change variable X to Y':
                actual=sp.Eq(premise.lhs.xreplace({feeds[0]:feeds[1]}),premise.rhs.xreplace({feeds[0]:feeds[1]}),evaluate=False)
            elif rule.name=='simplify':actual=sp.Eq(sp.simplify(premise.lhs),sp.simplify(premise.rhs),evaluate=False)
            elif rule.name=='add X to both sides':actual=sp.Eq(premise.lhs+feeds[0],premise.rhs+feeds[0],evaluate=False)
            else:raise ValueError('Unsupported Euler step')
            output=s['output'];expected=interpret(expressions[output])
            if any(sp.simplify(a-b)!=0 for a,b in zip(actual.args,expected.args)):raise ValueError('Interpreted source conclusion differs')
            computed[output]=actual
            records.append(dict(node_id=identity,output_expression_id=output,computed_srepr=serialize_expr(actual)))
    return dict(steps=records,root_identity_verified=True,terminal_expression_ids=sorted(set(producers)-needed),
        literal_source_parity=False,scope='Source proof under explicit reviewed constant interpretation, not publication approval.')
