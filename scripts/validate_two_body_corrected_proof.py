#!/usr/bin/env python3
"""Revalidate exact source alternatives and the separately corrected proof."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import sympy as sp

from scripts.validate_repaired_source_graph import validate
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_symbols import _ROW, _PROPERTY
from sciona.physics_ingest.two_body_corrected_proof import SOURCE_SYMBOL_NAMES, build_proof, symbols, verify_proof


def validate_source(selection, symbol_bytes):
    proof = build_proof()
    result = verify_proof(proof)
    mapped = {sp.Symbol(k): symbols()[v] for k, v in SOURCE_SYMBOL_NAMES.items()}
    expressions = {}
    for identity, record in selection['selected_expressions'].items():
        source = deserialize_expr(record['source_srepr'])
        if source.free_symbols-set(mapped):
            raise ValueError('Unreviewed source identity')
        expressions[identity] = sp.Eq(source.lhs.xreplace(mapped), source.rhs.xreplace(mapped), evaluate=False)
    nodes = sorted(selection['nodes'], key=lambda n: int(n['node_id'].rsplit('_', 1)[1]))
    if [n['node_id'] for n in nodes] != ['pdg_step_'+str(i) for i in range(1, 14)]:
        raise ValueError('Source step inventory differs')
    for node, corrected in zip(nodes, proof.steps):
        original = expressions[node['type_signature']['output']]
        if any(sp.cancel(a-b) != 0 for a, b in zip(original.args, corrected.args)):
            raise ValueError('Corrected equation differs from selected source output')
    needed = {k for n in nodes for k in n['type_signature']['inputs']}
    produced = {n['type_signature']['output'] for n in nodes}
    roots = [expressions[k] for k in sorted(needed-produced)]
    if len(roots) != 6 or any(sum(all(sp.cancel(a-b) == 0 for a, b in zip(root.args, premise.args)) for root in roots) != 1 for premise in proof.premises):
        raise ValueError('Source premise inventory differs')
    pi_declarations = []
    for block in re.split(r'(?m)^UNWIND ', symbol_bytes.decode())[1:]:
        row = _ROW.match(block)
        if row and row.group(1) == '0003141':
            props = {k: v[1:-1] if v.startswith('"') else v for k, v in _PROPERTY.findall(row.group(2))}
            pi_declarations.append(props)
    if len(pi_declarations) != 1:
        raise ValueError('Unique source pi declaration required')
    props = pi_declarations[0]
    if any(props.get(k) != v for k, v in dict(latex=r'\pi', name_latex='pi', variable_or_constant='constant', scope='real').items()):
        raise ValueError('Source pi meaning differs')
    dimensions = [v for k, v in props.items() if k.startswith('dimension_')]
    if len(dimensions) != 7 or any(v != '0' for v in dimensions):
        raise ValueError('Source pi must be dimensionless')
    p = symbols()['p']
    terminal = proof.steps[-1]
    result.update(selected_source_equations_matched=13, selected_source_premises_matched=6,
        equation_comparison_scope='Rational equation agreement under explicit positive-real identity mapping; original inference labels and rules are not certified.',
        source_symbol_mapping=SOURCE_SYMBOL_NAMES, pi_source_declaration=props,
        pi_interpretation='Separately reviewed specialization of positive p to exact mathematical pi, based on exact source identity pdg0003141.',
        physical_terminal=sp.srepr(terminal.xreplace({p: sp.pi})),
        physical_scope='Isolated Newtonian two point masses in a circular orbit; r is their separation and d1,d2 are positive barycentric distances. No orbit-validity classifier, relativistic correction, extended-body or perturbed-orbit claim.',
        reference='https://openstax.org/books/calculus-volume-3/pages/3-4-motion-in-space',
        reference_scope='Equation 3.30 supports the total-mass period formula. Circular separation semantics follow the explicit premises here.',
        symbol_file_sha256=hashlib.sha256(symbol_bytes).hexdigest(),
        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selection = validate(root, args.symbol_file, args.rule_file)
    result = validate_source(selection, args.symbol_file.read_bytes())
    files = ['scripts/validate_two_body_corrected_proof.py', 'sciona/physics_ingest/two_body_corrected_proof.py', 'tests/physics_ingest/test_two_body_corrected_proof.py']
    result['implementation_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
    result['selection_implementation_sha256'] = selection['implementation_sha256']
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['checks', 'selected_source_equations_matched', 'selected_source_premises_matched']}))
