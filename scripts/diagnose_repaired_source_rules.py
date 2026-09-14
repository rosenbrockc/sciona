#!/usr/bin/env python3
"""Read-only local rule diagnostics; never substitute for composed proof replay."""
import argparse
import hashlib
import json
from pathlib import Path

import sympy as sp

from sciona.ghost.symbolic import deserialize_expr, serialize_expr
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts
from sciona.physics_ingest.pdg_rational_rules_v4 import replay_rational_step
from sciona.physics_ingest.source_symbolic import parse_source_srepr


def diagnose(selection, rule_bytes):
    rules = load_pinned_rule_contracts(rule_bytes, selection['rule_file_sha256'])
    expressions = {k: deserialize_expr(v['source_srepr']) for k, v in selection['selected_expressions'].items()}
    records = []
    for node in sorted(selection['nodes'], key=lambda n: int(n['node_id'].rsplit('_', 1)[1])):
        signature = node['type_signature']
        rule = rules[signature['inference_rule_id']]
        inputs = [expressions[k] for k in signature['inputs']]
        expected = expressions[signature['output']]
        feeds = [parse_source_srepr(f['sympy']) for f in signature['variable_bindings']['feeds']]
        record = dict(node_id=node['node_id'], rule=rule.name)
        try:
            _, conditions = replay_rational_step(rule, inputs, feeds, expected)
            record.update(local_replay_passed=True, required_conditions=[serialize_expr(c) for c in conditions])
        except ValueError as error:
            record.update(local_replay_passed=False, blocker=str(error))
            if rule.name.startswith('substitute ') and len(inputs) == 2:
                premise, target = inputs
                matches = {}
                for label, mapping in [('insert_lhs', {premise.rhs: premise.lhs}), ('insert_rhs', {premise.lhs: premise.rhs})]:
                    with sp.evaluate(False):
                        actual = target.xreplace(mapping)
                    matches[label] = all(sp.cancel(a-b) == 0 for a, b in zip(actual.args, expected.args))
                record['structural_direction_matches'] = matches
            if rule.name == 'simplify':
                # A common factor is a proposed explicit division, not a passed
                # simplification: the zero-factor case must be excluded first.
                ratios = [sp.cancel(a/b) for a, b in zip(inputs[0].args, expected.args) if b != 0]
                if len(ratios) == 2 and sp.cancel(ratios[0]-ratios[1]) == 0:
                    record['candidate_common_factor'] = serialize_expr(ratios[0])
                    record['additional_division_condition'] = serialize_expr(sp.Ne(ratios[0], 0, evaluate=False))
        records.append(record)
    return dict(read_only=True, approval_applied=False, source_graph_version_id=selection['source_graph_version_id'],
                composed_replay_passed=selection['replay_passed'], steps=records,
                scope='Independent local diagnostics against stored intermediate equations; no composed proof, source correction or approval.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['selection', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(json.loads(args.selection.read_text()), args.rule_file.read_bytes())
    result['selection_sha256'] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    result['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([r for r in result['steps'] if not r['local_replay_passed']], indent=2))
