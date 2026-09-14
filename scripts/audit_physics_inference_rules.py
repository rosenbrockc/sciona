#!/usr/bin/env python3
"""Read-only pinned-rule arity audit and source-algebra replay for ready graphs."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from sciona.physics_ingest.pdg_rule_contracts import load_pinned_rule_contracts, contract_blockers, replay_rational_step
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.ghost.symbolic import deserialize_expr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule-file',required=True,type=Path)
    args=parser.parse_args()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on') as db:
        pins=db.execute("SELECT DISTINCT s.payload->'core_file_sha256'->>'conversion_of_data_formats/infrules.cypher' AS pin FROM physics_ingest_snapshots s JOIN physics_equation_candidates q USING(snapshot_id) JOIN artifact_symbolic_expressions e USING(candidate_id) WHERE s.payload ? 'core_file_sha256'").fetchall()
        if len(pins)!=1:raise ValueError('multiple source rule versions require separate audits')
        rules=load_pinned_rule_contracts(args.rule_file.read_bytes(),pins[0]['pin'])
        nodes=db.execute("SELECT n.* FROM artifact_cdg_nodes n JOIN artifact_versions v USING(version_id) JOIN artifacts a USING(artifact_id) WHERE a.fqdn LIKE 'physics.pdg.%' AND v.is_latest").fetchall()
        ready=db.execute("""SELECT v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_cdg_bindings b USING(version_id)
 LEFT JOIN artifacts t ON t.fqdn=b.bound_artifact_fqdn LEFT JOIN artifact_versions tv ON tv.artifact_id=t.artifact_id AND tv.content_hash=b.bound_version_content_hash LEFT JOIN artifact_symbolic_expressions e ON e.version_id=tv.version_id
 WHERE a.fqdn LIKE 'physics.pdg.%' AND v.is_latest GROUP BY v.version_id HAVING count(*)=count(*) FILTER(WHERE e.evidence_json->'dimensional_analysis'->>'status'='passed')""").fetchall()
        expressions=db.execute("SELECT expression_id,evidence_json FROM artifact_symbolic_expressions WHERE evidence_json->'pdg_source_comparison'->'upstream_symbolic'->>'status'='roundtrip_passed'").fetchall()
    by_id={str(r['expression_id']):r['evidence_json']['pdg_source_comparison']['upstream_symbolic']['sympy_srepr'] for r in expressions}
    ready_ids={r['version_id'] for r in ready}
    counts=Counter();graphs=defaultdict(lambda:Counter())
    signatures=defaultdict(list);grouped_failures=Counter()
    for node in nodes:
        g=graphs[node['version_id']];g['nodes']+=1
        signature=json.loads(node['type_signature'])
        signatures[node['version_id']].append(signature)
        rule=rules.get(signature.get('inference_rule_id'))
        blockers=['unknown_rule'] if rule is None else contract_blockers(signature,rule)
        if blockers:
            if signature.get('source_pdg_step_id'):
                grouped_failures[(rule.name if rule else 'unknown_rule',tuple(blockers))]+=1
            g['blocked_nodes']+=1;counts.update(blockers);continue
        counts['arity_passed_nodes']+=1
        if node['version_id'] not in ready_ids:continue
        try:
            inputs=[deserialize_expr(by_id[i]) for i in signature['inputs']]
            expected=deserialize_expr(by_id[signature['output']])
            feeds=[parse_source_srepr(f['sympy']) for f in signature['variable_bindings'].get('feeds',[])]
            actual,conditions=replay_rational_step(rule,inputs,feeds,expected)
        except (ValueError,KeyError,TypeError):
            g['replay_failed']+=1;continue
        g['replayed_nodes']+=1;g['necessary_conditions']+=len(conditions)
    projection_classes=Counter();families=defaultdict(list)
    for version,rows in signatures.items():
        grouped=all(row.get('source_pdg_step_id') for row in rows)
        valid=not graphs[version]['blocked_nodes']
        projection_classes[(grouped,valid)]+=1
        source_ids={row.get('variable_bindings',{}).get('derivation_id') for row in rows}
        if len(source_ids)==1 and None not in source_ids:
            families[next(iter(source_ids))].append((grouped,valid))
    print(json.dumps({
        'graph_count':len(graphs),'node_counts':dict(counts),
        'graphs_with_all_arities_valid':sum(not g['blocked_nodes'] for g in graphs.values()),
        'projection_classes':[
            {'explicit_source_steps':key[0],'all_arities_valid':key[1],'graphs':count}
            for key,count in sorted(projection_classes.items())
        ],
        'source_family_count':len(families),
        'source_family_sizes':dict(Counter(map(len,families.values()))),
        'paired_legacy_blocked_grouped_valid':sum(
            sorted(rows)==[(False,False),(True,True)] for rows in families.values()
        ),
        'grouped_contract_failures':[
            {'rule':key[0],'blockers':list(key[1]),'nodes':count}
            for key,count in sorted(grouped_failures.items())
        ],
        'dimension_ready_graph_replays':[
            dict(graphs[key]) for key in sorted(ready_ids,key=str)
        ],
        'scope':'source rational-algebra replay only; conditions and physical premises still require review',
    },indent=2))


if __name__=='__main__':main()
