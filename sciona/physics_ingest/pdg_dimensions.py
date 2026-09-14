"""Materialize source-backed dimensions for exact-correspondence expressions."""
from collections import Counter
import hashlib
from pathlib import Path
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb

from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import deserialize_expr, SymbolicExpression
from sciona.ghost.symbolic_normalization import normalize_symbolic_candidate
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest


def prepare_pdg_dimensions(row, symbol_bytes):
    if row['review_status'] != 'needs_human' or row['parse_status'] != 'normalized':
        raise ValueError('dimensions require a normalized review-pending expression')
    fresh = prepare_pdg_evidence(row, symbol_bytes)
    if fresh != row['evidence_json'].get('pdg_source_comparison'):
        raise ValueError('source comparison is missing or stale')
    comparison = fresh.get('source_comparison') or {}
    if comparison.get('correspondence') != 'exact_ast_match' or comparison.get('dimension_status') != 'checker_passed':
        raise ValueError('exact source correspondence and dimensional check must pass')
    source_variables = {r['symbol_name']:r for r in fresh['source_variable_dimensions']}
    expression = deserialize_expr(row['sympy_srepr'])
    if set(source_variables) != {str(s) for s in expression.free_symbols} or not source_variables:
        raise ValueError('source dimensions must cover all free symbols')
    dimensions = {name:DimensionalSignature.from_compact(r['dim_signature']) for name,r in source_variables.items()}
    errors = SymbolicExpression(srepr_str=row['sympy_srepr'], dim_map=dimensions).check_dimensional_consistency()
    if errors:
        raise ValueError('stored-expression dimensional check did not pass')
    normalized = normalize_symbolic_candidate({
        'sympy_expr':expression,
        'variable_hints':{name:{'dim_signature':dimension} for name,dimension in dimensions.items()},
    }, require_dimensions=True)
    if normalized.srepr_str != row['sympy_srepr'] or normalized.review_tasks:
        raise ValueError('dimension normalization changed the expression or left unresolved tasks')
    evidence = {
        'runner_version':'pdg-dimensions.v1', 'status':'passed',
        'implementation_hashes':{
            path:hashlib.sha256((Path(__file__).resolve().parents[1]/path).read_bytes()).hexdigest()
            for path in ['physics_ingest/pdg_dimensions.py','ghost/symbolic_normalization.py']
        },
        'source_comparison_sha256':_digest(fresh),
        'dimensional_hash':normalized.dimensional_hash,
        'scope':'dimensional consistency using pinned source scalar claims; not physical or human approval',
        'role_policy':'existing symbolic normalizer: free symbols are expression-evaluation inputs, not a selected solve direction',
    }
    variables = []
    for ordinal,(name,variable) in enumerate(sorted(normalized.variables.items())):
        variables.append({
            'variable_id':str(uuid5(UUID(str(row['expression_id'])), name)),
            'expression_id':str(row['expression_id']), 'symbol_name':name,
            'source_symbol':source_variables[name]['source_symbol'],
            'variable_role':variable.role, 'dim_signature':dimensions[name].to_compact(),
            'dimension_source':'source', 'evidence_json':evidence, 'ordinal':ordinal,
        })
    return evidence,variables


def materialize_pdg_dimensions(connection,symbol_bytes):
    counts=Counter()
    with connection.transaction():
        rows=connection.execute(
            "SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload "
            "FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) "
            "JOIN physics_ingest_snapshots s USING(snapshot_id) "
            "WHERE e.review_status='needs_human' AND e.parse_status='normalized' AND "
            "e.evidence_json->'pdg_source_comparison'->'source_comparison'->>'correspondence'='exact_ast_match' AND "
            "e.evidence_json->'pdg_source_comparison'->'source_comparison'->>'dimension_status'='checker_passed' "
            "ORDER BY e.expression_id FOR UPDATE OF e FOR SHARE OF q,s"
        ).fetchall()
        for row in rows:
            fresh,variables=prepare_pdg_dimensions(row,symbol_bytes)
            existing=connection.execute(
                'SELECT variable_id,expression_id,symbol_name,source_symbol,variable_role,dim_signature,dimension_source,evidence_json,ordinal FROM artifact_symbolic_variables WHERE expression_id=%s ORDER BY ordinal FOR UPDATE',
                (row['expression_id'],),
            ).fetchall()
            for r in existing:
                r['variable_id']=str(r['variable_id']);r['expression_id']=str(r['expression_id'])
            evidence=dict(row['evidence_json'])
            prior=evidence.get('dimensional_analysis')
            if prior==fresh and existing==variables and row['dimensional_hash']==fresh['dimensional_hash']:
                counts['unchanged']+=1
                continue
            if prior is not None or existing or row['dimensional_hash']:
                raise ValueError('existing dimensional records conflict; batch rolled back')
            for variable in variables:
                connection.execute(
                    'INSERT INTO artifact_symbolic_variables(variable_id,expression_id,symbol_name,source_symbol,variable_role,dim_signature,dimension_source,evidence_json,ordinal) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                    tuple(variable[k] for k in ['variable_id','expression_id','symbol_name','source_symbol','variable_role','dim_signature','dimension_source'])+(Jsonb(variable['evidence_json']),variable['ordinal']),
                )
            evidence['dimensional_analysis']=fresh
            connection.execute('UPDATE artifact_symbolic_expressions SET dimensional_hash=%s,evidence_json=%s WHERE expression_id=%s',
                               (fresh['dimensional_hash'],Jsonb(evidence),row['expression_id']))
            counts['expressions']+=1
            counts['variables']+=len(variables)
    return dict(counts)
