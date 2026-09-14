"""Reconcile intake parse confidence using freshly checked source correspondence."""
from collections import Counter
import hashlib
from pathlib import Path
from psycopg.types.json import Jsonb

from sciona.physics_ingest.normalization import _parse_confidence
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, _digest


def prepare_parse_corroboration(row, symbol_bytes):
    if row['review_status'] != 'needs_human' or row['parse_status'] != 'normalized':
        raise ValueError('corroboration requires a normalized review-pending expression')
    evidence = dict(row['evidence_json'] or {})
    if evidence.get('parse_roundtrip', {}).get('status') != 'passed':
        raise ValueError('stored parse roundtrip must pass')
    # Recompute against locked source records and hash-verified scalar bytes.
    # A stale attached report or edited JSON success flag cannot raise confidence.
    fresh = prepare_pdg_evidence(row, symbol_bytes)
    if fresh != evidence.get('pdg_source_comparison'):
        raise ValueError('source comparison is missing or stale')
    if (fresh.get('source_comparison') or {}).get('correspondence') != 'exact_ast_match':
        raise ValueError('source-defined mapping must exactly match the stored AST')
    score = _parse_confidence({}, parsed=True, diagnostics=[])
    if row['parse_confidence'] >= score:
        return None
    if 'parse_corroboration' in evidence:
        raise ValueError('previous corroboration requires explicit reconciliation')
    evidence['parse_corroboration'] = {
        'runner_version': 'parse-corroboration.v1',
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_parse_confidence': row['parse_confidence'],
        'parse_confidence': score,
        'source_comparison_sha256': _digest(fresh),
        'scope': 'parser confidence only; dimensions, physical validity, and human review remain separate gates',
        'basis': 'strict local parse roundtrip and exact source-defined upstream AST correspondence',
        'score_policy': 'existing normalizer successful-parse score; not a calibrated probability',
    }
    return {'parse_confidence': score, 'evidence_json': evidence}


def corroborate_pending_parses(connection, symbol_bytes):
    counts = Counter()
    with connection.transaction():
        rows = connection.execute(
            "SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload "
            "FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) "
            "JOIN physics_ingest_snapshots s USING(snapshot_id) "
            "WHERE e.review_status='needs_human' AND e.parse_status='normalized' AND "
            "e.evidence_json->'pdg_source_comparison'->'source_comparison'->>'correspondence'='exact_ast_match' "
            "ORDER BY e.expression_id FOR UPDATE OF e FOR SHARE OF q,s"
        ).fetchall()
        for row in rows:
            update = prepare_parse_corroboration(row, symbol_bytes)
            if update is None:
                counts['unchanged'] += 1
                continue
            connection.execute(
                'UPDATE artifact_symbolic_expressions SET parse_confidence=%s,evidence_json=%s WHERE expression_id=%s',
                (update['parse_confidence'], Jsonb(update['evidence_json']), row['expression_id']),
            )
            counts['corroborated'] += 1
    return dict(counts)
