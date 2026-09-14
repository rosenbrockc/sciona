"""Attach pinned source comparisons without changing symbolic or review states."""
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_symbols import (
    load_pinned_pdg_scalars, inspect_pdg_correspondence, map_pdg_scalar_names,
)
from sciona.physics_ingest.source_symbolic import inspect_source_symbolic


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_pdg_evidence(row, symbol_bytes):
    pin = (row['snapshot_payload'].get('core_file_sha256') or {}).get('conversion_of_data_formats/symbols.cypher')
    definitions = load_pinned_pdg_scalars(symbol_bytes, pin)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        upstream = inspect_source_symbolic(row['source_payload'], row['sympy_srepr'] or '')
        comparison = None
        dimensions = []
        if upstream['status'] == 'roundtrip_passed':
            comparison = inspect_pdg_correspondence(upstream['sympy_srepr'], row['sympy_srepr'] or '', definitions)
            if comparison['correspondence'] == 'exact_ast_match':
                expression = deserialize_expr(upstream['sympy_srepr'])
                mapping, unresolved = map_pdg_scalar_names(expression, definitions)
                if unresolved:
                    raise ValueError('source correspondence changed during preparation')
                for source, target in sorted(mapping.items(), key=lambda pair: str(pair[1])):
                    dimension = definitions[str(source)].dimension
                    if dimension is not None:
                        dimensions.append({
                            'symbol_name': str(target), 'source_symbol': str(source),
                            'dim_signature': dimension.to_compact(), 'dimension_source': 'source',
                        })
    root = Path(__file__).resolve().parents[1]
    paths = ['physics_ingest/pdg_evidence.py', 'physics_ingest/pdg_symbols.py',
             'physics_ingest/source_symbolic.py', 'ghost/symbolic.py', 'ghost/dimensions.py']
    return {
        'runner_version': 'pdg-source-evidence.v1',
        'scope': 'pinned upstream syntax, scalar correspondence, and dimensional checks; no approval',
        'expression_version_id': str(row['version_id']),
        'candidate_id': str(row['candidate_id']),
        'snapshot_id': str(row['snapshot_id']),
        'symbol_file_sha256': pin,
        'candidate_payload_sha256': _digest(row['source_payload']),
        'stored_expression_sha256': _digest(row['sympy_srepr']),
        'implementation_hashes': {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        'upstream_symbolic': upstream,
        'source_comparison': comparison,
        # No role is inferred from an unoriented equation. These source-backed
        # dimensions remain evidence until variable roles and IO are resolved.
        'source_variable_dimensions': dimensions,
    }


def attach_pdg_source_evidence(connection, symbol_bytes):
    """Lock source and destination rows; apply an all-or-nothing evidence batch."""
    counts = Counter()
    with connection.transaction():
        rows = connection.execute(
            "SELECT e.expression_id,e.version_id,e.candidate_id,e.sympy_srepr,e.evidence_json,"
            "q.source_payload,q.snapshot_id,s.payload AS snapshot_payload "
            "FROM artifact_symbolic_expressions e "
            "JOIN physics_equation_candidates q USING(candidate_id) "
            "JOIN physics_ingest_snapshots s USING(snapshot_id) "
            "WHERE e.review_status='needs_human' AND s.payload ? 'core_file_sha256' "
            "ORDER BY e.expression_id FOR UPDATE OF e FOR SHARE OF q,s"
        ).fetchall()
        for row in rows:
            fresh = prepare_pdg_evidence(row, symbol_bytes)
            evidence = dict(row['evidence_json'] or {})
            existing = evidence.get('pdg_source_comparison')
            if existing is not None and existing != fresh:
                raise ValueError('existing source evidence differs; preserve it for explicit reprocessing')
            if existing == fresh:
                counts['unchanged'] += 1
                continue
            evidence['pdg_source_comparison'] = fresh
            connection.execute(
                'UPDATE artifact_symbolic_expressions SET evidence_json=%s WHERE expression_id=%s',
                (Jsonb(evidence), row['expression_id']),
            )
            counts['attached'] += 1
            comparison = fresh['source_comparison'] or {}
            if comparison.get('correspondence') == 'exact_ast_match':
                counts['exact_ast_match'] += 1
                if comparison.get('dimension_status') == 'checker_passed':
                    counts['exact_match_and_dimension_check_passed'] += 1
    return dict(counts)
