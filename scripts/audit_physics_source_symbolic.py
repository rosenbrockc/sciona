#!/usr/bin/env python3
"""Read-only aggregate comparison of upstream and stored physics symbolic ASTs."""
import argparse
from pathlib import Path
import contextlib
from collections import Counter
import io
import json
import os

import psycopg
from psycopg.rows import dict_row

from sciona.physics_ingest.source_symbolic import inspect_source_symbolic
from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars, inspect_pdg_correspondence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file', type=Path, help='Local source scalar file; each snapshot pin must match')
    args = parser.parse_args()
    symbol_bytes = args.symbol_file.read_bytes() if args.symbol_file else None
    registries = {}
    url = os.environ.get('SCIONA_DATA_CATALOG_DATABASE_URL')
    if not url:
        raise SystemExit('Set SCIONA_DATA_CATALOG_DATABASE_URL in the runtime environment')
    with psycopg.connect(url, row_factory=dict_row, options='-c default_transaction_read_only=on') as db:
        rows = db.execute(
            "SELECT e.parse_status,e.sympy_srepr,q.source_payload,s.payload->'core_file_sha256' AS pins "
            'FROM artifact_symbolic_expressions e '
            'JOIN physics_equation_candidates q USING(candidate_id) '
            'JOIN physics_ingest_snapshots s USING(snapshot_id)'
        ).fetchall()
    counts = Counter()
    correspondence = Counter()
    for row in rows:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evidence = inspect_source_symbolic(row['source_payload'] or {}, row['sympy_srepr'] or '')
        if symbol_bytes is not None and evidence['status'] == 'roundtrip_passed':
            pin = (row['pins'] or {}).get('conversion_of_data_formats/symbols.cypher')
            if pin not in registries:
                registries[pin] = load_pinned_pdg_scalars(symbol_bytes, pin)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                comparison = inspect_pdg_correspondence(evidence['sympy_srepr'], row['sympy_srepr'] or '', registries[pin])
            correspondence[(row['parse_status'], comparison['mapping_status'], comparison['correspondence'], comparison['dimension_status'])] += 1
        counts[(row['parse_status'], evidence['status'], evidence.get('reason', evidence['correspondence']))] += 1
    print(json.dumps({'examined': len(rows), 'results': [
        {'stored_parse_status': key[0], 'source_status': key[1], 'result': key[2], 'count': count}
        for key, count in sorted(counts.items())
    ], 'pinned_source_comparison': [
        {'stored_parse_status': key[0], 'mapping_status': key[1], 'correspondence': key[2], 'dimension_status': key[3], 'count': count}
        for key, count in sorted(correspondence.items())
    ]}, indent=2))


if __name__ == '__main__':
    main()
