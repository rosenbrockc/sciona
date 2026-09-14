#!/usr/bin/env python3
"""Materialize dimensions only after recomputing pinned source correspondence."""
import argparse
import json
import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from sciona.physics_ingest.pdg_dimensions import materialize_pdg_dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    url = os.environ.get('SCIONA_DATA_CATALOG_DATABASE_URL')
    if not url:
        parser.error('Set SCIONA_DATA_CATALOG_DATABASE_URL in the runtime environment')
    with psycopg.connect(url, row_factory=dict_row) as db:
        result = materialize_pdg_dimensions(db, args.symbol_file.read_bytes())
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'counts': result}, indent=2))


if __name__ == '__main__':
    main()
