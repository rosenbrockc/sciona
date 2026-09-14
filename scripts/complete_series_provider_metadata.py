#!/usr/bin/env python3
"""Attach versioned IO and previously obtained local wheel-validation evidence."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import zipfile
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.physics_ingest.series_execution import PRIMITIVE
from sciona.physics_ingest.pdg_evidence import _digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',required=True,type=Path)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    report=json.loads(args.validation.read_text())
    if hashlib.sha256(args.wheel.read_bytes()).hexdigest()!=report['wheel_sha256']:raise ValueError('wheel evidence hash mismatch')
    with zipfile.ZipFile(args.wheel) as archive:source=archive.read('sciona/atoms/electrical/series_resistance.py')
    if source!=Path(inspect.getfile(series_resistance)).read_bytes() or hashlib.sha256(source).hexdigest()!=report['source_sha256']:raise ValueError('wheel source differs from validated provider')
    if any(report.get(key) is not True for key in ['wheel_import_confirmed','sympy_imports_blocked','domain_guard_verified']) or report.get('synthetic_cases')!=256:raise ValueError('local wheel validation incomplete')
    report={**report,'scope':'local wheel installation and behavior only; not release availability or license approval','release_available':False}
    counts={'io_rows_created':0,'evidence_created':0}
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        target=db.execute("SELECT a.artifact_id,v.version_id FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND a.status='draft' AND v.is_latest FOR UPDATE OF a,v",(PRIMITIVE,)).fetchone()
        if not target:raise ValueError('exact draft provider version unavailable')
        intake=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='provider-draft-intake.v1'",(target['version_id'],)).fetchall()
        if len(intake)!=1 or intake[0]['details']['source_file_sha256']!=report['source_sha256']:raise ValueError('catalog provider source hash mismatch')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE runner_version='series-cdg-execution.v1' AND passed").fetchall()
        graphs=[r['details']['execution_cdg'] for r in evidence if r['details'].get('provider_implementation_sha256')==report['source_sha256']]
        if len(graphs)!=1:raise ValueError('unique validated execution interface required')
        node=graphs[0]['nodes'][0]
        if node['matched_primitive']!=PRIMITIVE or [p['name'] for p in node['inputs']]!=list(inspect.signature(series_resistance).parameters):raise ValueError('validated ports do not match the callable')
        for direction,ports in [('input',node['inputs']),('output',node['outputs'])]:
            for ordinal,port in enumerate(ports):
                common={k:port[k] for k in ['name','type_desc','constraints','required','default_value_repr']}
                common.update(version_id=target['version_id'],direction=direction,ordinal=ordinal)
                for table,owner_key in [('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id')]:
                    row={**common,owner_key:target['artifact_id']}
                    if table=='artifact_io_specs':row['dim_signature']=port['dim_signature']
                    existing=db.execute(sql.SQL('SELECT {} FROM {} WHERE version_id=%s AND direction=%s AND name=%s').format(sql.SQL(',').join(map(sql.Identifier,row)),sql.Identifier(table)),(target['version_id'],direction,port['name'])).fetchall()
                    if existing:
                        if existing!=[row]:raise ValueError('existing provider IO conflicts')
                        continue
                    db.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,row)),sql.SQL(',').join(sql.Placeholder() for _ in row)),tuple(row.values()))
                    counts['io_rows_created']+=1
        for table in ['atom_io_specs','artifact_io_specs']:
            total=db.execute(sql.SQL('SELECT count(*) AS n FROM {} WHERE version_id=%s').format(sql.Identifier(table)),(target['version_id'],)).fetchone()['n']
            if total!=len(node['inputs'])+len(node['outputs']):raise ValueError('unexpected extra provider IO rows')
        evidence_id=uuid5(UUID(str(target['version_id'])),'wheel-validation:'+_digest(report))
        existing=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s',(evidence_id,)).fetchone()
        if existing:
            if existing['details']!=report:raise ValueError('wheel evidence drift')
        else:
            db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,details,source_kind,runner_version) VALUES(%s,%s,%s,'smoke_test',true,'completed',%s,'automated','provider-local-wheel.v1')",(evidence_id,target['artifact_id'],target['version_id'],Jsonb(report)))
            counts['evidence_created']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':counts},indent=2))


if __name__=='__main__':main()
