#!/usr/bin/env python3
"""Preserve verified failing source-step evidence without changing publication."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from scripts.audit_physics_branch_steps import validate
from scripts.import_residual_execution_drafts import ensure_row
from sciona.physics_ingest.pdg_evidence import _digest


def record(root,symbol_file,rule_file,apply=False):
    report=validate(root,symbol_file,rule_file)
    if report!=json.loads((root/'docs/reviews/physics_branch_step_diagnostics.json').read_text()):
        raise ValueError('Retained diagnostic evidence differs')
    created=0
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('pdg-branch-diagnostics.v1'))")
        for graph in report['graphs']:
            failures=[s for s in graph['steps'] if s.get('counterexample')]
            if len(failures)!=4 or any(not f['counterexample']['premise_satisfied'] or f['counterexample']['conclusion_satisfied'] for f in failures):
                raise ValueError('Expected verified counterexamples required')
            row=db.execute('SELECT a.artifact_id,a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v',(graph['version_id'],)).fetchone()
            if not row or row['content_hash']!=graph['content_hash'] or row['status']!='draft' or row['is_publishable']:
                raise ValueError('Source target no longer exact unapproved draft')
            prior=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s FOR SHARE',(graph['prior_failed_evidence_id'],)).fetchone()
            if not prior or _digest(prior['details'])!=graph['prior_evidence_sha256']:raise ValueError('Prior evidence changed')
            details=dict(graph=graph,implementation_sha256=report['implementation_sha256'],
                recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Four exact arithmetic counterexamples in pinned source graph. No repair or publication approval.')
            identity=uuid5(UUID(graph['version_id']),'pdg-branch-diagnostics.v1:'+_digest(details))
            created+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':identity},dict(evidence_id=identity,
                artifact_id=row['artifact_id'],version_id=UUID(graph['version_id']),audit_type='determinism_replay',
                passed=False,status='completed',source_kind='automated',runner_version='pdg-branch-diagnostics.v1',details=Jsonb(details)))
        if not apply:db.rollback()
    return dict(applied=apply,evidence_created=created,verified_counterexamples=4,publication='unchanged_draft')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file']:parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    print(json.dumps(record(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.apply)))
