#!/usr/bin/env python3
"""Resolve intake targets using explicit catalog import metadata and callable identity."""
import argparse
from collections import Counter
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from uuid import uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.competition_import import _canonical
from sciona.competition_promotion import assess_keyword_call_contract
from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import _ensure_atoms_imported
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    _ensure_atoms_imported()
    totals=Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        records=db.execute("SELECT e.*,v.content_hash FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) WHERE e.runner_version='competition-intake.v1' ORDER BY e.evidence_id FOR SHARE OF e,v").fetchall()
        catalog={}
        for table,versions,key in [('atoms','atom_versions','atom_id')]:
            rows=db.execute(f'SELECT a.fqdn,a.import_module,a.source_symbol,v.content_hash FROM {table} a JOIN {versions} v ON a.{key}=v.{key} AND v.is_latest FOR SHARE OF a,v').fetchall()
            for row in rows:
                catalog.setdefault(row['fqdn'],[]).append(row)
        for record in records:
            snapshot=record['details']['snapshot']
            if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=record['content_hash']:
                raise ValueError('snapshot drift')
            stages={s['stage_id']:s for s in snapshot['template']['stages']}
            reports=[]
            for binding in snapshot['bindings']['bindings']:
                target=binding.get('bound_artifact_fqdn')
                if not target or binding.get('status')!='active':
                    continue
                result={'stage_id':binding['stage_id'],'catalog_fqdn':target}
                metadata=catalog.get(target,[])
                identities={(r['import_module'],r['source_symbol']) for r in metadata}
                if len(identities)!=1:
                    result['outcome']='missing_or_conflicting_catalog_identity'
                else:
                    module,symbol=next(iter(identities))
                    if not module or not symbol or not module.startswith('sciona.'):
                        result['outcome']='unsupported_import_metadata'
                    else:
                        try:
                            fn=getattr(importlib.import_module(module),symbol)
                        except (ImportError,AttributeError):
                            result['outcome']='catalog_import_unavailable'
                        else:
                            runtime=fn.__module__+'.'+fn.__name__
                            if runtime not in REGISTRY or REGISTRY[runtime]['impl'] is not fn:
                                result['outcome']='callable_not_registered'
                            else:
                                path=inspect.getsourcefile(inspect.unwrap(fn))
                                if not path:raise ValueError('callable lacks source anchor')
                                result.update(runtime_fqdn=runtime,import_module=module,source_symbol=symbol,source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),source_line=inspect.getsourcelines(inspect.unwrap(fn))[1],catalog_content_hashes=sorted({r['content_hash'] for r in metadata}))
                                contract=assess_keyword_call_contract([p['name'] for p in stages[binding['stage_id']].get('inputs',[])],fn)
                                result['call_contract']=contract
                                result['outcome']='identity_resolved'
                                totals['keyword_compatible' if contract['keyword_compatible'] else 'call_adapter_required']+=1
                totals[result['outcome']]+=1
                reports.append(result)
            if not reports:continue
            details={'scope':'Explicit catalog import metadata resolved to the identical registered callable; proposed runtime identity mapping only. Source hashes pin this observation, not approval of historical catalog versions or stage semantics.','source_content_hash':record['content_hash'],'stages':reports}
            digest=hashlib.sha256(_canonical(details).encode()).hexdigest()
            evidence_id=uuid5(record['version_id'],'competition-runtime-identity.v1:'+digest)
            totals['evidence_created']+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},{'evidence_id':evidence_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'audit_type':'asset_integrity_check','passed':all(r['outcome']=='identity_resolved' for r in reports),'status':'completed','source_kind':'automated','runner_version':'competition-runtime-identity.v1','details':Jsonb(details)})
            totals['templates_examined']+=1
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(totals)},indent=2))


if __name__=='__main__':main()
