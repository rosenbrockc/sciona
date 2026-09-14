#!/usr/bin/env python3
"""Record immutable version-bound competition callable preflight evidence."""
import argparse
from collections import Counter
import hashlib
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
        for record in records:
            snapshot=record['details']['snapshot']
            if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=record['content_hash']:
                raise ValueError('competition snapshot hash mismatch')
            bindings={r['stage_id']:r for r in snapshot['bindings']['bindings']}
            reports=[]
            for stage in snapshot['template']['stages']:
                binding=bindings.get(stage['stage_id'],{})
                fqdn=binding.get('bound_artifact_fqdn')
                result={'stage_id':stage['stage_id'],'binding_status':binding.get('status'),'target':fqdn}
                if not fqdn:
                    result['outcome']='no_named_implementation'
                elif binding.get('status')!='active':
                    result['outcome']='binding_not_exact_active'
                elif fqdn not in REGISTRY:
                    result['outcome']='target_not_in_runner_registry'
                else:
                    fn=REGISTRY[fqdn]['impl']
                    contract=assess_keyword_call_contract([p['name'] for p in stage.get('inputs',[])],fn)
                    result.update(contract)
                    result['outcome']='keyword_compatible' if contract['keyword_compatible'] else 'call_adapter_required'
                    path=inspect.getsourcefile(inspect.unwrap(fn))
                    if not path:
                        raise ValueError('registered target lacks source anchor')
                    result['implementation_source_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    result['implementation_module']=fn.__module__
                    result['implementation_line']=inspect.getsourcelines(inspect.unwrap(fn))[1]
                totals[result['outcome']]+=1
                reports.append(result)
            passed=all(r['outcome']=='keyword_compatible' for r in reports) and bool(reports)
            totals['templates']+=1
            totals['templates_all_keyword_compatible']+=passed
            details={'scope':'Runner registry resolution and keyword-call compatibility only; no semantic equivalence, execution, dependency approval or publication claim. Missing score/type inputs may require new computations, not argument renaming.', 'source_content_hash':record['content_hash'],'stages':reports}
            digest=hashlib.sha256(_canonical(details).encode()).hexdigest()
            evidence_id=uuid5(record['version_id'],'competition-callable-preflight.v1:'+digest)
            totals['evidence_created']+=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},{'evidence_id':evidence_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'audit_type':'asset_integrity_check','passed':passed,'status':'completed','source_kind':'automated','runner_version':'competition-callable-preflight.v1','details':Jsonb(details)})
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(totals)},indent=2))


if __name__=='__main__':main()
