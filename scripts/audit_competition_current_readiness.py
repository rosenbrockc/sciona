#!/usr/bin/env python3
"""Read-only current competition coverage; never approves intake templates."""
import hashlib,json
from pathlib import Path
from collections import Counter
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.competition_import import _canonical
from sciona.competition_promotion import assess_keyword_call_contract
from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
root=Path(__file__).resolve().parents[1]
with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,options='-c default_transaction_read_only=on') as db:
 rows=db.execute("SELECT e.artifact_id,e.version_id,e.details,v.content_hash,a.fqdn,a.status,a.is_publishable FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) JOIN artifacts a ON a.artifact_id=e.artifact_id WHERE e.runner_version='competition-intake.v1' ORDER BY e.version_id").fetchall()
 catalog={}
 for r in db.execute('SELECT a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,v.content_hash FROM atoms a JOIN atom_versions v USING(atom_id) WHERE v.is_latest'):
  catalog.setdefault(r['fqdn'],[]).append(r)
 deps=db.execute("SELECT d.dependency_artifact_fqdn,d.dependency_content_hash,a.artifact_id FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id JOIN catalog_artifacts_served a ON a.artifact_id=v.artifact_id WHERE v.is_latest AND a.artifact_kind='cdg'").fetchall()
 totals=Counter();reports=[]
 for row in rows:
  snapshot=row['details']['snapshot']
  if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=row['content_hash']:raise ValueError('Snapshot drift')
  bindings={b['stage_id']:b for b in snapshot['bindings']['bindings']};outcomes=Counter();states=Counter()
  for stage in snapshot['template']['stages']:
   b=bindings.get(stage['stage_id'],{});states[b.get('status','missing')]+=1
   target=b.get('bound_artifact_fqdn');candidates=catalog.get(target,[])
   if not target:outcome='no_named_implementation'
   elif len(candidates)!=1:outcome='missing_or_ambiguous_current_catalog_version'
   else:
    c=candidates[0];runtime=(c['import_module'] or '')+'.'+(c['source_symbol'] or '')
    fn=REGISTRY.get(runtime,{}).get('impl')
    if fn is None:outcome='current_callable_not_registered'
    else:
     contract=assess_keyword_call_contract([p['name'] for p in stage.get('inputs',[])],fn)
     outcome='keyword_compatible' if contract['keyword_compatible'] else 'adapter_or_missing_computation'
   outcomes[outcome]+=1
  links={str(d['artifact_id']) for d in deps if d['dependency_artifact_fqdn']==row['fqdn'] and d['dependency_content_hash']==row['content_hash']}
  totals.update(outcomes)
  reports.append(dict(version_id=str(row['version_id']),stages=len(snapshot['template']['stages']),outcomes=dict(outcomes),binding_states=dict(states),original_status=row['status'],original_publishable=row['is_publishable'],served_realizations_with_direct_provenance=len(links)))
 report=dict(read_only=True,templates=len(rows),stage_outcomes=dict(totals),templates_with_direct_served_provenance=sum(bool(r['served_realizations_with_direct_provenance']) for r in reports),templates_all_keyword_compatible=sum(r['outcomes'].get('keyword_compatible',0)==r['stages'] for r in reports),templates_detail=reports,scope='Current callable/keyword coverage and direct served source-provenance only. A linked realization may be partial; no original-template completion or approval claim. Dataset content and identifying metadata excluded.')
 (root/'docs/reviews/competition_current_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='templates_detail'},indent=2))
 ranked=sorted(reports,key=lambda r:(-r['outcomes'].get('keyword_compatible',0),-r['outcomes'].get('adapter_or_missing_computation',0),r['stages']))
 print('ranked',json.dumps(ranked[:8],indent=2))
