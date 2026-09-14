"""Read-only reconciliation of pairwise PDG graphs with source-step projections.

Source equivalence is not execution approval: original graphs stay draft and
corrected executable realizations retain their own scope and evidence.
"""
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_inventory(signatures, bindings):
    derivations = {s.get('variable_bindings', {}).get('derivation_id') for s in signatures}
    derivations.discard(None)
    relations = defaultdict(set)
    for signature in signatures:
        ids = signature.get('source_pdg_inference_ids') or [signature.get('source_pdg_inference_id')]
        contract = {key: signature.get(key) for key in
                    ['inference_rule_id', 'assumptions', 'dimensions', 'variable_bindings', 'relationship_kind']}
        for identity in ids:
            if identity:
                relations[identity].add(digest(contract))
    return dict(derivations=sorted(derivations),
                projected=bool(signatures) and all(s.get('source_pdg_step_id') for s in signatures),
                relations={k: sorted(v) for k, v in sorted(relations.items())},
                binding_versions=sorted({(b['bound_artifact_fqdn'], b['bound_version_content_hash']) for b in bindings}),
                all_bindings_active=all(b['status']=='active' for b in bindings))


def audit(root):
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graphs=db.execute("SELECT a.artifact_id,a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_kind='cdg' AND a.fqdn LIKE 'physics.pdg.%' AND a.status='draft' AND v.is_latest ORDER BY a.fqdn").fetchall()
        inventories={}
        for graph in graphs:
            rows=db.execute('SELECT type_signature FROM artifact_cdg_nodes WHERE version_id=%s',(graph['version_id'],)).fetchall()
            signatures=[json.loads(r['type_signature']) if isinstance(r['type_signature'],str) else r['type_signature'] for r in rows]
            bindings=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(graph['version_id'],)).fetchall()
            inventories[str(graph['artifact_id'])]=source_inventory(signatures,bindings)
        projected=defaultdict(list)
        for graph in graphs:
            inv=inventories[str(graph['artifact_id'])]
            if inv['projected'] and len(inv['derivations'])==1:
                projected[inv['derivations'][0]].append(graph)
        reports=[]
        for legacy in graphs:
            inv=inventories[str(legacy['artifact_id'])]
            if inv['projected']: continue
            report=dict(legacy_artifact_id=str(legacy['artifact_id']),legacy_version_id=str(legacy['version_id']),
                        legacy_content_hash=legacy['content_hash'],derivation_ids=inv['derivations'])
            candidates=projected.get(inv['derivations'][0],[]) if len(inv['derivations'])==1 else []
            if len(candidates)!=1:
                report.update(classification='requires_separate_source_review',projected_candidates=len(candidates))
            else:
                candidate=candidates[0]; other=inventories[str(candidate['artifact_id'])]
                gates=dict(same_inference_contracts=inv['relations']==other['relations'],
                           same_bound_versions=inv['binding_versions']==other['binding_versions'],
                           all_bindings_active=inv['all_bindings_active'] and other['all_bindings_active'])
                served=db.execute("SELECT DISTINCT a.artifact_id,c.status,c.is_publishable,v.version_id,v.content_hash,v.trust_tier FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id JOIN catalog_artifacts_served a USING(artifact_id) JOIN artifacts c ON c.artifact_id=a.artifact_id WHERE d.dependency_artifact_fqdn=%s AND d.dependency_content_hash=%s AND d.dependency_role='cdg' AND a.artifact_kind='cdg' AND v.is_latest",(candidate['fqdn'],candidate['content_hash'])).fetchall()
                lineage=db.execute("SELECT DISTINCT a.artifact_id,c.status,c.is_publishable,v.version_id,v.content_hash,v.trust_tier,sv.version_id AS source_version_id,sv.content_hash AS source_content_hash,sv.derives_from AS source_derives_from FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id JOIN catalog_artifacts_served a USING(artifact_id) JOIN artifacts c ON c.artifact_id=a.artifact_id JOIN artifact_versions sv ON sv.artifact_id=%s AND sv.content_hash=d.dependency_content_hash WHERE d.dependency_artifact_fqdn=%s AND d.dependency_role='cdg' AND a.artifact_kind='cdg' AND v.is_latest AND sv.derives_from=%s",(candidate['artifact_id'],candidate['fqdn'],candidate['version_id'])).fetchall()
                report.update(projected_artifact_id=str(candidate['artifact_id']),projected_version_id=str(candidate['version_id']),
                              projected_content_hash=candidate['content_hash'],comparison=gates,
                              legacy_relation_digest=digest(inv['relations']),projected_relation_digest=digest(other['relations']),
                              legacy_binding_digest=digest(inv['binding_versions']),projected_binding_digest=digest(other['binding_versions']),
                              source_differences=dict(
                                  legacy_only_relations=sorted(set(inv['relations'])-set(other['relations'])),
                                  projected_only_relations=sorted(set(other['relations'])-set(inv['relations'])),
                                  changed_contract_relations=sorted(k for k in inv['relations'].keys() & other['relations'].keys() if inv['relations'][k]!=other['relations'][k])),
                              served_realizations=served,served_derived_source_lineage=lineage,
                              classification='same_source_representations' if all(gates.values()) else 'source_comparison_mismatch')
            reports.append(report)
        served_count=db.execute("SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_kind='cdg'").fetchone()['n']
    return dict(read_only=True,approval_applied=False,served_cdgs=served_count,
                projected_graphs=sum(inv['projected'] for inv in inventories.values()),legacy_graphs=len(reports),
                classifications=dict(Counter(r['classification'] for r in reports)),
                exact_current_source_served=sum(bool(r.get('served_realizations')) for r in reports),
                derived_source_version_served=sum(bool(r.get('served_derived_source_lineage')) for r in reports),graphs=reports,
                limitations=['Matching source relations and bound versions does not approve malformed source equations.',
                             'Served corrected realizations have explicit domains; this audit does not claim full literal source parity.',
                             'No legacy status, dependency, publication or selection behavior changed.'],
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    (root/'docs/reviews/physics_legacy_reconciliation.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='graphs'}))
