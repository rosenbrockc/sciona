"""Audit current mandatory physics provenance and served provider bindings.

Read-only database audit. Does not execute graphs or certify original equations.
"""
import json
import hashlib
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from scripts.audit_physics_served_coverage import refresh


def audit(root):
    coverage=refresh(root)
    missing=[]
    checked=set()
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            row_factory=dict_row,options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for family in coverage['families']:
            source=family['source_version_id']
            linked=[]
            for realization in family['served_realizations']:
                version=str(realization['version_id'])
                matched=db.execute("""SELECT 1 FROM artifact_dependencies d
                    JOIN artifact_versions original ON original.version_id=%s
                    JOIN artifacts a ON a.artifact_id=original.artifact_id
                    JOIN artifact_versions referenced ON referenced.artifact_id=a.artifact_id
                      AND referenced.content_hash=d.dependency_content_hash
                    WHERE d.dependent_version_id=%s AND NOT d.optional
                      AND d.dependency_role='cdg' AND d.dependency_artifact_fqdn=a.fqdn
                      AND (referenced.version_id=original.version_id OR referenced.derives_from=original.version_id)""",
                    (source,version)).fetchone()
                if matched: linked.append(version)
                checked.add(version)
            if not linked: missing.append(str(source))
        checked.update(str(v['version_id']) for v in coverage['other_current_physics_graphs'])
        tiers={};issues=[];bindings=0
        for version in sorted(checked):
            served=db.execute("""SELECT v.trust_tier FROM catalog_artifacts_served a
                JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s AND v.is_latest""",(version,)).fetchone()
            if not served:
                issues.append(dict(version_id=version,issue='execution version not served'));continue
            tier=str(served['trust_tier']);tiers[tier]=tiers.get(tier,0)+1
            bs=db.execute('SELECT status,bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(version,)).fetchall()
            bindings+=len(bs)
            if not bs:issues.append(dict(version_id=version,issue='no provider bindings'))
            for b in bs:
                found=db.execute("""SELECT 1 FROM catalog_artifacts_served a JOIN artifact_versions v USING(artifact_id)
                    WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest""",
                    (b['bound_artifact_fqdn'],b['bound_version_content_hash'])).fetchone()
                if b['status']!='active' or not found:
                    issues.append(dict(version_id=version,issue='provider binding not active/currently served'))
    return dict(read_only=True,catalog_mutations=0,status='passed' if not missing and not issues else 'incomplete',
        physics_source_families=coverage['physics_families'],families_with_mandatory_served_provenance=coverage['physics_families']-len(missing),
        legacy_representations=coverage['representation_counts']['legacy'],projected_representations=coverage['representation_counts']['projected'],
        separately_accounted_equation_scopes=len(coverage['other_current_physics_graphs']),
        physics_execution_versions=len(checked),trust_tiers=tiers,provider_bindings_checked=bindings,
        missing_source_families=missing,issues=issues,
        limits=['Current catalog provenance and selection metadata audit only; no new numerical execution.',
                'Corrected served scopes do not certify every original equation or inference.',
                'Coverage includes exact source versions or one retained derives_from source-version link.'],
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=audit(root)
    (root/'docs/reviews/physics_mandatory_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
