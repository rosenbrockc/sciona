"""Refresh physics family coverage including independent first-wave sources.

Counts served derived scopes, not blanket approval of their source equations.
Does not change previous publication-bound evidence or any catalog rows.
"""
import hashlib
import json
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from scripts.audit_physics_legacy_reconciliation import audit


def refresh(root):
    reconciliation=audit(root)
    families=[]
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for row in reconciliation['graphs']:
            if row.get('projected_version_id'):
                version=row['projected_version_id']
                served=row['served_realizations']+row['served_derived_source_lineage']
                kind='remote_source_family'
            else:
                version=row['legacy_version_id'];kind='independent_source_family'
                source=db.execute('SELECT a.fqdn FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(version,)).fetchone()
                served=db.execute("""SELECT DISTINCT s.artifact_id,v.version_id,v.content_hash,v.trust_tier
                    FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id AND v.is_latest
                    JOIN catalog_artifacts_served s USING(artifact_id)
                    WHERE s.artifact_kind='cdg' AND d.dependency_artifact_fqdn=%s AND d.dependency_content_hash=%s
                    AND d.dependency_role='cdg'""",(source['fqdn'],row['legacy_content_hash'])).fetchall()
            families.append(dict(kind=kind,source_version_id=version,served_realizations=served,
                                 source_reconciliation=row['classification']))
        inventory=db.execute("""SELECT a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_kind='cdg'
            AND a.fqdn LIKE 'physics.%' AND v.is_latest ORDER BY a.fqdn""").fetchall()
        accounted={str(v) for row in reconciliation['graphs'] for v in
                   [row['legacy_version_id'],row.get('projected_version_id')] if v}
        accounted.update(str(v['version_id']) for f in families for v in f['served_realizations'])
        remaining=[r for r in inventory if str(r['version_id']) not in accounted]
    return dict(read_only=True,approval_applied=False,served_cdgs=reconciliation['served_cdgs'],
        physics_families=len(families),families_with_served_scope=sum(bool(f['served_realizations']) for f in families),
        families_without_served_scope=sum(not f['served_realizations'] for f in families),
        other_current_physics_graphs=remaining,families=families,
        representation_counts=dict(projected=reconciliation['projected_graphs'],legacy=reconciliation['legacy_graphs']),
        limitations=['A served corrected scope does not certify every original inference or equation.',
                     'Family coverage is based on exact source hashes or retained derives_from links.',
                     'Other current physics graph versions are reported separately, not silently treated as covered.'],
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];result=refresh(root)
    (root/'docs/reviews/physics_served_coverage_refresh.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['families','implementation_sha256']},default=str))
