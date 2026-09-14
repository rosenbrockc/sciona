#!/usr/bin/env python3
"""Preserve the tested frequency realization and exact provider as catalog drafts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_URL
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms, AtomSeedRow
from sciona.architect.handoff import CDGExport
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.physics_ingest.period_frequency_execution import PRIMITIVE
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provider_root = root.parent / 'sciona-atoms'
    specs = [p for p in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms', provider_root), artifact_root=provider_root/'src/sciona/atoms') if p.fqdn == PRIMITIVE]
    if len(specs) != 1:
        raise ValueError('one registered frequency implementation required')
    spec = specs[0]
    counts = Counter()
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('period-frequency-intake.v1'))")
        records = db.execute("SELECT e.*,a.fqdn,v.content_hash AS source_hash FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='period-frequency-execution.v1' AND e.passed FOR SHARE OF e,a,v").fetchall()
        if len(records) != 1:
            raise ValueError('one validated frequency realization required')
        record = records[0]
        details = record['details']
        if details['source_content_hash'] != record['source_hash'] or hashlib.sha256(spec.file_path.read_bytes()).hexdigest() != details['provider_source_sha256']:
            raise ValueError('source or provider drift')
        for path, digest in details['implementation_hashes'].items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest() != digest:
                raise ValueError('execution implementation drift')
        replay = db.execute('SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s FOR SHARE', (details['source_replay_evidence_id'],)).fetchone()
        if not replay or not replay['passed'] or replay['version_id'] != record['version_id'] or _digest(replay['details']) != details['source_replay_sha256']:
            raise ValueError('source replay evidence mismatch')
        graph = CDGExport.model_validate(details['execution_graph'])
        digest, nodes, edges = encode_execution_graph(graph)
        if digest != details['execution_graph_sha256'] or len(nodes) != 1 or edges or graph.nodes[0].matched_primitive != PRIMITIVE:
            raise ValueError('execution graph mismatch')
        owners = db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms'").fetchall()
        if len(owners) != 1:
            raise ValueError('ambiguous provider ownership')
        owner = owners[0]
        atom_id = uuid5(NAMESPACE_URL, 'sciona-provider-draft:' + PRIMITIVE)
        fields = {name: getattr(spec, name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path','import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
        fields.update(source_package=spec.namespace_root, status='flagged', is_publishable=False)
        atom = AtomSeedRow(**fields).as_dict(owner_id=str(owner['owner_id']), source_repo_id=str(owner['source_repo_id']))
        atom['atom_id'] = atom_id
        counts['provider_atoms_created'] += ensure_row(db, 'atoms', {'atom_id': atom_id}, atom)
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'").fetchall()}
        canonical = {k: v for k, v in atom.items() if k in columns}
        canonical.update(artifact_id=atom_id, artifact_kind='atom', status='draft')
        ensure_row(db, 'artifacts', {'artifact_id': atom_id}, canonical)
        for table, key in [('atom_versions','atom_id'), ('artifact_versions','artifact_id')]:
            ensure_row(db, table, {'version_id': spec.version_id}, {'version_id': spec.version_id, key: atom_id, 'content_hash': spec.content_hash, 'semver': spec.semver, 'is_latest': True, 's3_key': '', 'fingerprint': spec.fingerprint, 'trust_tier': 3})
        artifact_id = uuid5(record['artifact_id'], 'period-frequency-execution.v1')
        version_id = uuid5(artifact_id, digest)
        fqdn = record['fqdn'] + '.frequency_execution'
        counts['execution_cdgs_created'] += ensure_row(db, 'artifacts', {'artifact_id': artifact_id}, {'artifact_id': artifact_id, 'artifact_kind': 'cdg', 'fqdn': fqdn, 'description': 'Convert a finite positive cycle duration in seconds to ordinary frequency in hertz; does not estimate period or compute angular frequency.', 'status': 'draft', 'is_publishable': False, 'verified_leaf_coverage': 0.0})
        ensure_row(db, 'artifact_versions', {'version_id': version_id}, {'version_id': version_id, 'artifact_id': artifact_id, 'content_hash': digest, 'semver': '0.0.0+execution.' + digest[:12], 'is_latest': False, 'trust_tier': 3})
        for node in nodes:
            row = {'version_id': version_id, **{k: node[k] for k in ['node_id','parent_node_id','name','description','concept_type','status','matched_primitive','type_signature']}}
            ensure_row(db, 'artifact_cdg_nodes', {'version_id': version_id, 'node_id': node['node_id']}, row)
        for direction, ports in [('input',graph.nodes[0].inputs), ('output',graph.nodes[0].outputs)]:
            for ordinal, port in enumerate(ports):
                common = {'direction': direction, 'name': port.name, 'ordinal': ordinal, 'type_desc': port.type_desc, 'constraints': port.constraints, 'required': port.required, 'default_value_repr': port.default_value_repr}
                for table, key, identity, selected in [('atom_io_specs','atom_id',atom_id,spec.version_id), ('artifact_io_specs','artifact_id',atom_id,spec.version_id), ('artifact_io_specs','artifact_id',artifact_id,version_id)]:
                    row = {key: identity, 'version_id': selected, **common}
                    if table == 'artifact_io_specs':
                        row['dim_signature'] = port.dim_signature
                    counts['io_rows_created'] += ensure_row(db, table, {k: row[k] for k in [key,'version_id','direction','name']}, row)
        ensure_row(db, 'artifact_cdg_bindings', {'version_id': version_id, 'node_id': 'frequency'}, {'version_id': version_id, 'node_id': 'frequency', 'bound_artifact_fqdn': PRIMITIVE, 'bound_version_content_hash': spec.content_hash, 'binding_confidence': 1.0, 'binding_source': 'period-frequency-intake.v1', 'status': 'active', 'alternatives': Jsonb([]), 'evidence_summary': Jsonb({'execution_evidence_id': str(record['evidence_id'])})})
        ensure_row(db, 'artifact_dependencies', {'dependent_version_id': version_id, 'dependency_artifact_fqdn': record['fqdn'], 'dependency_content_hash': record['source_hash'], 'port_name': ''}, {'dependent_version_id': version_id, 'dependency_artifact_fqdn': record['fqdn'], 'dependency_content_hash': record['source_hash'], 'port_name': '', 'dependency_role': 'cdg', 'optional': False, 'binding_metadata': Jsonb({'scope': 'mandatory source proof provenance, not an invoked numerical dependency', 'source_version_id': str(record['version_id'])})})
        for identity, selected, kind in [(atom_id, UUID(str(spec.version_id)), 'provider'), (artifact_id, version_id, 'execution')]:
            audit_id = uuid5(selected, 'period-frequency-intake.v1')
            counts['audit_records_created'] += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': audit_id}, {'evidence_id': audit_id, 'artifact_id': identity, 'version_id': selected, 'audit_type': 'asset_integrity_check', 'passed': True, 'status': 'completed', 'source_kind': 'automated', 'runner_version': 'period-frequency-intake.v1', 'details': Jsonb({'kind': kind, 'source_execution_evidence_id': str(record['evidence_id']), 'source_execution_evidence_sha256': _digest(details), 'provider_content_hash': spec.content_hash, 'scope': 'Exact validated local catalog intake; approval is a separate review.'})})
        document = db.execute('SELECT get_artifact_document(%s) AS d', (fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(document, version_id=str(version_id), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('catalog graph differs from validated graph')
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'counts': dict(counts), 'catalog_roundtrip': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
