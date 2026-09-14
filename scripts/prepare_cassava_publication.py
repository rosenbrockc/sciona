"""Prepare a reviewable Tier 3 catalog plan without changing catalog state."""
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.competition_graph import load_competition_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.build_cassava_execution_graph import build_graph
from scripts.review_cassava_execution import ROOT, SOURCE_VERSION, SOURCE_HASH, STAGE_MAPPING, audit, sha

RUNNER = 'cassava-execution-community.v1'


def prepare():
    from sciona.atoms.ml import cassava_execution as provider
    readiness = audit()
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    if digest != readiness['serialized_graph_sha256']:
        raise ValueError('Graph changed since structural review')
    directory = ROOT.parent / 'sciona-atoms-ml'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                    artifact_root=directory / 'src/sciona/atoms')
    with psycopg.connect(dotenv_values(ROOT / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            row_factory=dict_row, options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        source = db.execute('''SELECT a.artifact_id,a.fqdn,a.status,a.is_publishable,v.content_hash
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s''',
            (SOURCE_VERSION,)).fetchone()
        if (not source or source['content_hash'] != SOURCE_HASH or source['status'] != 'draft'
                or source['is_publishable']):
            raise ValueError('Original intake identity or draft state differs')
        original = load_competition_graph(db, version_id=SOURCE_VERSION)
        if {node.node_id for node in original.nodes} != set(STAGE_MAPPING):
            raise ValueError('Original intake stage inventory differs')
        owners = db.execute('''SELECT DISTINCT a.owner_id,a.source_repo_id
            FROM atoms a JOIN atom_source_repositories r USING(source_repo_id)
            WHERE r.repo_name='sciona-atoms-ml' ''').fetchall()
        if len(owners) != 1:
            raise ValueError('Ambiguous provider ownership')
    graph_id = uuid5(UUID(str(source['artifact_id'])), RUNNER)
    providers, ports, bindings = [], {}, []
    for node in graph.nodes:
        matches = [row for row in parsed if row.import_module + '.' + row.source_symbol == node.matched_primitive]
        if len(matches) != 1:
            raise ValueError('Unique registered provider required')
        spec = matches[0]
        function = inspect.unwrap(getattr(provider, spec.source_symbol))
        if spec.file_path.resolve() != Path(inspect.getfile(function)).resolve():
            raise ValueError('Provider source anchor differs')
        signature = inspect.signature(function)
        if list(signature.parameters) != [port.name for port in node.inputs]:
            raise ValueError('Graph and callable parameter order differs')
        provider_id = uuid5(NAMESPACE_URL, 'sciona-provider-draft:' + spec.fqdn)
        ports[node.node_id] = {direction: [port.model_dump(mode='json') for port in selected]
            for direction, selected in [('input', node.inputs), ('output', node.outputs)]}
        providers.append(dict(artifact_id=str(provider_id), version_id=str(spec.version_id),
            fqdn=spec.fqdn, content_hash=spec.content_hash, fingerprint=spec.fingerprint,
            semver=spec.semver, target_trust_tier=3, description=spec.description,
            import_module=spec.import_module, source_symbol=spec.source_symbol,
            source_sha256=sha(spec.file_path), signature=str(signature),
            owner_id=str(owners[0]['owner_id']), source_repo_id=str(owners[0]['source_repo_id']),
            io_specs=ports[node.node_id]))
        bindings.append(dict(node_id=node.node_id, bound_artifact_fqdn=spec.fqdn,
            bound_version_content_hash=spec.content_hash, binding_source=RUNNER))
    connected = {(edge.target_id, edge.input_name) for edge in graph.edges}
    consumed = {(edge.source_id, edge.output_name) for edge in graph.edges}
    boundary = {direction: [port for node in graph.nodes for port in ports[node.node_id][direction]
                 if (node.node_id, port['name']) not in excluded]
                for direction, excluded in [('input', connected), ('output', consumed)]}
    for selected in boundary.values():
        if len({port['name'] for port in selected}) != len(selected):
            raise ValueError('Ambiguous boundary ports')
    return {'approved': False, 'applied': False, 'catalog_mutations': 0,
        'runner_version': RUNNER, 'publication_ready': False,
        'required_actions': readiness['required_actions'], 'providers': providers,
        'execution': {'artifact_id': str(graph_id), 'version_id': str(uuid5(graph_id, digest)),
            'fqdn': source['fqdn'] + '.execution', 'content_hash': digest,
            'target_trust_tier': 3, 'nodes': nodes, 'edges': edges, 'bindings': bindings,
            'metadata': graph.metadata, 'io_specs': boundary,
            'mandatory_source_dependency': {'dependency_artifact_fqdn': source['fqdn'],
                'dependency_content_hash': SOURCE_HASH, 'optional': False,
                'scope': 'Conceptual provenance; original intake is not invoked and remains draft'}},
        'review_conditions': readiness['limitations'] + readiness['reference_conditions'],
        'sha256': {'scripts/prepare_cassava_publication.py': sha(Path(__file__)),
                   'scripts/build_cassava_execution_graph.py': sha(ROOT / 'scripts/build_cassava_execution_graph.py'),
                   'docs/reviews/competition_cassava_source_review.json': sha(ROOT / 'docs/reviews/competition_cassava_source_review.json')}}


if __name__ == '__main__':
    plan = prepare()
    (ROOT / 'docs/reviews/competition_cassava_publication_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'providers': len(plan['providers']), 'execution_cdgs': 1,
        'target_trust_tier': 3, 'publication_ready': False, 'catalog_mutations': 0}))
