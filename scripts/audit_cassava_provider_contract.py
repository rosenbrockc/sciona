"""Review draft provider registration, code anchors, ports and graph encoding.

This structural preflight performs no catalog writes and cannot approve a CDG.
Full trained graph execution and semantic/publication reviews remain separate.
"""
import hashlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.build_cassava_execution_graph import build_graph

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    from sciona.atoms.ml import cassava_execution as provider
    directory = ROOT.parent / 'sciona-atoms-ml'
    provider_path = directory / 'src/sciona/atoms/ml/cassava_execution.py'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml', directory),
                                    artifact_root=directory / 'src/sciona/atoms')
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    if len(nodes) != 2 or len(edges) != 1 or graph.metadata['publication_status'] != 'draft':
        raise ValueError('Expected unpublished two-node execution graph')
    registrations, symbolic = {}, {}
    for node in graph.nodes:
        matches = [row for row in parsed
                   if row.import_module + '.' + row.source_symbol == node.matched_primitive]
        if len(matches) != 1:
            raise ValueError('Unique provider registration required')
        row = matches[0]
        function = inspect.unwrap(getattr(provider, row.source_symbol))
        witness = getattr(provider, 'witness_' + row.source_symbol)
        if Path(inspect.getfile(function)).resolve() != provider_path.resolve():
            raise ValueError('Provider source anchor differs')
        parameters = inspect.signature(function).parameters
        if list(parameters) != [port.name for port in node.inputs]:
            raise ValueError('Callable parameter order differs from graph')
        if parameters != inspect.signature(witness).parameters:
            raise ValueError('Symbolic parameter contract differs')
        for port, parameter in zip(node.inputs, parameters.values()):
            if not port.required or parameter.default is not inspect.Parameter.empty:
                raise ValueError('All controls must be explicit')
            if not port.constraints or parameter.annotation.__name__ != port.type_desc:
                raise ValueError('Missing constraints or mismatched input type')
        if (len(node.outputs) != 1 or not node.outputs[0].constraints
                or inspect.signature(function).return_annotation.__name__ != node.outputs[0].type_desc):
            raise ValueError('Missing or mismatched output contract')
        arguments = {p.name: symbolic.get(p.name, object()) for p in node.inputs}
        symbolic[node.outputs[0].name] = witness(**arguments)
        lines, first_line = inspect.getsourcelines(function)
        registrations[node.node_id] = {
            'fqdn': row.fqdn, 'symbol': row.source_symbol,
            'relative_file': str(provider_path.relative_to(directory)),
            'first_line': first_line, 'last_line': first_line + len(lines) - 1,
            'signature': str(inspect.signature(function)),
            'version_id': str(row.version_id),
        }
    if symbolic['result'] != {'kind': 'Cassava.ExecutionResult'}:
        raise ValueError('Unexpected composed symbolic result')
    return {'approved': False, 'catalog_mutations': 0, 'passed': True,
            'scope': 'Structural registration, code anchors, explicit IO contracts and symbolic composition only',
            'serialized_graph_sha256': digest, 'registrations': registrations,
            'provider_sha256': sha(provider_path),
            'sha256': {name: sha(ROOT / name) for name in [
                'scripts/build_cassava_execution_graph.py',
                'scripts/audit_cassava_provider_contract.py']}}


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/competition_cassava_provider_structure.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print('PASS draft provider registration, anchors, explicit ports and symbolic composition')
