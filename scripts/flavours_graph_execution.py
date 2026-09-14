"""Execute the serialized Flavours draft graph with synthetic diagnostics only.

Only intermediate persistence is intercepted. Providers and the execution
runner remain real; private runtime values are kept in memory by this helper.
"""
import asyncio
import hashlib
import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch

from sciona.atoms.ml import flavours_execution as provider
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.build_flavours_execution_graph import build_graph


def execute_through_graph(training_inputs, labels, mass, query_inputs, controls, excluded_column):
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Serialized graph roundtrip changed execution identity')
    paths = [Path(__file__), Path(inspect.getfile(provider)),
             Path(inspect.getfile(build_graph)), Path(inspect.getfile(runner))]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {p: sha(p) for p in paths}
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[(node, name)] = value
    arguments = dict(training_inputs=training_inputs, labels=labels, mass=mass,
                     query_inputs=query_inputs, controls=controls, excluded_column=excluded_column)
    with tempfile.TemporaryDirectory(prefix='flavours-synthetic-graph-') as directory:
        with patch.object(runner, 'RUNS_DIR', Path(directory)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-flavours', 'case').execute(
                arguments, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Flavours graph did not complete')
    if set(captured) != {('execute', 'out_result')}:
        raise ValueError('The complete provider must execute and return its declared output')
    if before != {p: sha(p) for p in paths}:
        raise ValueError('Graph, provider or runner changed during execution')
    result = dict(captured[('execute', 'out_result')])
    result['_graph_evidence'] = {'serialized_graph_sha256': digest,
        'provider_sha256': before[Path(inspect.getfile(provider))],
        'actual_runner_nodes': 1, 'codec_roundtrip': True}
    return result
