#!/usr/bin/env python3
"""Run reconstructed expectation CDGs against independent complex matrix algebra."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import hermitian_expectation as provider
from sciona.physics_ingest.hermitian_execution import SOURCE_VERSION, build_hermitian_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_hermitian_expectation_source import validate as validate_source
from tests.physics_ingest.test_hermitian_expectation import reference


async def validate(root, symbol_file, rule_file):
    proof = validate_source(root, symbol_file, rule_file)
    graph = build_hermitian_execution()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': SOURCE_VERSION} for n in nodes], 'cdg_edges': edges},
        version_id=SOURCE_VERSION, content_hash=digest, require_execution_envelope=True)
    if restored != graph:
        raise ValueError('Graph serialization differs')
    rng = np.random.default_rng(249)
    matrix = rng.normal(size=(2, 4, 3, 3))+1j*rng.normal(size=(2, 4, 3, 3))
    matrix += matrix.swapaxes(-1, -2).conj().copy()
    state = rng.normal(size=(2, 4, 3))+1j*rng.normal(size=(2, 4, 3))
    cases = [(matrix, state), (np.array([[0, -1j], [1j, 0]]), np.array([1, 1j])),
             (np.diag([1e308, -1e308]), np.array([1e308, 1e308])),
             (np.array([[np.nextafter(0., 1.)]]), np.array([1e-300])),
             (np.array([[1, -1j], [1j, 3]]), np.array([1, 1j])*1e-300)]
    with tempfile.TemporaryDirectory(prefix='sciona-hermitian-runner-') as directory:
        prior = runner.RUNS_DIR
        runner.RUNS_DIR = Path(directory)
        try:
            for i, (a, p) in enumerate(cases):
                run = 'case-'+str(i)
                result = await runner.CDGExecutionSession(None, 'synthetic-hermitian', run).execute(dict(operator=a, state=p), cdg=restored)
                if result['status'] != 'completed':
                    raise ValueError('Full runner failed: '+str(result))
                actual = np.load(Path(directory)/run/'expectation/out_expectation.npy')
                n = p.shape[-1]
                expected = np.array([reference(m, v) for m, v in zip(a.reshape(-1, n, n), p.reshape(-1, n))]).reshape(p.shape[:-1])
                np.testing.assert_array_equal(actual, expected)
        finally:
            runner.RUNS_DIR = prior
    files = ['sciona/physics_ingest/hermitian_execution.py', 'scripts/validate_hermitian_execution.py',
             'tests/physics_ingest/test_hermitian_expectation.py', 'tests/physics_ingest/test_hermitian_expectation_proof.py',
             'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    return dict(approved=False, synthetic_only=True, graph_digest=digest, full_runner_cases=len(cases),
                synthetic_states=12, maximum_ulp_error=0, source_proof=proof,
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['full_runner_cases', 'synthetic_states', 'maximum_ulp_error']}))
