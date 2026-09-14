#!/usr/bin/env python3
"""Full-runner tests for the reconstructed momentum-vector realization."""
import argparse
import asyncio
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.atoms.physical_quantities import momentum_recoil as provider
from sciona.physics_ingest.momentum_execution import build_momentum_execution, SOURCE_VERSION
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_momentum_vector_source import validate as validate_source


def reference(a, b):
    with localcontext() as ctx:
        ctx.prec = 2500
        differences = [[Decimal.from_float(float(x))-Decimal.from_float(float(y)) for x, y in zip(u, v)]
                       for u, v in zip(a.reshape(-1, 3), b.reshape(-1, 3))]
        return (np.asarray([[float(d) for d in row] for row in differences]).reshape(a.shape),
                np.asarray([float(sum(d*d for d in row)) for row in differences]).reshape(a.shape[:-1]))


async def validate(root, symbol_file, rule_file):
    proof = validate_source(root, symbol_file, rule_file)
    graph = build_momentum_execution()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': SOURCE_VERSION} for n in nodes], 'cdg_edges': edges},
        version_id=SOURCE_VERSION, content_hash=digest, require_execution_envelope=True)
    if restored != graph:
        raise ValueError('Graph serialization differs')
    rng = np.random.default_rng(6237)
    cases = [(rng.normal(size=(2, 40, 3)), rng.normal(size=(2, 40, 3))),
             (np.array([1e308]*3), np.array([1e308]*3)),
             (np.array([1., 2., 3.]), np.array([np.nextafter(1., 0.), 2., 3.])),
             (np.array([2e-162, 0., 0.]), np.zeros(3)),
             (np.array([-1e150, 2e150, -3e150]), np.array([1e150, -2e150, 3e150]))]
    with tempfile.TemporaryDirectory(prefix='sciona-momentum-') as directory:
        prior = runner.RUNS_DIR
        runner.RUNS_DIR = Path(directory)
        try:
            for i, (a, b) in enumerate(cases):
                run = 'case-'+str(i)
                result = await runner.CDGExecutionSession(None, 'synthetic-momentum', run).execute(
                    dict(incoming_momentum=a, outgoing_momentum=b), cdg=restored)
                if result['status'] != 'completed':
                    raise ValueError('Full runner failed')
                for name, expected in zip(['recoil_momentum', 'squared_momentum'], reference(a, b)):
                    actual = np.load(Path(directory)/run/'recoil'/('out_'+name+'.npy'))
                    if actual.shape != expected.shape:
                        raise ValueError('Output shape differs')
                    np.testing.assert_array_equal(actual, expected)
        finally:
            runner.RUNS_DIR = prior
    files = ['sciona/physics_ingest/momentum_execution.py', 'scripts/validate_momentum_execution.py',
             'tests/physics_ingest/test_momentum_recoil.py', 'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    return dict(graph_digest=digest, execution_graph=graph.model_dump(mode='json'), full_runner_cases=len(cases),
                synthetic_vectors=sum(a.size//3 for a, _ in cases), maximum_ulp_error=0, source_proof=proof,
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                limitations=['Real Euclidean three-vectors in one orthonormal frame and consistent SI momentum units.',
                             'Conservation premises required for recoil interpretation; no energy or full scattering-event validation.',
                             'Squared norm uses exact converted-input differences before rounding; it need not equal the norm of the rounded vector.',
                             'Nonzero outputs rounding to zero and nonfinite outputs rejected; exact zeros and subnormals supported.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['full_runner_cases', 'synthetic_vectors', 'maximum_ulp_error']}))
