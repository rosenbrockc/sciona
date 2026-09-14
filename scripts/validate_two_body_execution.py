#!/usr/bin/env python3
"""Full-runner validation of corrected two-body period on synthetic inputs."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile

import mpmath as mp
import numpy as np

from sciona.atoms.physical_quantities import circular_orbital_period as provider
from sciona.physics_ingest.two_body_execution import build_two_body_execution, SOURCE_VERSION
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_repaired_source_graph import validate as select
from scripts.validate_two_body_corrected_proof import validate_source


def reference(r, a, b, g):
    with mp.workdps(180):
        return np.asarray([float(2*mp.pi*mp.sqrt(mp.mpf(float(x))**3/(mp.mpf(float(g))*(mp.mpf(float(y))+mp.mpf(float(z))))))
                           for x, y, z in zip(r.flat, a.flat, b.flat)]).reshape(r.shape)


async def validate(root, symbol_file, rule_file):
    selection = select(root, symbol_file, rule_file)
    proof = validate_source(selection, symbol_file.read_bytes())
    graph = build_two_body_execution()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': SOURCE_VERSION} for n in nodes], 'cdg_edges': edges},
        version_id=SOURCE_VERSION, content_hash=digest, require_execution_envelope=True)
    if restored != graph:
        raise ValueError('Graph serialization differs')
    wide = np.logspace(-100, 100, 201).reshape(3, 67)
    cases = [(wide, 2*wide, 3*wide, 5.),
             tuple(np.asarray(v) for v in (1e308, 1e308, 1e308))+(1e308,),
             tuple(np.asarray(v) for v in (1e-300, 1e-300, 1e-300))+(1e-300,),
             tuple(np.asarray(np.nextafter(0., 1.)) for _ in range(3))+(1.,),
             (np.array([[1., 2.], [4., 8.]]), np.ones((2, 2)), np.full((2, 2), 3.), 2.)]
    maximum = 0.
    with tempfile.TemporaryDirectory(prefix='sciona-two-body-') as directory:
        prior = runner.RUNS_DIR
        runner.RUNS_DIR = Path(directory)
        try:
            for i, (r, a, b, g) in enumerate(cases):
                run = 'case-'+str(i)
                result = await runner.CDGExecutionSession(None, 'synthetic-two-body', run).execute(
                    dict(separation_metres=r, mass1_kg=a, mass2_kg=b, gravitational_constant=g), cdg=restored)
                if result['status'] != 'completed':
                    raise ValueError('Full runner failed')
                actual = np.load(Path(directory)/run/'period'/'out_period_seconds.npy')
                expected = reference(r, a, b, g)
                if actual.shape != r.shape:
                    raise ValueError('Output shape differs')
                np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
                maximum = max(maximum, float(np.max(np.abs(actual-expected)/np.spacing(expected))))
        finally:
            runner.RUNS_DIR = prior
    files = ['sciona/physics_ingest/two_body_execution.py', 'sciona/physics_ingest/two_body_corrected_proof.py',
             'scripts/validate_two_body_execution.py', 'scripts/validate_two_body_corrected_proof.py',
             'tests/physics_ingest/test_circular_orbital_period.py', 'tests/physics_ingest/test_two_body_corrected_proof.py',
             'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    return dict(graph_digest=digest, execution_graph=graph.model_dump(mode='json'), full_runner_cases=len(cases),
                synthetic_numeric_cases=sum(r.size for r, _, _, _ in cases), maximum_ulp_error=maximum, ulp_tolerance=1,
                source_proof=proof, source_selection_implementation_sha256=selection['implementation_sha256'],
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                limitations=['Caller must establish isolated Newtonian circular two-point-mass model and SI units.',
                             'Corrected inference operations; no literal source-proof parity.',
                             'Binary64 pi; positive subnormals may lose relative precision; per-element Decimal prioritizes range over throughput.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file', 'rule-file', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(Path(__file__).resolve().parents[1], args.symbol_file, args.rule_file))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['full_runner_cases', 'synthetic_numeric_cases', 'maximum_ulp_error']}))
