#!/usr/bin/env python3
"""Execute corrected projectile CDG with independent high-precision reference."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import projectile_trajectory as provider
from sciona.physics_ingest.projectile_execution import SOURCE_VERSION, build_projectile_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_projectile_source import validate as validate_source
from tests.physics_ingest.test_projectile_trajectory import reference


async def validate(root, symbol_file, rule_file, expression_file):
    proof = validate_source(root, symbol_file, rule_file, expression_file)
    graph = build_projectile_execution(); digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph != restored: raise ValueError('Graph round trip differs')
    rng = np.random.default_rng(744)
    shape = (2, 3, 4); x0 = rng.normal(size=shape); vx = rng.uniform(1, 10, size=shape)*rng.choice([-1,1],size=shape)
    cases = [[x0+vx*rng.uniform(0,10,size=shape),x0,rng.normal(size=shape),vx,rng.normal(size=shape),rng.uniform(0,20,size=shape)]]
    for values in [[12.,2.,3.,5.,7.,10.],[-12.,-2.,3.,-5.,7.,10.],[1e308,-1e308,0.,1e308,0.,1.],
                   [np.nextafter(0.,1.),0.,0.,1.,1.,0.],[1.,0.,-1e308,1.,1e308,0.]]:
        cases.append([np.array(v) for v in values])
    with tempfile.TemporaryDirectory(prefix='sciona-projectile-runner-') as directory:
        prior = runner.RUNS_DIR; runner.RUNS_DIR = Path(directory)
        try:
            for i,args in enumerate(cases):
                run = 'case-'+str(i)
                result = await runner.CDGExecutionSession(None,'synthetic-projectile',run).execute(
                    dict(zip(['x','x0','y0','vx0','vy0','gravity'],args)),cdg=restored)
                if result['status'] != 'completed': raise ValueError('Full runner failed')
                for name,expected in zip(['elapsed_time','height'],reference(args)):
                    actual = np.load(Path(directory)/run/'trajectory'/('out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally: runner.RUNS_DIR = prior
    files = ['sciona/physics_ingest/projectile_execution.py','scripts/validate_projectile_execution.py',
             'tests/physics_ingest/test_projectile_trajectory.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),synthetic_points=29,
                maximum_ulp_error=0,source_proof=proof,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']: parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_points','maximum_ulp_error']}))
