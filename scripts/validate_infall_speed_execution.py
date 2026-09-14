#!/usr/bin/env python3
"""Execute radial infall speed and energy through the serialized CDG runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import infall_speed as provider
from sciona.physics_ingest.infall_speed_execution import SOURCE_VERSION,build_infall_speed_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_infall_speed_source import validate as validate_source
from tests.physics_ingest.test_infall_speed import reference


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_infall_speed_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    rng=np.random.default_rng(909)
    G,M,r=(np.exp(rng.uniform(-80,80,size=(2,3,4))) for _ in range(3))
    cases=[(G,M*1e-6,M,r)]
    cases += [tuple(np.array(v) for v in row) for row in [(1.,1e-6,1.,2.),
        (1e308,1e-6,1.,1e308),(1e-308,1e-6,1.,1.),(1.,1.,1e308,1e308),(1.,np.nextafter(0.,1.),1.,1.)]]
    with tempfile.TemporaryDirectory(prefix='sciona-infall-speed-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,args in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-infall-speed',run).execute(
                    dict(gravitational_constant=args[0],test_mass=args[1],central_mass=args[2],radius=args[3]),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                expected_values=[reference(*row) for row in zip(*(a.flat for a in args))]
                expected_arrays=[np.array([e[j] for e in expected_values]).reshape(args[0].shape) for j in range(4)]
                for name,expected in zip(['speed','inward_radial_velocity','gravitational_work','potential_energy'],expected_arrays):
                    actual=np.load(Path(directory)/run/('infall/out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/infall_speed_execution.py','scripts/validate_infall_speed_execution.py',
           'tests/physics_ingest/test_infall_speed.py','tests/physics_ingest/test_infall_speed_proof.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),synthetic_states=29,
                maximum_ulp_error=0,source_proof=proof,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_states','maximum_ulp_error']}))
