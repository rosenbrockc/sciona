#!/usr/bin/env python3
"""Execute periodic wave parameters through the serialized CDG runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import circular_orbit_speed as provider
from sciona.physics_ingest.circular_orbit_speed_execution import SOURCE_VERSION,build_circular_orbit_speed_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_circular_orbit_speed_source import validate as validate_source
from tests.physics_ingest.test_circular_orbit_speed import oracle


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_circular_orbit_speed_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    rng=np.random.default_rng(909)
    cases=[(np.exp(rng.uniform(-300,300,size=(2,3,4))),np.exp(rng.uniform(-300,300,size=(2,3,4))))]
    cases += [(np.array(f),np.array(v)) for f,v in [(149600000000.,31536000.),(149600000000.,31600000.),(1e307,1e307),(np.nextafter(0.,1.),1.),(1.,1e308)]]
    with tempfile.TemporaryDirectory(prefix='sciona-wave-relations-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,args in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-wave-relations',run).execute(
                    dict(radius=args[0],period=args[1]),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                expected_values=[oracle(r,t) for r,t in zip(args[0].flat,args[1].flat)]
                expected_arrays=[np.array([e[j] for e in expected_values]).reshape(args[0].shape) for j in range(2)]
                for name,expected in zip(['circumference','average_speed'],expected_arrays):
                    actual=np.load(Path(directory)/run/('orbit/out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/circular_orbit_speed_execution.py','scripts/validate_circular_orbit_speed_execution.py',
           'tests/physics_ingest/test_circular_orbit_speed.py','tests/physics_ingest/test_circular_orbit_speed_proof.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
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
