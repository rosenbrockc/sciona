#!/usr/bin/env python3
"""Execute normalized two-wave interference through the serialized CDG runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import wave_interference as provider
from sciona.physics_ingest.wave_interference_execution import SOURCE_VERSION,build_wave_interference_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_wave_interference_source import validate as validate_source
from tests.physics_ingest.test_wave_interference import reference


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_wave_interference_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    rng=np.random.default_rng(128)
    cases=[(np.exp(rng.uniform(-80,80,(2,3,4))),np.exp(rng.uniform(-80,80,(2,3,4))),rng.uniform(-100,100,(2,3,4)))]
    cases += [tuple(np.array(v,dtype=float) for v in row) for row in [
        (1,1,0),(1,1,np.pi),(1,2,1e308),(0,2,-1e308),(1e-160,0,0)]]
    with tempfile.TemporaryDirectory(prefix='sciona-wave-interference-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,args in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-wave-interference',run).execute(
                    dict(amplitude_a=args[0],amplitude_b=args[1],relative_phase=args[2]),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                expected_values=[reference(*row) for row in zip(*(a.flat for a in args))]
                expected_arrays=[np.array([e[j] for e in expected_values]).reshape(args[0].shape) for j in range(6)]
                for name,expected in zip(['intensity','incoherent_mean','constructive','destructive','interference','ratio'],expected_arrays):
                    actual=np.load(Path(directory)/run/('interference/out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/wave_interference_execution.py','scripts/validate_wave_interference_execution.py',
           'tests/physics_ingest/test_wave_interference.py','tests/physics_ingest/test_wave_interference_proof.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
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
