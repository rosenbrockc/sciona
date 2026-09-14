#!/usr/bin/env python3
"""Full-runner variance validation with independent raw moment arithmetic."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import weighted_population_variance as provider
from sciona.physics_ingest.variance_execution import SOURCE_VERSION, build_variance_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_variance_source import validate as validate_source
from tests.physics_ingest.test_weighted_population_variance import reference


async def validate(root,symbol_file,rule_file):
    proof=validate_source(root,symbol_file,rule_file)
    graph=build_variance_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    rng=np.random.default_rng(314)
    cases=[(rng.normal(size=(2,3,8)),rng.uniform(.01,10,size=(2,3,8))),
           (np.array([-1.,1.]),np.array([1e300,1e300])),
           (np.array([1e16,1e16+2]),np.ones(2)),
           (np.array([1e308,-1e308]),np.array([1e308,0.])),
           (np.array([-2e-162,2e-162]),np.ones(2)),
           (np.array([-1.,1.]),np.array([1e-300,1e-300]))]
    with tempfile.TemporaryDirectory(prefix='sciona-variance-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,(values,weights) in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-variance',run).execute(dict(values=values,weights=weights),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                for name,expected in zip(['mean','population_variance'],reference(values,weights)):
                    actual=np.load(Path(directory)/run/'variance'/('out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/variance_execution.py','scripts/validate_variance_execution.py',
           'tests/physics_ingest/test_weighted_population_variance.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),synthetic_distributions=11,
                maximum_ulp_error=0,source_proof=proof,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_distributions','maximum_ulp_error']}))
