#!/usr/bin/env python3
"""Execute corrected quadratic CDG against independent direct high-precision roots."""
import argparse
import asyncio
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.atoms.physical_quantities import quadratic_roots as provider
from sciona.physics_ingest.quadratic_corrected_proof import build_proof,verify_proof
from sciona.physics_ingest.quadratic_execution import build_quadratic_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


def direct_reference(a,b,c):
    with localcontext() as ctx:
        # Independent direct plus/minus formula, with ample precision for the
        # cancellation in these explicitly bounded validation cases.
        ctx.prec=800
        A,B,C=[Decimal.from_float(float(v)) for v in [a,b,c]]
        radical=(B*B-4*A*C).sqrt()
        return sorted([float((-B-radical)/(2*A)),float((-B+radical)/(2*A))])


async def validate(root):
    proof=verify_proof(build_proof())
    graph=build_quadratic_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'corrected-quadratic'} for n in nodes],'cdg_edges':edges},version_id='corrected-quadratic',content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph roundtrip differs')
    rng=np.random.default_rng(871)
    first=rng.integers(-100,100,size=300).astype(float)/2
    second=rng.integers(-100,100,size=300).astype(float)/2
    aa=rng.choice([-4.,-2.,-1.,1.,2.,4.],size=300)
    cases=[(aa,-aa*(first+second),aa*first*second),
        (np.array([[1.,-1.]]),np.array([[2.,-2.]]),np.array([[1.,-1.]])),
        tuple(np.asarray(v) for v in (1.,-1e308,1.)),
        tuple(np.asarray(v) for v in (1e308,0.,-1e308)),
        tuple(np.asarray(v) for v in (1e-300,0.,-1e-300))]
    maximum=0.
    with tempfile.TemporaryDirectory(prefix='sciona-quadratic-') as temp:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(temp)
        try:
            for index,(a,b,c) in enumerate(cases):
                run='case-'+str(index)
                result=await runner.CDGExecutionSession(None,'synthetic-quadratic',run).execute(dict(a=a,b=b,c=c),cdg=restored)
                if result['status']!='completed':raise ValueError('Full graph execution failed')
                lo=np.load(Path(temp)/run/'roots/out_lower_root.npy')
                hi=np.load(Path(temp)/run/'roots/out_upper_root.npy')
                expected=np.asarray([direct_reference(A,B,C) for A,B,C in zip(a.flat,b.flat,c.flat)])
                for actual,wanted in [(lo,expected[:,0].reshape(a.shape)),(hi,expected[:,1].reshape(a.shape))]:
                    if actual.shape!=a.shape:raise ValueError('Root shape differs')
                    np.testing.assert_array_max_ulp(actual,wanted,maxulp=3)
                    maximum=max(maximum,float(np.max(np.abs(actual-wanted)/np.abs(np.spacing(wanted)))))
                if np.any(lo>hi):raise ValueError('Root ordering differs')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/quadratic_corrected_proof.py','sciona/physics_ingest/quadratic_execution.py',
        'tests/physics_ingest/test_quadratic_corrected_proof.py','tests/physics_ingest/test_real_quadratic_roots.py',
        'scripts/validate_quadratic_execution.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(corrected_proof=proof,execution_graph=graph.model_dump(mode='json'),graph_digest=digest,
        full_runner_cases=len(cases),synthetic_polynomials=sum(a.size for a,_,_ in cases),maximum_ulp_error=maximum,
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        publication='unapproved numerical realization; source-correction review and catalog intake pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1]))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_polynomials','maximum_ulp_error']}))
