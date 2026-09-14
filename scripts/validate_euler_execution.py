#!/usr/bin/env python3
"""Run the exact Euler proof CDG and compare every step to reviewed source semantics."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from sciona.atoms.physical_quantities import euler_identity_proof as provider
from sciona.physics_ingest.euler_execution import build_euler_execution,SOURCE_VERSION
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_euler_source_interpretation import validate as validate_source


async def validate(root,symbol_file,rule_file):
    source=validate_source(root,symbol_file,rule_file)
    if len(source['graphs'])!=1 or source['graphs'][0]['version_id']!=SOURCE_VERSION:raise ValueError('Unique interpreted source proof required')
    proof=source['graphs'][0]
    graph=build_euler_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph roundtrip differs')
    certificates=[]
    with tempfile.TemporaryDirectory(prefix='sciona-euler-') as temp:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(temp)
        try:
            for index in range(2):
                run='case-'+str(index)
                result=await runner.CDGExecutionSession(None,'exact-euler-proof',run).execute({},cdg=restored)
                if result['status']!='completed':raise ValueError('Exact proof execution failed')
                metadata=json.loads((Path(temp)/run/'proof/out_certificate.json').read_text())
                certificate=metadata['value']
                if not isinstance(certificate,dict) or certificate['schema']!='sciona.euler-proof.v1':raise ValueError('Certificate serialization failed')
                actual=[s['equation'] for s in certificate['steps']]
                expected=[s['computed_srepr'] for s in proof['steps']]
                if actual!=expected:raise ValueError('Source interpreted step differs')
                if certificate['terminal_equation']!='Equality(Integer(0), Integer(0))':raise ValueError('Terminal proof differs')
                certificates.append(certificate)
        finally:runner.RUNS_DIR=prior
    if certificates[0]!=certificates[1]:raise ValueError('Fresh proof runs differ')
    files=['sciona/physics_ingest/euler_execution.py','scripts/validate_euler_execution.py',
        'tests/physics_ingest/test_euler_identity_proof.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(graph_digest=digest,execution_graph=graph.model_dump(mode='json'),source_interpretation=source,
        full_runner_cases=2,proof_steps_per_run=4,exact_source_interpreted_step_parity=True,
        certificate=certificates[0],provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        limitations=['Fixed exact identity proof; not a generic theorem prover or physical simulation.',
            'Explicit reviewed interpretation of source constants; no literal source-AST parity.',
            'Root identity verified through SymPy complex expansion, not derived from first principles.'],
        publication='unapproved realization pending semantic/catalog review')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','proof_steps_per_run','exact_source_interpreted_step_parity']}))
