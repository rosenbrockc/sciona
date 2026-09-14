#!/usr/bin/env python3
"""Verify AST-string transport and symbolic rewrite through the actual runner."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import sympy as sp
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import curl_curl as provider
from sciona.physics_ingest.curl_curl_execution import SOURCE_VERSION,build_curl_curl_execution
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_curl_curl_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_curl_curl_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    x,y,z=sp.symbols('x y z',real=True);q=sp.Tuple(x,y,z);f=sp.Function('f')
    cases=[(x*x*y,y*y*z,z*z*x),(sp.cos(z),0,0),(0,0,sp.cos(z)),(0,1,2),
           (f(x*y),f(z*z),f(x+y+z)),(sp.I*sp.sin(y),sp.exp(z),sp.cos(x))]
    names=['curl_curl_srepr','gradient_divergence_srepr','laplacian_srepr']
    from sciona.physics_ingest.vacuum_wave_proof import curl,gradient,divergence,laplacian
    with tempfile.TemporaryDirectory(prefix='sciona-curl-curl-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,field in enumerate(cases):
                run='case-'+str(i)
                inputs=dict(zip(['field_srepr','coordinates_srepr'],map(sp.srepr,[sp.Tuple(*field),q])))
                result=await runner.CDGExecutionSession(None,'synthetic-curl-curl',run).execute(inputs,cdg=restored)
                if result['status']!='completed':raise ValueError('Full symbolic runner failed')
                outputs=[]
                for name in names:
                    record=json.loads((Path(directory)/run/'curl_identity'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('AST must remain a string')
                    outputs.append(provider._checked_parse(record['value']))
                vector=sp.ImmutableMatrix(field)
                expected=[curl(curl(vector,q),q),gradient(divergence(vector,q),q),laplacian(vector,q)]
                for actual,wanted in zip(outputs,expected):
                    if any(sp.simplify(a.doit()-b)!=0 for a,b in zip(actual,wanted)):
                        raise ValueError('Component differential result differs')
                double,grad,lap=outputs
                if any(sp.simplify(a.doit()-b.doit()+c.doit())!=0 for a,b,c in zip(double,grad,lap)):
                    raise ValueError('Curl-curl identity failed')
                if i==0 and (grad.doit()==sp.Tuple(0,0,0) or double.doit()!=sp.Tuple(2*z,2*x,2*y)):
                    raise ValueError('Explicit polynomial/divergence oracle differs')
                if not all(e.has(sp.Derivative) for e in outputs):raise ValueError('Formal derivatives lost')
                cached=runner.load_cached_outputs(Path(directory)/run,'curl_identity')
                for name in names:
                    if not isinstance(cached['out_'+name],str):raise ValueError('Cache lost AST string type')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/curl_curl_execution.py','scripts/validate_curl_curl_execution.py',
           'tests/physics_ingest/test_curl_curl.py','tests/physics_ingest/test_curl_curl_proof.py',
           'sciona/physics_ingest/vacuum_wave_proof.py','sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),
                symbolic_derivative_checks=6,formal_derivatives_preserved=True,nonzero_divergence_preserved=True,cache_ast_strings_preserved=True,
                source_proof=proof,sympy_version=importlib.metadata.version('sympy'),
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','symbolic_derivative_checks','formal_derivatives_preserved','cache_ast_strings_preserved']}))
