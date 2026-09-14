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
from sciona.atoms.physics import integration_by_parts as provider
from sciona.physics_ingest.integration_parts_execution import SOURCE_VERSION,build_integration_parts_execution
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_integration_parts_source import validate as validate_source


async def validate(root,symbol_file,rule_file):
    proof=validate_source(root,symbol_file,rule_file)
    graph=build_integration_parts_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    x=sp.Symbol('x',real=True);constant=sp.Symbol('C')
    cases=[(x,sp.exp(x)),(x*x,sp.sin(x)),(sp.log(x),x),
           (sp.Function('u')(x),sp.Function('v')(x)),(sp.Function('f')(x*x),sp.Function('g')(sp.sin(x))),
           (sp.I*x,sp.cos(x))]
    with tempfile.TemporaryDirectory(prefix='sciona-parts-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,(u,v) in enumerate(cases):
                run='case-'+str(i)
                inputs=dict(zip(['u_srepr','v_srepr','variable_srepr','constant_srepr'],map(sp.srepr,[u,v,x,constant])))
                result=await runner.CDGExecutionSession(None,'synthetic-parts',run).execute(inputs,cdg=restored)
                if result['status']!='completed':raise ValueError('Full symbolic runner failed')
                outputs=[]
                for name in ['integrand_srepr','antiderivative_srepr']:
                    record=json.loads((Path(directory)/run/'parts'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('AST must remain a string')
                    outputs.append(parse_source_srepr(record['value']))
                integrand,primitive=outputs
                if not primitive.has(sp.Integral):raise ValueError('Residual integral was lost')
                if sp.expand(integrand.doit()-u*sp.diff(v,x))!=0:raise ValueError('Original integrand differs')
                if sp.expand((sp.diff(primitive,x)-integrand).doit())!=0:raise ValueError('Derivative verification failed')
                cached=runner.load_cached_outputs(Path(directory)/run,'parts')
                for name in ['integrand_srepr','antiderivative_srepr']:
                    if not isinstance(cached['out_'+name],str):raise ValueError('Cache lost AST string type')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/integration_parts_execution.py','scripts/validate_integration_parts_execution.py',
           'tests/physics_ingest/test_integration_by_parts.py','tests/physics_ingest/test_integration_parts_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),
                symbolic_derivative_checks=6,residual_integrals_preserved=True,cache_ast_strings_preserved=True,
                source_proof=proof,sympy_version=importlib.metadata.version('sympy'),
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','symbolic_derivative_checks','residual_integrals_preserved','cache_ast_strings_preserved']}))
