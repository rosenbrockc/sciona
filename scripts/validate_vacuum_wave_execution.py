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
from sciona.atoms.physics import vacuum_wave as provider
from sciona.physics_ingest.vacuum_wave_execution import SOURCE_VERSION,build_vacuum_wave_execution
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_vacuum_wave_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_vacuum_wave_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    x,y,z,t=sp.symbols('x y z t',real=True);q=sp.Tuple(x,y,z,t);f=sp.Function('f')
    cases=[(x*x*y+t*t*z,y*y*z+t**3*x,z*z*x+t**4*y),
           (sp.cos(z-t),0,0),(0,0,sp.cos(z-t)),(0,0,0),
           (f(x*y+t),f(z*z-t),f(x+y+z+t)),(sp.I*sp.sin(y+t),sp.exp(z-t),sp.cos(x+t))]
    names=['laplacian_srepr','scaled_acceleration_srepr','divergence_srepr']
    with tempfile.TemporaryDirectory(prefix='sciona-vacuum-wave-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,field in enumerate(cases):
                run='case-'+str(i)
                inputs=dict(zip(['field_srepr','coordinates_srepr','permeability_srepr','permittivity_srepr'],
                                map(sp.srepr,[sp.Tuple(*field),q,sp.Integer(1),sp.Integer(1)])))
                result=await runner.CDGExecutionSession(None,'synthetic-vacuum-wave',run).execute(inputs,cdg=restored)
                if result['status']!='completed':raise ValueError('Full symbolic runner failed')
                outputs=[]
                for name in names:
                    record=json.loads((Path(directory)/run/'wave'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('AST must remain a string')
                    outputs.append(provider._checked_parse(record['value']))
                left,right,div=outputs
                for j,e in enumerate(field):
                    if sp.simplify(left[j].doit()-sum(sp.diff(e,c,2) for c in [x,y,z]))!=0:raise ValueError('Laplacian differs')
                    if sp.simplify(right[j].doit()-sp.diff(e,t,2))!=0:raise ValueError('Time derivative differs')
                if sp.simplify(div.doit()-sum(sp.diff(e,c) for e,c in zip(field,[x,y,z])))!=0:raise ValueError('Divergence differs')
                if i==2 and (div.doit()==0 or any(sp.simplify(a.doit()-b.doit())!=0 for a,b in zip(left,right))):
                    raise ValueError('Longitudinal counterexample was lost')
                if not all(e.has(sp.Derivative) for e in outputs):raise ValueError('Formal derivatives lost')
                cached=runner.load_cached_outputs(Path(directory)/run,'wave')
                for name in names:
                    if not isinstance(cached['out_'+name],str):raise ValueError('Cache lost AST string type')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/vacuum_wave_execution.py','scripts/validate_vacuum_wave_execution.py',
           'tests/physics_ingest/test_vacuum_wave.py','tests/physics_ingest/test_vacuum_wave_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),
                symbolic_derivative_checks=6,formal_derivatives_preserved=True,maxwell_counterexample_preserved=True,cache_ast_strings_preserved=True,
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
