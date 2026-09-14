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
from sciona.atoms.physics import helmholtz as provider
from sciona.physics_ingest.helmholtz_execution import SOURCE_VERSION,build_helmholtz_execution
from sciona.physics_ingest.source_symbolic import parse_source_srepr
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_helmholtz_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_helmholtz_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    x,y,z,t=sp.symbols('x y z t',real=True);q=sp.Tuple(x,y,z,t);f=sp.Function('f')
    cases=[((sp.exp(sp.I*2*z),0,0),2,1),((0,0,sp.exp(sp.I*2*z)),2,1),
           ((x*x,y*y,z*z),2,3),((x*x-y*y,0,1),0,3),
           ((sp.I*sp.sin(y),sp.exp(z),sp.cos(x)),-3,2),((f(x*y),f(z*z),0),2,1)]
    names=['harmonic_field_srepr','laplacian_srepr','helmholtz_residual_srepr']
    with tempfile.TemporaryDirectory(prefix='sciona-curl-curl-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,(field,w,c) in enumerate(cases):
                run='case-'+str(i)
                inputs=dict(zip(['amplitude_srepr','coordinates_srepr','angular_frequency_srepr','wave_speed_srepr'],map(sp.srepr,[sp.Tuple(*field),q,sp.sympify(w),sp.sympify(c)])))
                result=await runner.CDGExecutionSession(None,'synthetic-curl-curl',run).execute(inputs,cdg=restored)
                if result['status']!='completed':raise ValueError('Full symbolic runner failed')
                outputs=[]
                for name in names:
                    record=json.loads((Path(directory)/run/'harmonic'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('AST must remain a string')
                    outputs.append(provider._checked_parse(record['value']))
                phase=sp.exp(sp.I*w*t)
                expected_lap=sp.Tuple(*(sum(sp.diff(e,v,2) for v in q[:3]) for e in field))
                expected=[sp.Tuple(*(e*phase for e in field)),expected_lap,
                          sp.Tuple(*(l+sp.Rational(w,c)**2*e for l,e in zip(expected_lap,field)))]
                for actual,wanted in zip(outputs,expected):
                    if any(sp.simplify(a.doit()-b)!=0 for a,b in zip(actual,wanted)):
                        raise ValueError('Independent derivative oracle mismatch')
                harmonic,lap,residual=outputs
                for h,r in zip(harmonic,residual):
                    wave=sum(sp.diff(h,v,2) for v in q[:3])-sp.diff(h,t,2)/c**2
                    if sp.simplify(wave-phase*r.doit())!=0:raise ValueError('Residual equivalence failed')
                if i==2 and all(sp.simplify(r.doit())==0 for r in residual):raise ValueError('Non-solution lost')
                if i==1 and sp.diff(harmonic[2],z)==0:raise ValueError('Longitudinal counterexample lost')
                if not lap.has(sp.Derivative):raise ValueError('Formal spatial derivatives lost')
                cached=runner.load_cached_outputs(Path(directory)/run,'harmonic')
                for name in names:
                    if not isinstance(cached['out_'+name],str):raise ValueError('Cache lost AST string type')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/helmholtz_execution.py','scripts/validate_helmholtz_execution.py',
           'tests/physics_ingest/test_helmholtz.py','tests/physics_ingest/test_helmholtz_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),
                symbolic_derivative_checks=6,formal_derivatives_preserved=True,nonzero_residual_preserved=True,cache_ast_strings_preserved=True,
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
