"""Validate formula outputs and ODE certificate through serialized full runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import sympy as sp
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import sine_squared as provider
from sciona.physics_ingest.sine_squared_execution import SOURCE_VERSION,build_sine_squared_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_sine_squared_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_sine_squared_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph round trip failed')
    u=sp.Symbol('u',real=True)
    cases=[sp.Integer(0),-sp.pi/2,sp.Rational(1,10)**100,sp.sqrt(2),u,u*u+sp.Rational(1,3)]
    names=['sine_squared_srepr','half_angle_srepr','certificate_json']
    with tempfile.TemporaryDirectory(prefix='sciona-sine-squared-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,angle in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-sine-squared',run).execute({'angle_srepr':sp.srepr(angle)},cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                outputs=[]
                for name in names:
                    record=json.loads((Path(directory)/run/'formula'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('String transport lost')
                    outputs.append(record['value'])
                lhs,rhs=map(provider._checked_parse,outputs[:2]);c=json.loads(outputs[2])
                if sp.simplify(lhs-sp.sin(angle)**2)!=0 or sp.simplify(rhs-(1-sp.cos(2*angle))/2)!=0:
                    raise ValueError('Specialized expressions differ')
                if c['angle_srepr']!=sp.srepr(angle) or c['specialized_lhs_srepr']!=outputs[0] or c['specialized_rhs_srepr']!=outputs[1]:
                    raise ValueError('Certificate specialization differs')
                from sciona.physics_ingest.sine_squared_proof import build_proof
                if c['steps']!=[sp.srepr(step) for step in build_proof().steps]:raise ValueError('Full source reconstruction differs')
                if c['execution_order']!=[1,3,4,2,5,6,7,8,9,10,11,12,13]:raise ValueError('Dependency order differs')
                if c['euler_ode_residual']!='Integer(0)' or c['integrating_factor_derivative']!='Integer(0)' or c['sine_sign_inferred'] or c['source_ast_parity']:
                    raise ValueError('Prerequisite or scope certificate differs')
                if c['differential_certificates']!=[dict(residual='Integer(0)',initial_values=['0','0','2'])]*2:
                    raise ValueError('Endpoint differential certificate differs')
                if i>=4 and not lhs.has(u):raise ValueError('General angle scope lost')
                cache=runner.load_cached_outputs(Path(directory)/run,'formula')
                if any(cache['out_'+n]!=s for n,s in zip(names,outputs)):raise ValueError('Cache changed output')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/sine_squared_execution.py','scripts/validate_sine_squared_execution.py',
           'tests/physics_ingest/test_sine_squared.py','tests/physics_ingest/test_sine_squared_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=6,general_angle_preserved=True,
                certificate_verified=True,cache_ast_strings_preserved=True,source_proof=proof,
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','general_angle_preserved','certificate_verified','cache_ast_strings_preserved']}))
