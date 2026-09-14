"""Validate three-dimensional free plane wave and differential certificate through serialized full runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import sympy as sp
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import free_schrodinger as provider
from sciona.physics_ingest.free_schrodinger_execution import SOURCE_VERSION,build_free_schrodinger_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_free_schrodinger_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_free_schrodinger_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph round trip failed')
    from tests.physics_ingest.test_free_schrodinger import cases,check_outputs
    inputs=cases()
    input_names=['mass_srepr','hbar_srepr','momentum_srepr','amplitude_srepr']
    names=['wave_srepr','energy_srepr','gradient_json','laplacian_srepr','time_derivative_srepr','certificate_json']
    with tempfile.TemporaryDirectory(prefix='sciona-free-schrodinger-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,args in enumerate(inputs):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-free-schrodinger',run).execute(dict(zip(input_names,map(sp.srepr,args))),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                outputs=[]
                for name in names:
                    record=json.loads((Path(directory)/run/'wave'/('out_'+name+'.json')).read_text())
                    if record['type']!='str':raise ValueError('String transport lost')
                    outputs.append(record['value'])
                check_outputs(args,outputs)
                if i>=4:
                    field=provider._checked_parse(outputs[0])
                    if not set().union(*(a.free_symbols for a in args)).issubset(field.free_symbols):
                        raise ValueError('General symbolic parameters lost')
                cache=runner.load_cached_outputs(Path(directory)/run,'wave')
                if any(cache['out_'+n]!=s for n,s in zip(names,outputs)):raise ValueError('Cache changed output')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/free_schrodinger_execution.py','scripts/validate_free_schrodinger_execution.py',
           'tests/physics_ingest/test_free_schrodinger.py','tests/physics_ingest/test_free_schrodinger_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=6,general_argument_preserved=True,
                certificate_verified=True,cache_ast_strings_preserved=True,source_proof=proof,
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','general_argument_preserved','certificate_verified','cache_ast_strings_preserved']}))
