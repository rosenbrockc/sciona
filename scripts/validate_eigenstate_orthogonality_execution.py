"""Validate exact complex outputs through graph serialization and full runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import sympy as sp
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import eigenstate_orthogonality as provider
from sciona.physics_ingest.eigenstate_orthogonality_execution import SOURCE_VERSION,INPUT_NAMES,OUTPUT_NAMES,build_eigenstate_orthogonality_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_eigenstate_orthogonality_source import validate as validate_source


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_eigenstate_orthogonality_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph serialization changed')
    a=sp.Symbol('a',real=True)
    cases=[([[2,sp.I],[-sp.I,2]],[sp.I,1],[-sp.I,1],3,1),
           ([[3,0],[0,3]],[sp.I,2],[1,sp.I],3,3),
           ([[-2]],[2+sp.I],[3-sp.I],-2,-2),
           ([[0,sp.sqrt(2)],[sp.sqrt(2),0]],[1,1],[1,-1],sp.sqrt(2),-sp.sqrt(2)),
           ([[a,0],[0,a]],[1,2],[3,4],a,a),
           ([[0]],[7],[11],0,0)]
    with tempfile.TemporaryDirectory(prefix='sciona-eigenstates-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,(A,u,v,a,b) in enumerate(cases):
                source=[sp.Tuple(*(sp.Tuple(*row) for row in A)),sp.Tuple(*u),sp.Tuple(*v),sp.sympify(a),sp.sympify(b)]
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-eigenstates',run).execute(dict(zip(INPUT_NAMES,map(sp.srepr,source))),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                # Independent scalar sums, not provider matrix multiplication.
                overlap=sum(sp.conjugate(x)*y for x,y in zip(u,v))
                element=sum(sp.conjugate(u[j])*A[j][k]*v[k] for j in range(len(u)) for k in range(len(v)))
                expected=[overlap,element,(b-a)*overlap]
                for name,wanted in zip(OUTPUT_NAMES,expected):
                    record=json.loads((Path(directory)/run/'eigenstates'/('out_'+name+'.json')).read_text())
                    if record['type']!='str' or sp.simplify(provider._checked_parse(record['value'])-wanted)!=0:
                        raise ValueError('Exact output mismatch')
                cache=runner.load_cached_outputs(Path(directory)/run,'eigenstates')
                if any(not isinstance(cache['out_'+name],str) for name in OUTPUT_NAMES):raise ValueError('Cache lost AST strings')
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/eigenstate_orthogonality_execution.py','scripts/validate_eigenstate_orthogonality_execution.py',
           'tests/physics_ingest/test_eigenstate_orthogonality.py','tests/physics_ingest/test_eigenstate_orthogonality_proof.py',
           'sciona/physics_ingest/source_symbolic.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),
                exact_scalar_sum_oracle=True,nonzero_degenerate_overlap_preserved=True,cache_ast_strings_preserved=True,
                source_proof=proof,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','exact_scalar_sum_oracle','nonzero_degenerate_overlap_preserved','cache_ast_strings_preserved']}))
