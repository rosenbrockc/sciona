#!/usr/bin/env python3
"""Validate shared residual atoms through catalog graph materialization and the CDG runner."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID,uuid5
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import sympy as sp
from sciona.atoms.electrical import residuals
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.equation_runtime import compile_equation_runtime
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.residual_execution import build_residual_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol-file',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();symbols=args.symbol_file.read_bytes();counts=Counter();errors=[]
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        proofs=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='residual-reuse.v1' AND passed ORDER BY evidence_id FOR SHARE").fetchall()
        with tempfile.TemporaryDirectory(prefix='sciona-residual-execution-') as directory:
            previous=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
            try:
                for index,proof in enumerate(proofs):
                    row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload,v.content_hash FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) JOIN artifact_versions v ON v.version_id=e.version_id WHERE e.version_id=%s FOR SHARE OF e,q,s,v',(proof['version_id'],)).fetchone()
                    fresh=prepare_pdg_evidence(row,symbols)
                    if fresh!=row['evidence_json']['pdg_source_comparison'] or _digest(fresh)!=proof['details']['source_comparison_sha256'] or row['content_hash']!=proof['details']['source_content_hash']:raise ValueError('source proof drift')
                    dims={r['symbol_name']:r['dim_signature'] for r in db.execute('SELECT symbol_name,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE',(row['expression_id'],)).fetchall()}
                    equation=deserialize_expr(row['sympy_srepr'])
                    graph,mapping=build_residual_execution(equation,dims,source_version_id=row['version_id'],source_content_hash=row['content_hash'])
                    digest,nodes,edges=encode_execution_graph(graph)
                    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':str(row['version_id'])} for n in nodes],'cdg_edges':edges},version_id=str(row['version_id']),content_hash=digest,require_execution_envelope=True)
                    rng=np.random.default_rng(20260915)
                    values={name:rng.uniform(-10.,10.,256) for name in sorted(dims)}
                    inputs={argument:values[symbol] for argument,symbol in mapping.items()}
                    run_id='residual-'+str(index)
                    result=await runner.CDGExecutionSession(None,'synthetic-residual',run_id).execute(inputs,cdg=restored)
                    if result['status']!='completed':raise ValueError('runner did not complete')
                    actual=np.load(Path(directory)/run_id/'residual'/'out_residual.npy')
                    compiled=compile_equation_runtime(sp.Eq(sp.Symbol('_validation_residual'),equation.lhs-equation.rhs,evaluate=False))
                    namespace={};exec(compiled.source,namespace)
                    expected=namespace['evaluate'](*(values[name] for name in compiled.argument_symbols))
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
                    maximum=float(np.max(np.abs(actual-expected)));errors.append(maximum)
                    details={'scope':'shared residual implementation and argument mapping through serialized CDG execution; mathematical residual only, no new physical validation or publication approval',
                             'source_content_hash':row['content_hash'],'reuse_proof_evidence_id':str(proof['evidence_id']),
                             'reuse_proof_sha256':_digest(proof['details']),'execution_graph':graph.model_dump(mode='json'),
                             'execution_graph_sha256':digest,'synthetic_cases':256,'maximum_absolute_error':maximum,
                             'reference_runtime_sha256':compiled.source_sha256,
                             'provider_source_sha256':hashlib.sha256(Path(residuals.__file__).read_bytes()).hexdigest(),
                             'implementation_hashes':{path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in ['sciona/physics_ingest/residual_execution.py','sciona/physics_ingest/residual_equivalence.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']}}
                    evidence_id=uuid5(UUID(str(row['version_id'])),'residual-cdg-execution.v1')
                    old=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s FOR UPDATE',(evidence_id,)).fetchone()
                    if old:
                        if old['details']!=details:raise ValueError('existing execution evidence changed')
                        counts['unchanged']+=1
                    else:
                        db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,source_kind,runner_version,details) VALUES(%s,%s,%s,'regression_test',true,'completed','automated','residual-cdg-execution.v1',%s)",(evidence_id,row['artifact_id'],row['version_id'],Jsonb(details)))
                        counts['validated']+=1
                    counts['examined']+=1
            finally:runner.RUNS_DIR=previous
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts),'maximum_absolute_error':max(errors,default=0)},indent=2))


if __name__=='__main__':asyncio.run(main())
