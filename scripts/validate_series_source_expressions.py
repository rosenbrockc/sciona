#!/usr/bin/env python3
"""Validate source equations numerically before advancing symbolic-validation state."""
import argparse
import builtins
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import sympy as sp
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.equation_runtime import compile_equation_runtime
from sciona.physics_ingest.series_scenarios import build_series_scenarios
from sciona.physics_ingest.series_source_review import CURRENT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    symbol_bytes=args.symbol_file.read_bytes();counts=Counter();errors=[]
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        replays=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='pdg-graph-replay.v1' AND passed FOR SHARE").fetchall()
        for replay in replays:
            ids=list(replay['details']['expression_evidence_sha256'])
            rows=db.execute("SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE e.expression_id=ANY(%s::uuid[]) ORDER BY e.expression_id FOR UPDATE OF e FOR SHARE OF q,s",(ids,)).fetchall()
            if len(rows)!=len(ids):raise ValueError('source dependencies missing')
            expressions={};variables={};dimensions={}
            for row in rows:
                if row['review_status']!='needs_human':raise ValueError('human review must be preserved')
                fresh=prepare_pdg_evidence(row,symbol_bytes)
                if fresh!=row['evidence_json'].get('pdg_source_comparison') or _digest(fresh)!=replay['details']['expression_evidence_sha256'][str(row['expression_id'])]:raise ValueError('source evidence is stale')
                expressions[str(row['expression_id'])]=deserialize_expr(fresh['upstream_symbolic']['sympy_srepr'])
                specs=db.execute('SELECT symbol_name,source_symbol,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE',(row['expression_id'],)).fetchall()
                variables[str(row['expression_id'])]=specs
                for spec in specs:
                    old=dimensions.get(spec['source_symbol'])
                    if old is not None and old!=spec['dim_signature']:raise ValueError('inconsistent source dimensions')
                    dimensions[spec['source_symbol']]=spec['dim_signature']
            values=build_series_scenarios(expressions,dimensions)
            for row in rows:
                equation=deserialize_expr(row['sympy_srepr'])
                compiled=compile_equation_runtime(sp.Eq(sp.Symbol('_validation_residual'),equation.lhs-equation.rhs,evaluate=False))
                specs=variables[str(row['expression_id'])]
                inputs={spec['symbol_name']:values[spec['source_symbol']] for spec in specs}
                if set(compiled.argument_symbols)!=set(inputs):raise ValueError('stored-expression runtime interface mismatch')
                def guarded_import(name,*a,**kw):
                    if name!='numpy':raise ValueError('unexpected runtime import')
                    return builtins.__import__(name,*a,**kw)
                namespace={'__builtins__':{**vars(builtins),'__import__':guarded_import}};exec(compiled.source,namespace)
                fn=namespace['evaluate']
                residual=fn(*(inputs[s] for s in compiled.argument_symbols))
                max_error=float(np.max(np.abs(residual)))
                np.testing.assert_allclose(residual,0,rtol=0,atol=1e-10)
                # Deliberately violate one non-current quantity. Perturbing only
                # current would preserve the balance equation and is no control.
                perturbed=next(spec['symbol_name'] for spec in specs if spec['dim_signature']!=CURRENT)
                wrong={**inputs,perturbed:inputs[perturbed]+1.0}
                bad=fn(*(wrong[s] for s in compiled.argument_symbols))
                if not np.all(np.abs(bad)>1e-6):raise ValueError('residual validator failed to detect an inconsistent circuit')
                evidence={
                    'runner_version':'series-source-validation.v1','kind':'equation_residual_validator',
                    'scope':'stored-equation validation on synthetic circuit scenarios; physical regime and human review remain required',
                    'source':compiled.source,'runtime_source_sha256':compiled.source_sha256,
                    'argument_symbols':list(compiled.argument_symbols),'runtime_imports':['numpy'],'no_sympy_runtime':True,'tests_passed':True,
                    'consistent_cases':256,'inconsistent_cases_detected':256,'maximum_absolute_residual':max_error,'absolute_tolerance':1e-10,
                    'source_replay_evidence_id':str(replay['evidence_id']),'source_comparison_sha256':_digest(row['evidence_json']['pdg_source_comparison']),
                    'numpy_version':np.__version__,'sympy_build_version':sp.__version__,
                    'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['scripts/validate_series_source_expressions.py','sciona/physics_ingest/series_scenarios.py','sciona/physics_ingest/equation_runtime.py']},
                }
                updated=dict(row['evidence_json']);old=updated.get('numpy_runtime')
                if old is not None:
                    if old!=evidence or row['validation_status']!='passed':raise ValueError('existing runtime validation conflicts')
                    counts['unchanged']+=1;continue
                if row['validation_status']!='unknown':raise ValueError('existing validation state must be preserved')
                updated['numpy_runtime']=evidence
                db.execute("UPDATE artifact_symbolic_expressions SET validation_status='passed',evidence_json=%s WHERE expression_id=%s",(Jsonb(updated),row['expression_id']))
                counts['validated']+=1;errors.append(max_error)
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'counts':dict(counts),'maximum_absolute_residual':max(errors,default=0)},indent=2))


if __name__=='__main__':main()
