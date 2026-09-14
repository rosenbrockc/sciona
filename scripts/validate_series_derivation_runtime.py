#!/usr/bin/env python3
"""Validate the replayed two-element ohmic series model against circuit equations."""
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID,uuid5
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import sympy as sp
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.equation_runtime import compile_equation_runtime
from sciona.physics_ingest.pdg_evidence import _digest

REFERENCE_URL='https://openstax.org/books/university-physics-volume-2/pages/10-2-resistors-in-series-and-parallel'
REFERENCE_KEY='openstax-university-physics-v2-10-2'
REFERENCE_TITLE='University Physics Volume 2, Section 10.2: Resistors in Series and Parallel'
REGIME='Lumped ohmic resistors connected in one series branch; common branch current and additive voltage drops; nonnegative resistances. No claim for nonlinear devices or alternate circuit topologies.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    reports=[]
    root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        replays=db.execute("SELECT r.* FROM artifact_audit_evidence r JOIN artifact_versions v USING(version_id) JOIN artifacts a ON a.artifact_id=v.artifact_id WHERE r.runner_version='pdg-graph-replay.v1' AND r.passed AND v.is_latest AND a.status='draft' FOR SHARE OF r,v,a").fetchall()
        for replay in replays:
            details=replay['details']
            for name,digest in details['implementation_hashes'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('replay implementation changed')
            version=db.execute('SELECT content_hash FROM artifact_versions WHERE version_id=%s',(replay['version_id'],)).fetchone()
            nodes=db.execute('SELECT * FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id FOR SHARE',(replay['version_id'],)).fetchall()
            edges=db.execute('SELECT * FROM artifact_cdg_edges WHERE version_id=%s ORDER BY source_id,target_id FOR SHARE',(replay['version_id'],)).fetchall()
            projection={'nodes':[{k:str(v) if k=='version_id' else v for k,v in n.items()} for n in nodes], 'edges':[{k:str(v) if k=='version_id' else v for k,v in e.items()} for e in edges]}
            if version['content_hash']!=details['graph_content_hash'] or _digest(projection)!=details['graph_projection_sha256']:raise ValueError('graph changed since replay')
            terminals=details['terminal_expression_ids']
            if len(terminals)!=1:raise ValueError('one terminal equation is required')
            terminal=next(s for s in details['steps'] if s['output_expression_id']==terminals[0])
            equation=deserialize_expr(terminal['computed_srepr'])
            conditions=[deserialize_expr(s) for s in details['required_conditions']]
            if len(conditions)!=1 or not isinstance(conditions[0],sp.Unequality) or conditions[0].rhs!=0 or not isinstance(conditions[0].lhs,sp.Symbol):raise ValueError('unexpected derivation domain')
            current=conditions[0].lhs
            resistors=sorted(equation.rhs.free_symbols-{current},key=sp.srepr)
            if len(resistors)!=2 or sp.cancel(equation.rhs-sum(resistors))!=0:raise ValueError('terminal is not the two-element series equation')
            # Model-specific identification must be supported by source dimensions,
            # not solely by an algebraically similar equation from another domain.
            dims=db.execute('SELECT source_symbol,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s',(terminals[0],)).fetchall()
            by_symbol={r['source_symbol']:r['dim_signature'] for r in dims}
            from sciona.ghost.dimensions import DimensionalSignature
            resistance=DimensionalSignature(M=1,L=2,T=-3,I=-2).to_compact()
            if any(by_symbol.get(str(s))!=resistance for s in [equation.lhs,*resistors]):raise ValueError('source variables are not resistance quantities')
            current_dims=db.execute("SELECT DISTINCT dim_signature FROM artifact_symbolic_variables WHERE source_symbol=%s AND expression_id=ANY(%s::uuid[])",(str(current),list(details['expression_evidence_sha256']))).fetchall()
            if {r['dim_signature'] for r in current_dims}!={DimensionalSignature(I=1).to_compact()}:raise ValueError('source current dimension is not established')
            compiled=compile_equation_runtime(equation,[*conditions,*(sp.Ge(r,0) for r in resistors)])
            def guarded_import(name,*args,**kwargs):
                if name!='numpy':raise ValueError('unexpected runtime import')
                return builtins.__import__(name,*args,**kwargs)
            namespace={'__builtins__':{**vars(builtins),'__import__':guarded_import}}
            exec(compiled.source,namespace)
            evaluate=namespace['evaluate']
            rng=np.random.default_rng(20260910)
            first=10**rng.uniform(-3,4,256);second=10**rng.uniform(-3,4,256)
            currents=rng.uniform(0.01,2,256)*rng.choice([-1,1],256)
            def call(r1,r2,i):
                values={str(resistors[0]):r1,str(resistors[1]):r2,str(current):i}
                return evaluate(*(values[s] for s in compiled.argument_symbols))
            actual=call(first,second,currents)
            matrix=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,-1.,1.]])
            voltages=np.linalg.solve(matrix,np.stack([currents*first,currents*second,np.zeros(256)]))
            expected=voltages[2]/currents
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(call(second,first,currents),actual,rtol=1e-12)
            np.testing.assert_allclose(call(first,second,-2*currents),actual,rtol=1e-12)
            for invalid in [(first,second,np.zeros(256)),(-first,second,currents)]:
                try:call(*invalid)
                except ValueError:pass
                else:raise ValueError('required physical/algebraic guard was bypassed')
            report={
                'scope':'compiled terminal equation under source-supported model assumptions; human regime review still required',
                'replay_evidence_id':str(replay['evidence_id']), 'replay_details_sha256':_digest(details),
                'runtime_source':compiled.source,'runtime_source_sha256':compiled.source_sha256,
                'argument_symbols':list(compiled.argument_symbols),'output_symbol':compiled.output_symbol,
                'runtime_imports':['numpy'],'no_sympy_runtime':True,'tests_passed':True,
                'synthetic_case_count':256,'max_relative_error':float(np.max(np.abs(actual-expected)/np.abs(expected))),
                'checks':['Kirchhoff linear-system reference','resistor permutation','current scaling and reversal','zero-current rejection','negative-resistance rejection'],
                'physical_regime':REGIME,'scientific_reference':REFERENCE_URL,
                'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['scripts/validate_series_derivation_runtime.py','sciona/physics_ingest/equation_runtime.py']},
            }
            evidence_id=uuid5(UUID(str(replay['version_id'])),'series-runtime:'+_digest(report))
            old=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s',(evidence_id,)).fetchone()
            if old:
                if old['details']!=report:raise ValueError('runtime evidence drifted')
                bound=db.execute('SELECT validity_statement FROM artifact_validity_bounds WHERE bound_id=%s',(uuid5(evidence_id,'physical-regime'),)).fetchone()
                reference=db.execute('SELECT url,verified FROM artifact_references WHERE artifact_id=%s AND ref_key=%s',(replay['artifact_id'],REFERENCE_KEY)).fetchone()
                if not bound or bound['validity_statement']!=REGIME or not reference or reference!={'url':REFERENCE_URL,'verified':True}:raise ValueError('runtime regime or reference evidence drifted')
                reports.append({'unchanged':True});continue
            db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,details,source_kind,runner_version) VALUES(%s,%s,%s,'golden_eval',true,'completed',%s,'automated','series-terminal-runtime.v1')",(evidence_id,replay['artifact_id'],replay['version_id'],Jsonb(report)))
            prior_reference=db.execute('SELECT url,title FROM references_registry WHERE ref_id=%s',(REFERENCE_KEY,)).fetchone()
            if prior_reference and prior_reference!={'url':REFERENCE_URL,'title':REFERENCE_TITLE}:raise ValueError('reference identity conflicts')
            prior_link=db.execute('SELECT url,verified FROM artifact_references WHERE artifact_id=%s AND ref_key=%s',(replay['artifact_id'],REFERENCE_KEY)).fetchone()
            if prior_link and prior_link!={'url':REFERENCE_URL,'verified':True}:raise ValueError('existing graph reference conflicts')
            db.execute("INSERT INTO references_registry(ref_id,ref_type,title,url) VALUES(%s,'book',%s,%s) ON CONFLICT(ref_id) DO NOTHING",(REFERENCE_KEY,REFERENCE_TITLE,REFERENCE_URL))
            db.execute("INSERT INTO artifact_references(artifact_id,ref_id,ref_key,title,url,relevance_note,source,verified,confidence) VALUES(%s,%s,%s,%s,%s,%s,'llm_extracted',true,'high') ON CONFLICT(artifact_id,ref_key) DO NOTHING",(replay['artifact_id'],REFERENCE_KEY,REFERENCE_KEY,REFERENCE_TITLE,REFERENCE_URL,'Publication and supporting section inspected: same series current, additive voltage drops, Ohm law and additive resistance. Verification does not constitute human approval of this graph.'))
            db.execute("INSERT INTO artifact_validity_bounds(bound_id,artifact_id,version_id,scope,bound_kind,validity_statement,evidence_ref_key,review_status,metadata) VALUES(%s,%s,%s,'version','regime',%s,%s,'needs_human',%s)",(uuid5(evidence_id,'physical-regime'),replay['artifact_id'],replay['version_id'],REGIME,REFERENCE_KEY,Jsonb({'runtime_evidence_id':str(evidence_id),'scientific_reference':REFERENCE_URL})))
            reports.append({'validated':True,'synthetic_cases':256,'max_relative_error':report['max_relative_error']})
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'results':reports},indent=2))


if __name__=='__main__':main()
