#!/usr/bin/env python3
"""Replay and execute a frequency/period derivation against independent synthetic measurements."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid5
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import sympy as sp
from sciona.atoms.physical_quantities import period_frequency
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence,_digest
from sciona.physics_ingest.period_frequency_execution import build_period_frequency_execution
from sciona.physics_ingest.period_frequency_validation import periodic_signal_measurements
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.import_residual_execution_drafts import ensure_row

REFERENCE='openstax-university-physics-v1-15-1'
URL='https://openstax.org/books/university-physics-volume-1/pages/15-1-simple-harmonic-motion'


async def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--symbol-file',required=True,type=Path);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    symbols=args.symbol_file.read_bytes();root=Path(__file__).resolve().parents[1]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        records=db.execute("SELECT e.*,v.content_hash FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) WHERE e.runner_version='pdg-source-graph-replay.v2' AND e.passed FOR SHARE OF e,v").fetchall()
        selected=[]
        for record in records:
            for path,digest in record['details']['implementation_hashes'].items():
                if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('source replay implementation drift')
            if record['content_hash']!=record['details']['content_hash']:raise ValueError('source graph hash drift')
            for key,expected in record['details']['expression_evidence_sha256'].items():
                row=db.execute('SELECT e.*,q.source_payload,q.snapshot_id,s.payload AS snapshot_payload FROM artifact_symbolic_expressions e JOIN physics_equation_candidates q USING(candidate_id) JOIN physics_ingest_snapshots s USING(snapshot_id) WHERE e.expression_id=%s FOR SHARE OF e,q,s',(key,)).fetchone()
                fresh=prepare_pdg_evidence(row,symbols)
                if fresh!=row['evidence_json']['pdg_source_comparison'] or _digest(fresh)!=expected:raise ValueError('source expression evidence drift')
                if key in record['details']['terminal_expression_ids']:
                    equation=deserialize_expr(row['sympy_srepr'])
                    variables=db.execute('SELECT symbol_name,source_symbol,dim_signature FROM artifact_symbolic_variables WHERE expression_id=%s FOR SHARE',(key,)).fetchall()
                    dims={r['symbol_name']:r['dim_signature'] for r in variables}
                    periods=[s for s in equation.free_symbols if dims.get(str(s))=='T1']
                    if isinstance(equation,sp.Equality) and isinstance(equation.lhs,sp.Symbol) and dims.get(str(equation.lhs))=='T-1' and len(periods)==1 and equation.rhs==1/periods[0]:
                        guaranteed={sp.srepr(sp.Ne(sp.Symbol(v['source_symbol']),0,evaluate=False)) for v in variables if v['symbol_name'] in {str(equation.lhs),str(periods[0])}}
                        if not set(record['details']['required_conditions'])<=guaranteed:raise ValueError('implementation does not cover all replay conditions')
                        selected.append(record)
        if len(selected)!=1:raise ValueError('one exact frequency-period derivation required')
        record=selected[0];graph=build_period_frequency_execution(source_version_id=record['version_id'],replay_evidence_id=record['evidence_id'])
        digest,nodes,edges=encode_execution_graph(graph)
        restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':str(record['version_id'])} for n in nodes],'cdg_edges':edges},version_id=str(record['version_id']),content_hash=digest,require_execution_envelope=True)
        periods,expected=periodic_signal_measurements()
        with tempfile.TemporaryDirectory(prefix='sciona-period-validation-') as directory:
            prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
            try:
                result=await runner.CDGExecutionSession(None,'synthetic-period','period-validation').execute({'period_seconds':periods},cdg=restored)
                if result['status']!='completed':raise ValueError('execution failed')
                actual=np.load(Path(directory)/'period-validation'/'frequency'/'out_frequency_hz.npy')
            finally:runner.RUNS_DIR=prior
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=0)
        maximum=float(np.max(np.abs(actual-expected)/expected))
        details={'scope':'source derivation numerical realization in the positive periodic-motion regime; no period estimation or angular-frequency claim',
                 'source_content_hash':record['content_hash'],'source_replay_evidence_id':str(record['evidence_id']),'source_replay_sha256':_digest(record['details']),
                 'execution_graph':graph.model_dump(mode='json'),'execution_graph_sha256':digest,
                 'synthetic_cases':len(periods),'measurement_methods':['interpolated rising-zero-crossing period','independent FFT spectral peak'],
                 'maximum_relative_measurement_error':maximum,'relative_tolerance':1e-6,'sample_rate_hz':8192,'samples_per_case':8192,
                 'reference':URL,'domain':'finite positive cycle duration in seconds; finite positive ordinary frequency in hertz',
                 'provider_source_sha256':hashlib.sha256(Path(period_frequency.__file__).read_bytes()).hexdigest(),
                 'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['sciona/physics_ingest/period_frequency_execution.py','sciona/physics_ingest/period_frequency_validation.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']}}
        evidence_id=uuid5(record['version_id'],'period-frequency-execution.v1')
        created=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},{'evidence_id':evidence_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'audit_type':'golden_eval','passed':True,'status':'completed','source_kind':'automated','runner_version':'period-frequency-execution.v1','details':Jsonb(details)})
        title='OpenStax University Physics Volume 1: Simple Harmonic Motion'
        ensure_row(db,'references_registry',{'ref_id':REFERENCE},{'ref_id':REFERENCE,'ref_type':'book','title':title,'url':URL})
        ensure_row(db,'artifact_references',{'artifact_id':record['artifact_id'],'ref_key':REFERENCE},{'artifact_id':record['artifact_id'],'ref_id':REFERENCE,'ref_key':REFERENCE,'title':title,'url':URL,'source':'llm_extracted','verified':True,'confidence':'high','relevance_note':'Section 15.1 defines positive period and ordinary frequency and their reciprocal relation; source verification is not human certification.'})
        bound_id=uuid5(evidence_id,'positive-periodic-domain')
        ensure_row(db,'artifact_validity_bounds',{'bound_id':bound_id},{'bound_id':bound_id,'artifact_id':record['artifact_id'],'version_id':record['version_id'],'scope':'version','bound_kind':'regime','validity_statement':details['domain'],'evidence_ref_key':REFERENCE,'review_status':'automated_pass','metadata':Jsonb({'execution_evidence_id':str(evidence_id),'scope':'automated source-supported regime; user supplies an applicable cycle duration'})})
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'created':created,'synthetic_cases':len(periods),'maximum_relative_measurement_error':maximum},indent=2))


if __name__=='__main__':asyncio.run(main())
