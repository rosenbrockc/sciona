#!/usr/bin/env python3
"""Validate the provider-backed execution realization against canonical runtime evidence."""
import argparse
import asyncio
import builtins
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID,uuid5
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.physics_ingest.series_execution import build_series_execution_cdg
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.visualizer import runner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();results=[]
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        records=db.execute("SELECT e.* FROM artifact_audit_evidence e JOIN artifact_versions v USING(version_id) JOIN artifacts a ON a.artifact_id=v.artifact_id WHERE e.runner_version='series-terminal-runtime.v1' AND e.passed AND v.is_latest AND a.status='draft' FOR SHARE OF e,v,a").fetchall()
        for record in records:
            details=record['details'];source=details['runtime_source']
            if hashlib.sha256(source.encode()).hexdigest()!=details['runtime_source_sha256']:raise ValueError('canonical runtime hash mismatch')
            for name,digest in details['implementation_hashes'].items():
                if hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()!=digest:raise ValueError('canonical runtime evidence is stale')
            graph=build_series_execution_cdg(source_version_id=str(record['version_id']),runtime_evidence_id=str(record['evidence_id']))
            # Locate resistance/current arguments by the source-backed SI signatures.
            variables=db.execute('SELECT DISTINCT source_symbol,dim_signature FROM artifact_symbolic_variables WHERE source_symbol=ANY(%s)',(details['argument_symbols'],)).fetchall()
            from sciona.ghost.dimensions import DimensionalSignature
            current_dim=DimensionalSignature(I=1).to_compact()
            resistance_dim=DimensionalSignature(M=1,L=2,T=-3,I=-2).to_compact()
            mapping={}
            for r in variables:
                if r['source_symbol'] in mapping and mapping[r['source_symbol']]!=r['dim_signature']:raise ValueError('ambiguous source dimensions')
                mapping[r['source_symbol']]=r['dim_signature']
            currents=[s for s in details['argument_symbols'] if mapping.get(s)==current_dim]
            resistors=sorted(s for s in details['argument_symbols'] if mapping.get(s)==resistance_dim)
            if len(currents)!=1 or len(resistors)!=2:raise ValueError('unexpected numerical interface')
            rng=np.random.default_rng(20260911);first=10**rng.uniform(-3,4,256);second=10**rng.uniform(-3,4,256);current=rng.uniform(.01,2,256)
            values={resistors[0]:first,resistors[1]:second,currents[0]:current}
            def guarded_import(name,*a,**kw):
                if name!='numpy':raise ValueError('unexpected runtime import')
                return builtins.__import__(name,*a,**kw)
            ns={'__builtins__':{**vars(builtins),'__import__':guarded_import}};exec(source,ns)
            expected=ns['evaluate'](*(values[s] for s in details['argument_symbols']))
            old_runs=runner.RUNS_DIR
            try:
                with tempfile.TemporaryDirectory(prefix='sciona-series-execution-') as temporary:
                    runner.RUNS_DIR=Path(temporary)
                    session=runner.CDGExecutionSession(None,'synthetic-series','validation')
                    outcome=asyncio.run(session.execute({'resistance_a':first.tolist(),'resistance_b':second.tolist(),'current':current.tolist()},cdg=graph))
                    if outcome['status']!='completed':raise ValueError('CDG execution did not complete')
                    actual=np.load(Path(temporary)/'validation'/'series_equivalent_resistance'/'out_equivalent_resistance.npy')
                    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            finally:runner.RUNS_DIR=old_runs
            root=Path(__file__).resolve().parents[1]
            report={'scope':'provider-backed numerical realization; source derivation and dependency reviews remain mandatory',
                    'source_runtime_evidence_id':str(record['evidence_id']),'source_runtime_details_sha256':_digest(details),
                    'execution_cdg':graph.model_dump(mode='json'),'synthetic_cases':256,
                    'max_relative_error':float(np.max(np.abs(actual-expected)/np.abs(expected))),
                    'provider_implementation_sha256':hashlib.sha256(Path(inspect.getfile(series_resistance)).read_bytes()).hexdigest(),
                    'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['sciona/physics_ingest/series_execution.py','sciona/visualizer/runner.py','scripts/validate_series_execution.py']}}
            evidence_id=uuid5(UUID(str(record['version_id'])),'series-cdg-execution:'+_digest(report))
            old=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s',(evidence_id,)).fetchone()
            if old:
                if old['details']!=report:raise ValueError('execution evidence drift')
                results.append({'unchanged':True});continue
            db.execute("INSERT INTO artifact_audit_evidence(evidence_id,artifact_id,version_id,audit_type,passed,status,details,source_kind,runner_version) VALUES(%s,%s,%s,'regression_test',true,'completed',%s,'automated','series-cdg-execution.v1')",(evidence_id,record['artifact_id'],record['version_id'],Jsonb(report)))
            results.append({'executed':True,'synthetic_cases':256,'max_relative_error':report['max_relative_error']})
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'results':results},indent=2))


if __name__=='__main__':main()
