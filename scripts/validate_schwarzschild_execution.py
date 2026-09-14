#!/usr/bin/env python3
"""Validate full radius execution against Decimal arithmetic and pinned algebra."""
import argparse
import asyncio
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import sympy as sp
from sciona.atoms.physical_quantities import schwarzschild_radius as provider
from sciona.ghost.symbolic import deserialize_expr
from sciona.physics_ingest.schwarzschild_execution import build_schwarzschild_execution,SOURCE_VERSION,SOURCE_HASH
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_physics_real_power_replay import validate as validate_source


def decimal_reference(masses,g,c):
    with localcontext() as ctx:
        ctx.prec=100
        return np.asarray([float(2*Decimal.from_float(float(g))*Decimal.from_float(float(m))/Decimal.from_float(float(c))**2) for m in masses.flat]).reshape(masses.shape)


async def validate(root,symbol_file,rule_file):
    source=validate_source(root,symbol_file,rule_file)
    selected=[g for g in source['graphs'] if g['version_id']==SOURCE_VERSION and g['content_hash']==SOURCE_HASH]
    if len(selected)!=1:raise ValueError('Exact source proof missing')
    proof=selected[0]
    terminal=deserialize_expr(proof['steps'][-1]['computed_srepr'])
    R,G,M,c=map(sp.Symbol,['pdg0004518','pdg0006277','pdg0005156','pdg0004567'])
    from sciona.physics_ingest.pdg_symbols import load_pinned_pdg_scalars
    from sciona.ghost.dimensions import DimensionalSignature
    # validate_source already checks these bytes against ingestion pins.
    symbol_bytes=symbol_file.read_bytes()
    definitions=load_pinned_pdg_scalars(symbol_bytes,hashlib.sha256(symbol_bytes).hexdigest())
    expected_dimensions={R:DimensionalSignature(L=1),G:DimensionalSignature(L=3,M=-1,T=-2),
        M:DimensionalSignature(M=1),c:DimensionalSignature(L=1,T=-1)}
    for symbol,dimension in expected_dimensions.items():
        if definitions[str(symbol)].dimension!=dimension:raise ValueError('Source quantity dimension mismatch')
    if definitions[str(G)].latex!='G' or definitions[str(M)].latex!='m' or definitions[str(c)].latex!='c':
        raise ValueError('Source quantity identity mismatch')
    if terminal.lhs!=R or sp.cancel(terminal.rhs-2*G*M/c**2)!=0:
        raise ValueError('Source conclusion differs from implemented formula')
    graph=build_schwarzschild_execution()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if restored!=graph:raise ValueError('Graph serialization differs')
    cases=[(np.logspace(-100,100,201),6.6743e-11,299792458.),
        (np.array([[1.,2.],[4.,8.]]),3.,2.),
        (np.asarray(1e308),1e308,1e308),
        (np.asarray(1e-300),1e-300,1e-300),
        (np.asarray(np.nextafter(0.,1.)),1.,1.)]
    maximum=0.;executed=0
    with tempfile.TemporaryDirectory(prefix='sciona-radius-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for index,(masses,g,light) in enumerate(cases):
                run='case-'+str(index)
                result=await runner.CDGExecutionSession(None,'synthetic-radius',run).execute(
                    dict(mass_kg=masses,gravitational_constant=g,speed_of_light=light),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                actual=np.load(Path(directory)/run/'radius'/'out_radius_metres.npy')
                expected=decimal_reference(masses,g,light)
                if actual.shape!=masses.shape:raise ValueError('Shape changed')
                np.testing.assert_array_max_ulp(actual,expected,maxulp=3)
                maximum=max(maximum,float(np.max(np.abs(actual-expected)/np.spacing(expected))))
                executed+=1
        finally:runner.RUNS_DIR=prior
    # Check each computed proof equality and every inherited domain condition
    # on exact positive rational assignments. Original and renamed radii are
    # both the solution; the root escape speed and c share the positive branch.
    for mass in [sp.Rational(1,8),sp.Integer(1),sp.Integer(123)]:
        mapping={M:mass,G:sp.Rational(3,2),c:sp.Integer(7),R:3*mass/49,
            sp.Symbol('pdg0002530'):3*mass/49,sp.Symbol('pdg0008656'):sp.Integer(7)}
        for step in proof['steps']:
            equation=deserialize_expr(step['computed_srepr'])
            if sp.simplify((equation.lhs-equation.rhs).subs(mapping))!=0:raise ValueError('Intermediate equality failed')
        for condition in proof['required_conditions']:
            if sp.simplify(deserialize_expr(condition).subs(mapping))!=sp.true:raise ValueError('Domain condition failed')
    files=['sciona/physics_ingest/schwarzschild_execution.py','scripts/validate_schwarzschild_execution.py',
        'tests/physics_ingest/test_schwarzschild_radius.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py','sciona/physics_ingest/pdg_symbols.py']
    return dict(graph_digest=digest,execution_graph=graph.model_dump(mode='json'),full_runner_cases=executed,
        synthetic_numeric_cases=sum(m.size for m,_,_ in cases),maximum_ulp_error=maximum,ulp_tolerance=3,
        exact_rational_proof_cases=3,source_proof=source,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        limitations=['Formula and conditional proof realization; not a general-relativistic field-equation solver.',
            'Physical horizon interpretation requires caller-established Schwarzschild regime; no black-hole classification.',
            'Constants and units are explicit caller inputs; positive subnormal outputs can have reduced relative precision.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_numeric_cases','maximum_ulp_error','exact_rational_proof_cases']}))
