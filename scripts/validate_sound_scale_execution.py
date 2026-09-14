#!/usr/bin/env python3
"""Execute conditional sound-speed scaling through the serialized CDG runner."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.physics import sound_scale as provider
from sciona.physics_ingest.sound_scale_execution import SOURCE_VERSION,build_sound_scale_execution
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.validate_sound_scale_source import validate as validate_source
from tests.physics_ingest.test_sound_scale import reference


async def validate(root,symbol_file,rule_file,expression_file):
    proof=validate_source(root,symbol_file,rule_file,expression_file)
    graph=build_sound_scale_execution();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':SOURCE_VERSION} for n in nodes],'cdg_edges':edges},
        version_id=SOURCE_VERSION,content_hash=digest,require_execution_envelope=True)
    if graph!=restored:raise ValueError('Graph serialization differs')
    rng=np.random.default_rng(909)
    cases=[(np.exp(rng.uniform(0,20,size=(2,3,4))),)+tuple(np.exp(rng.uniform(-80,80,size=(2,3,4))) for _ in range(6))]
    cases += [tuple(np.array(v) for v in row) for row in [
        (1.,1.,1.,2.,1.,1.,0.), (4.,1.,1.,2.,1.,4.,3.),
        (1.,1e308,1e-308,2.,1.,1.,0.), (1.,1.,1.,1e308,1e308,1.,1.),
        (1.,1.,np.nextafter(0.,1.),2.,1.,1.,0.)]]
    with tempfile.TemporaryDirectory(prefix='sciona-sound-scale-runner-') as directory:
        prior=runner.RUNS_DIR;runner.RUNS_DIR=Path(directory)
        try:
            for i,args in enumerate(cases):
                run='case-'+str(i)
                result=await runner.CDGExecutionSession(None,'synthetic-sound-scale',run).execute(
                    dict(zip(["atomic_mass","fine_structure","light_speed","electron_mass","proton_mass","bulk_prefactor","shear_to_bulk"],args)),cdg=restored)
                if result['status']!='completed':raise ValueError('Full runner failed')
                expected_values=[reference(*row) for row in zip(*(a.flat for a in args))]
                expected_arrays=[np.array([e[j] for e in expected_values]).reshape(args[0].shape) for j in range(5)]
                for name,expected in zip(['speed_estimate','upper_scale','bulk_model_speed','longitudinal_model_speed','shear_factor'],expected_arrays):
                    actual=np.load(Path(directory)/run/('scaling/out_'+name+'.npy'))
                    np.testing.assert_array_equal(actual,expected)
        finally:runner.RUNS_DIR=prior
    files=['sciona/physics_ingest/sound_scale_execution.py','scripts/validate_sound_scale_execution.py',
           'tests/physics_ingest/test_sound_scale.py','tests/physics_ingest/test_sound_scale_proof.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    return dict(approved=False,synthetic_only=True,graph_digest=digest,full_runner_cases=len(cases),synthetic_states=sum(args[0].size for args in cases),
                maximum_ulp_error=0,source_proof=proof,provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['symbol-file','rule-file','expression-file','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(validate(Path(__file__).resolve().parents[1],args.symbol_file,args.rule_file,args.expression_file))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['full_runner_cases','synthetic_states','maximum_ulp_error']}))
