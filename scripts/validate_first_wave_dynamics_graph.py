"""Synthetic serialized graph execution for corrected first-wave dynamics."""
import asyncio
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import sciona.atoms.physics.first_wave_dynamics as provider
from sciona.physics_ingest.first_wave_dynamics_graph import build_first_wave_dynamics_graph
from sciona.physics_ingest.first_wave_dynamics_execution import execute_dynamics
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def run_graph(graph,payload):
    outputs={}
    def capture(directory,node,name,value):
        if name.startswith('out_'): outputs[(node,name[4:])]=value
    with tempfile.TemporaryDirectory(prefix='synthetic-dynamics-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-dynamics','case').execute({'payload':payload},cdg=graph))
    return result,outputs


def validate(root):
    digest,nodes,edges=encode_execution_graph(build_first_wave_dynamics_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert (len(nodes),len(edges))==(1,0)
    assert list(inspect.signature(provider.first_wave_dynamics).parameters)==['payload']
    assert provider.witness_first_wave_dynamics({})=={'kind':'FirstWave.SymbolicAndSampledDynamics'}
    cases=[dict(version=1,mass=2,position={'op':'pow','args':['t',4]},times=[-2,0,1,3]),
           dict(version=1,mass=.25,position={'op':'sin','args':['t']},times=[0,1,2]),
           dict(version=1,mass=7,position={'op':'position','args':[]},times=[])]
    for payload in cases:
        wire=json.loads(json.dumps(payload,allow_nan=False)); original=copy.deepcopy(wire)
        outcome,outputs=run_graph(graph,wire)
        assert outcome['status']=='completed'
        assert outputs[('dynamics','result')]==execute_dynamics(wire)
        assert wire==original
        json.dumps(outputs[('dynamics','result')],allow_nan=False)
    bad=dict(version=1,mass=0,position='t',times=[1])
    try: run_graph(graph,bad)
    except RuntimeError as error: assert 'first_wave_dynamics' in str(error)
    else: raise AssertionError('Runner accepted invalid mass')
    paths=[Path(__file__),*sorted((root/'sciona/physics_ingest').glob('first_wave_dynamics_*.py')),
        root/'docs/reviews/physics_first_wave_dynamics_review.json',root/'docs/reviews/physics_first_wave_dynamics_execution.json']
    return dict(approved=False,synthetic_only=True,checks=dict(provider_contracts=1,serialized_graph_cases=3,runner_rejections=1),
        serialized_graph_sha256=digest,provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        implementation_sha256={str(p.resolve().relative_to(root.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Graph validation only; no served catalog selection or approval yet','Explicit corrected premises; original source remains draft'])

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];report=validate(root)
    (root/'docs/reviews/physics_first_wave_dynamics_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))
