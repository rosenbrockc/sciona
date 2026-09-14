from uuid import uuid4
import numpy as np
import pytest
from sciona.atoms.electrical import residuals
from sciona.physics_ingest.residual_execution import residual_templates,build_residual_execution
from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import CDGExecutionSession


@pytest.mark.asyncio
@pytest.mark.parametrize('index',range(4))
async def test_shared_residual_through_materialized_cdg(index,tmp_path,monkeypatch):
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    fn,equation=residual_templates()[index]
    dims=REGISTRY[fn.__module__+'.'+fn.__name__]['dim_signature']
    graph,mapping=build_residual_execution(equation,{k:v.to_compact() for k,v in dims.items() if k!='return'},source_version_id=uuid4(),source_content_hash='synthetic')
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'test'} for n in nodes],'cdg_edges':edges},version_id='test',content_hash=digest,require_execution_envelope=True)
    values=[{'voltage':[8.,13.],'current':[2.,3.],'resistance':4.},
            {'total_voltage':[8.,9.],'voltage_a':3.,'voltage_b':5.},
            {'total_resistance':[8.,9.],'resistance_a':3.,'resistance_b':5.},
            {'current':[2.,3.],'total_resistance':[8.,9.],'resistance_a':3.,'resistance_b':5.}][index]
    result=await CDGExecutionSession(None,'synthetic','residual-check').execute(values,cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_allclose(np.load(tmp_path/'residual-check'/'residual'/'out_residual.npy'),[0.,3.] if index==3 else [0.,1.])


@pytest.mark.parametrize('fn,args',[(residuals.ohmic_voltage_residual,(8.,2.,4.)),(residuals.series_voltage_residual,(8.,3.,5.)),(residuals.series_resistance_residual,(8.,3.,5.)),(residuals.series_ohmic_balance_residual,(2.,8.,3.,5.))])
def test_residual_boundaries(fn,args):
    for bad in [float('nan'),float('inf'),True,'1',1j]:
        with pytest.raises(ValueError):fn(bad,*args[1:])
    with pytest.raises(ValueError):fn(np.ones(2),np.ones(3),*args[2:])
    arrays=[np.array([x,x]) for x in args];copies=[x.copy() for x in arrays]
    assert fn(*arrays).dtype==np.float64
    for array,copy in zip(arrays,copies):np.testing.assert_array_equal(array,copy)
