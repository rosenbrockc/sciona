from uuid import uuid4
import numpy as np
import pytest
from sciona.atoms.electrical.series_resistance import series_resistance
from sciona.physics_ingest.series_execution import build_series_execution_cdg,PRIMITIVE
from sciona.visualizer.runner import CDGExecutionSession


@pytest.mark.asyncio
async def test_real_primitive_executes_through_cdg_ports(tmp_path,monkeypatch):
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    graph=build_series_execution_cdg(source_version_id=str(uuid4()),runtime_evidence_id=str(uuid4()))
    assert graph.metadata['publication_status']=='draft'
    assert graph.metadata['requires_source_dependency_review']
    assert graph.nodes[0].matched_primitive==PRIMITIVE
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**r,'version_id':'selected'} for r in nodes],'cdg_edges':edges},version_id='selected',content_hash=digest,require_execution_envelope=True)
    session=CDGExecutionSession(None,'synthetic-series','series-check')
    result=await session.execute({'resistance_a':[2.,5.],'resistance_b':[3.,7.],'current':[1.,-1.]},cdg=graph)
    assert result['status']=='completed'
    output=np.load(tmp_path/'series-check'/'series_equivalent_resistance'/'out_equivalent_resistance.npy')
    np.testing.assert_array_equal(output,[5.,12.])
    with pytest.raises(RuntimeError,match='nonzero'):
        await session.execute({'resistance_a':[2.],'resistance_b':[3.],'current':[0.]},cdg=graph)
