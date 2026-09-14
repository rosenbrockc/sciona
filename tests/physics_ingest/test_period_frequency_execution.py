import numpy as np
import pytest
from sciona.atoms.physical_quantities.period_frequency import frequency_from_period
from sciona.physics_ingest.period_frequency_execution import build_period_frequency_execution
from sciona.visualizer.runner import CDGExecutionSession


@pytest.mark.parametrize('invalid',[0.,-1.,np.nan,np.inf,True,1j,'1'])
def test_period_domain(invalid):
    with pytest.raises(ValueError):frequency_from_period(invalid)


def test_frequency_shape_nonmutation_and_overflow():
    periods=np.array([[.25,.5],[1.,2.]])
    original=periods.copy()
    np.testing.assert_array_equal(frequency_from_period(periods),[[4.,2.],[1.,.5]])
    np.testing.assert_array_equal(periods,original)
    with pytest.raises(FloatingPointError):frequency_from_period(np.nextafter(0.,1.))


@pytest.mark.asyncio
async def test_materialized_period_graph_executes(tmp_path,monkeypatch):
    monkeypatch.setattr('sciona.visualizer.runner.RUNS_DIR',tmp_path)
    from sciona.services.execution_graph_codec import encode_execution_graph
    from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
    graph=build_period_frequency_execution(source_version_id='synthetic',replay_evidence_id='synthetic')
    digest,nodes,edges=encode_execution_graph(graph)
    graph=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'synthetic'} for n in nodes],'cdg_edges':edges},version_id='synthetic',content_hash=digest,require_execution_envelope=True)
    result=await CDGExecutionSession(None,'synthetic-period','period-test').execute({'period_seconds':[.25,.5,2.]},cdg=graph)
    assert result['status']=='completed'
    np.testing.assert_array_equal(np.load(tmp_path/'period-test'/'frequency'/'out_frequency_hz.npy'),[4.,2.,.5])
