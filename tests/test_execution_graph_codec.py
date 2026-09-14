from copy import deepcopy
from uuid import uuid4
import pytest
from sciona.physics_ingest.series_execution import build_series_execution_cdg
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg


def graph():return build_series_execution_cdg(source_version_id=str(uuid4()),runtime_evidence_id=str(uuid4()))


def test_execution_graph_retains_ports_dimensions_and_provenance():
    g=graph();digest,nodes,edges=encode_execution_graph(g)
    restored=decode_execution_graph(nodes,edges,digest)
    assert restored==g
    assert len(restored.nodes[0].inputs)==3
    assert restored.nodes[0].inputs[0].dim_signature
    assert restored.metadata['requires_source_dependency_review']


@pytest.mark.parametrize('change',['hash','port','primitive','missing_node'])
def test_execution_projection_or_snapshot_drift_rejected(change):
    import json
    digest,nodes,edges=encode_execution_graph(graph())
    if change=='hash':digest='wrong'
    elif change=='port':
        payload=json.loads(nodes[0]['type_signature']);payload['graph']['nodes'][0]['inputs']=[];nodes[0]['type_signature']=json.dumps(payload)
    elif change=='primitive':nodes[0]['matched_primitive']='other'
    elif change=='missing_node':nodes.append(deepcopy(nodes[0]))
    with pytest.raises(ValueError):decode_execution_graph(nodes,edges,digest)


def test_catalog_materialization_selects_exact_version():
    g=graph();digest,nodes,edges=encode_execution_graph(g)
    old={'version_id':'old','node_id':'legacy','name':'Old','status':'atomic','matched_primitive':'wrong'}
    document={'artifact':{},'cdg_nodes':[old,*[{**r,'version_id':'new'} for r in nodes]],'cdg_edges':[{'version_id':'old','source_id':'legacy','target_id':'legacy','output_name':'x','input_name':'x'}]}
    assert _artifact_document_to_cdg(document,version_id='new',content_hash=digest)==g
    with pytest.raises(ValueError):_artifact_document_to_cdg(document)
    with pytest.raises(ValueError):_artifact_document_to_cdg(document,version_id='absent')


def test_execution_envelope_cannot_silently_downgrade_to_legacy():
    with pytest.raises(ValueError,match='envelope'):
        _artifact_document_to_cdg({'cdg_nodes':[{'node_id':'n','name':'broken','type_signature':'broken JSON'}]},require_execution_envelope=True)


def test_versioned_nodes_cannot_absorb_unversioned_edges():
    document={'cdg_nodes':[{'version_id':'new','node_id':'n'}],'cdg_edges':[{'source_id':'n','target_id':'n'}]}
    with pytest.raises(ValueError,match='version'):
        _artifact_document_to_cdg(document,version_id='new')
