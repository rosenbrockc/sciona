"""Explicit frequency-from-period execution graph for source-derivation validation."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus

PRIMITIVE='sciona.atoms.physical_quantities.period_frequency.frequency_from_period'


def build_period_frequency_execution(*,source_version_id,replay_evidence_id):
    return CDGExport(nodes=[AlgorithmicNode(node_id='frequency',name='Convert period to ordinary frequency',
        description='A positive cycle duration in seconds determines ordinary frequency in hertz; no period estimation.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='period_seconds',type_desc='numpy.ndarray',constraints='finite positive real float64-compatible seconds',dim_signature='T1')],
        outputs=[IOSpec(name='frequency_hz',type_desc='numpy.ndarray',constraints='finite positive hertz; same shape as input',dim_signature='T-1')],
    )],edges=[],metadata={'artifact_source':'physics_numerical_realization','publication_status':'draft',
        'source_version_id':str(source_version_id),'source_replay_evidence_id':str(replay_evidence_id),
        'physical_regime':'positive, constant duration of a repeating cycle','num_nodes':1,'num_edges':0})
